import asyncio
import logging
import os
import signal
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional

from dotenv import load_dotenv
from realtime import AsyncRealtimeClient, RealtimePostgresChangesListenEvent
from supabase import create_client, client as supa_client

from landscape import process

# ──────────────────────────────────────────────────────────────────────────────
# Logging configuration
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("ip_landscapes_worker")


# ──────────────────────────────────────────────────────────────────────────────
# Settings & helpers
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Settings:
    supabase_url: str
    supabase_key: str
    table: str = "ip_landscapes"
    channel: str = "ip_landscapes"

    @staticmethod
    def from_env(env: Optional[Mapping[str, str]] = None) -> "Settings":
        env = env or os.environ
        url = env.get("SUPABASE_URL")
        key = env.get("SUPABASE_KEY")
        missing = [name for name, value in (("SUPABASE_URL", url), ("SUPABASE_KEY", key)) if not value]
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(f"Missing required environment variables: {joined}")
        return Settings(supabase_url=url, supabase_key=key)


SupabaseClient = supa_client.Client


async def _run_in_threadpool(fn: Callable, *args, **kwargs):
    """Run blocking Supabase SDK calls without stalling the event loop."""
    try:
        return await asyncio.to_thread(fn, *args, **kwargs)  # type: ignore[attr-defined]
    except AttributeError:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))


class SupabaseService:
    """Thin async wrapper around the Supabase Python client."""

    def __init__(self, client: SupabaseClient, table: str):
        self._client = client
        self._table = table

    @property
    def client(self) -> SupabaseClient:
        return self._client

    async def update_row(self, uid: str, fields: Mapping[str, Any]) -> bool:
        def _do_update():
            return (
                self._client.table(self._table)
                .update(dict(fields))
                .eq("uid", uid)
                .execute()
            )

        try:
            resp = await _run_in_threadpool(_do_update)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to update row uid=%s: %s", uid, exc)
            return False

        updated = bool(resp.data)
        if updated:
            logger.info("Updated %s for uid=%s", list(fields.keys()), uid)
        else:
            logger.warning("No row updated for uid=%s (may not exist)", uid)
        return updated

    async def send_status_update(
            self,
            uid: str,
            status: str,
            progress: int,
            extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {"_status": status, "_progress": progress}
        if extra:
            payload.update(extra)
        await self.update_row(uid, payload)

    async def send_results(self, uid: str, result: Dict[str, Any]) -> None:
        await self.send_status_update(uid, "done", 100, {"_output": result})


class LandscapeJobProcessor:
    """Validate and process incoming realtime payloads."""

    def __init__(self, supabase_service: SupabaseService):
        self._supabase = supabase_service

    async def __call__(self, payload: Mapping[str, Any]) -> None:
        logger.debug("Received payload: %s", payload)

        uid = self._extract_uid(payload)
        if not uid:
            logger.error("Payload missing uid: %s", payload)
            return

        record = self._extract_record(payload)
        if not record:
            logger.error("Payload missing record for uid=%s", uid)
            await self._supabase.send_status_update(uid, "error", 100, {"_message": "Missing record"})
            return

        try:
            result = await self._process(uid, record)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Error processing uid=%s: %s", uid, exc)
            await self._supabase.send_status_update(
                uid,
                "error",
                100,
                {"_message": str(exc)},
            )
            return

        await self._supabase.send_results(uid, result)

    async def _process(self, uid: str, record: Mapping[str, Any]) -> Dict[str, Any]:
        if "search_vectors" not in record:
            raise ValueError("Missing 'search_vectors' in record")

        search_vectors = process.parse(record["search_vectors"])
        client_uid = record.get("client_uid")
        if not client_uid:
            raise ValueError("Missing 'client_uid' in record")

        async def _status(status: str, progress: int, extra: Optional[Dict[str, Any]] = None) -> None:
            await self._supabase.send_status_update(uid, status, progress, extra)

        return await process.run(uid, client_uid, search_vectors, self._supabase.client, _status)

    @staticmethod
    def _extract_uid(payload: Mapping[str, Any]) -> Optional[str]:
        try:
            return payload["data"]["record"]["uid"]
        except Exception:
            return None

    @staticmethod
    def _extract_record(payload: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            return payload["data"]["record"]
        except Exception:
            return None


class RealtimeWorker:
    """Manage realtime connection, subscription, and graceful shutdown."""

    def __init__(
            self,
            settings: Settings,
            processor: Callable[[Mapping[str, Any]], Awaitable[None]],
    ) -> None:
        self.settings = settings
        self._processor = processor
        self.client = AsyncRealtimeClient(
            f"{settings.supabase_url}/realtime/v1",
            settings.supabase_key,
            auto_reconnect=True,
        )
        self._stop_event = asyncio.Event()
        self._channel = None

    async def start(self) -> None:
        await self.client.connect()
        logger.info("Realtime connected")

        self._channel = self.client.channel(self.settings.channel)

        await self._channel.on_postgres_changes(
            RealtimePostgresChangesListenEvent.Update,
            schema="public",
            table=self.settings.table,
            filter="_status=eq.to_calculate",
            callback=self._wrap_callback(self._processor),
        ).subscribe()

        logger.info("Subscribed to %s (filter: _status=eq.to_calculate)", self.settings.table)
        await self._stop_event.wait()

    async def stop(self) -> None:
        logger.info("Shutting down…")
        self._stop_event.set()
        try:
            if self._channel:
                await self._channel.unsubscribe()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Error unsubscribing channel: %s", exc)
        try:
            await self.client.disconnect()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Error disconnecting realtime client: %s", exc)
        logger.info("Shutdown complete")

    @staticmethod
    def _wrap_callback(
            handler: Callable[[Mapping[str, Any]], Awaitable[None]]
    ) -> Callable[[Mapping[str, Any]], None]:
        def _callback(payload: Mapping[str, Any]) -> None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                logger.error("No running event loop for callback; payload dropped")
                return
            loop.create_task(handler(payload))

        return _callback


# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────


def _create_worker(settings: Settings) -> RealtimeWorker:
    client = create_client(settings.supabase_url, settings.supabase_key)
    service = SupabaseService(client, settings.table)
    processor = LandscapeJobProcessor(service)
    return RealtimeWorker(settings, processor)


async def main(settings: Optional[Settings] = None) -> None:
    if settings is None:
        load_dotenv()
        settings = Settings.from_env()

    worker = _create_worker(settings)

    loop = asyncio.get_running_loop()
    stop_signals = (signal.SIGINT, signal.SIGTERM)

    def _handle_sig() -> None:
        asyncio.create_task(worker.stop())

    for sig in stop_signals:
        try:
            loop.add_signal_handler(sig, _handle_sig)
        except NotImplementedError:  # pragma: no cover - platform guard
            pass

    await worker.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
