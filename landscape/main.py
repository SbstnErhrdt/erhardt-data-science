import asyncio
import logging
import os
import signal
from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable, Mapping, List

from dotenv import load_dotenv
from realtime import AsyncRealtimeClient, RealtimePostgresChangesListenEvent
from supabase import create_client, client as supa_client

from landscape import process

# ──────────────────────────────────────────────────────────────────────────────
# Configuration & Logging
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger("ip_landscapes_worker")

load_dotenv()  # load .env if present


@dataclass(frozen=True)
class Settings:
    supabase_url: str
    supabase_key: str
    table: str = "ip_landscapes"
    channel: str = "ip_landscapes"

    @staticmethod
    def from_env() -> "Settings":
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            missing = ["SUPABASE_URL" if not url else None, "SUPABASE_KEY" if not key else None]
            missing = ", ".join(m for m in missing if m)
            raise RuntimeError(f"Missing required environment variables: {missing}")
        return Settings(supabase_url=url, supabase_key=key)


SETTINGS = Settings.from_env()

# Global Supabase client (lightweight wrapper)
supabase: supa_client = create_client(
    supabase_url=SETTINGS.supabase_url,
    supabase_key=SETTINGS.supabase_key,
)


# ──────────────────────────────────────────────────────────────────────────────
# Database helpers
# ──────────────────────────────────────────────────────────────────────────────

async def _run_in_threadpool(fn: Callable, *args, **kwargs):
    """Run a blocking SDK call in a default executor to avoid blocking the loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))


async def update_row(uid: str, fields: Mapping[str, Any]) -> bool:
    """
    Update a single row by uid with provided fields.
    Returns True if any row was updated.
    """

    def _do_update():
        return (
            supabase.table(SETTINGS.table)
            .update(dict(fields))
            .eq("uid", uid)
            .execute()
        )

    try:
        resp = await _run_in_threadpool(_do_update)
        updated = bool(resp.data)
        if updated:
            logger.info("Updated %s for uid=%s", list(fields.keys()), uid)
        else:
            logger.warning("No row updated for uid=%s (may not exist)", uid)
        return updated
    except Exception as e:
        logger.exception("Failed to update row uid=%s: %s", uid, e)
        return False


async def send_status_update(uid: str, status: str, progress: int, extra: Optional[Dict[str, Any]] = None) -> None:
    """
    Update the process status/progress plus any extra fields.
    """
    payload: Dict[str, Any] = {"_status": status, "_progress": progress}
    if extra:
        payload.update(extra)
    await update_row(uid, payload)


async def send_results(uid: str, result: Dict[str, Any]) -> None:
    """
    Mark the job done and attach results in "_output".
    """
    await send_status_update(uid, "done", 100, {"_output": result})


# ──────────────────────────────────────────────────────────────────────────────
# Realtime handling
# ──────────────────────────────────────────────────────────────────────────────

def _extract_uid(payload: Mapping[str, Any]) -> Optional[str]:
    """
    Defensive extraction of UID from the Realtime payload.
    """
    try:
        return payload["data"]["record"]["uid"]
    except Exception:
        return None


def _extract_record(payload: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        return payload["data"]["record"]
    except Exception:
        return None


async def process_record(uid: str, record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Core business logic for handling a record.
    Validates input, delegates to process.run, and returns the result dict for `_output`.
    """
    if "search_vectors" not in record:
        raise ValueError("Missing 'search_vectors' in record")

    search_vectors = process.parse(record["search_vectors"])
    client_uid = record.get("client_uid")
    if not client_uid:
        raise ValueError("Missing 'client_uid' in record")

    async def _status(uid_: str, status: str, progress: int, extra: Optional[Dict[str, Any]] = None):
        await send_status_update(uid_, status, progress, extra)

    result = await process.run(uid, client_uid, search_vectors, supabase, _status)
    return result



def make_callback() -> Callable[[Dict[str, Any]], None]:
    """
    Wraps the async handler into a sync callback compatible with the realtime client.
    """

    def handle_callback(payload: Dict[str, Any]) -> None:
        # Fire-and-forget: schedule async work on the running loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # If no loop, log and bail (shouldn't happen once started)
            logger.error("No running event loop for callback; payload dropped")
            return

        loop.create_task(_handle_payload(payload))

    return handle_callback


async def _handle_payload(payload: Dict[str, Any]) -> None:
    logger.debug("Received payload: %s", payload)

    uid = _extract_uid(payload)
    if not uid:
        logger.error("Payload missing uid: %s", payload)
        return

    record = _extract_record(payload)
    if not record:
        logger.error("Payload missing record for uid=%s", uid)
        await send_status_update(uid, "error", 100)
        return

    try:
        # Example: send an initial status if desired
        # await send_status_update(uid, "processing", 5)

        result = await process_record(uid, record)
        await send_results(uid, result)

    except Exception as e:
        logger.exception("Error processing uid=%s: %s", uid, e)
        await send_status_update(
            uid,
            "error",
            100,
            {
                "_message": str(e)
            }
        )


class RealtimeWorker:
    """
    Manages realtime connection, subscription, and lifecycle.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncRealtimeClient(
            f"{settings.supabase_url}/realtime/v1",
            settings.supabase_key,
            auto_reconnect=True,  # let the client try to reconnect automatically
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
            callback=make_callback,
        ).subscribe()

        logger.info("Subscribed to %s (filter: _status=eq.to_calculate)", self.settings.table)

        # NOTE:
        # If your client warns that `.listen()` is deprecated, you can:
        #   1) Use a simple wait loop on an Event to keep the task alive, OR
        #   2) If the library offers a replacement (e.g., `.run()`), use that.
        #
        # We'll keep the task alive with an Event to avoid relying on deprecated APIs.
        await self._stop_event.wait()

    async def stop(self) -> None:
        logger.info("Shutting down…")
        self._stop_event.set()
        try:
            if self._channel:
                await self._channel.unsubscribe()
        except Exception as e:
            logger.warning("Error unsubscribing channel: %s", e)
        try:
            await self.client.disconnect()
        except Exception as e:
            logger.warning("Error disconnecting realtime client: %s", e)
        logger.info("Shutdown complete")


# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    worker = RealtimeWorker(SETTINGS)

    loop = asyncio.get_running_loop()
    stop_signals = (signal.SIGINT, signal.SIGTERM)

    def _handle_sig():
        asyncio.create_task(worker.stop())

    for sig in stop_signals:
        try:
            loop.add_signal_handler(sig, _handle_sig)
        except NotImplementedError:
            # Signal handling may not be available on some platforms (e.g., Windows)
            pass

    # Start and run until stopped
    await worker.start()


if __name__ == "__main__":
    # Prefer asyncio.run for modern, clean event loop management
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Graceful exit on Ctrl+C
        pass
