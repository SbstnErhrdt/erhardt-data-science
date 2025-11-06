"""Cron job script for generating PAECTER embeddings for patent families."""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import List, Optional, Sequence, Tuple

import numpy as np
import psycopg2
from psycopg2.extras import execute_values
import torch
from transformers import AutoModel, AutoTokenizer


LOGGER = logging.getLogger(__name__)


@dataclass
class PatentFamily:
    family_id: int
    title: str
    abstract: str

    def combined_text(self) -> str:
        return f"{self.title.strip()}\n\n{self.abstract.strip()}"


@dataclass(frozen=True)
class RuntimeConfig:
    device: torch.device
    device_description: str
    batch_size: int
    fetch_limit: int
    dtype: torch.dtype
    auto_batch_size: bool
    auto_fetch_limit: bool


def detect_compute_device() -> Tuple[torch.device, str]:
    """Determine the most capable execution device available."""

    if torch.cuda.is_available():
        try:
            requested_index = int(os.getenv("PAECTER_CUDA_DEVICE", "0"))
        except ValueError:
            LOGGER.warning("Invalid PAECTER_CUDA_DEVICE value; defaulting to 0.")
            requested_index = 0

        device_count = torch.cuda.device_count()
        if requested_index < 0 or requested_index >= device_count:
            LOGGER.warning(
                "Requested CUDA device %d out of range (total %d). Falling back to device 0.",
                requested_index,
                device_count,
            )
            requested_index = 0

        device = torch.device(f"cuda:{requested_index}")
        props = torch.cuda.get_device_properties(requested_index)
        memory_gib = props.total_memory / (1024**3)
        description = (
            f"{torch.cuda.get_device_name(requested_index)} "
            f"({memory_gib:.0f} GiB, compute {props.major}.{props.minor})"
        )
        return device, description

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "Apple MPS"

    return torch.device("cpu"), "CPU"


def recommend_batch_size(device: torch.device) -> int:
    """Pick a conservative batch size based on available accelerator memory."""

    if device.type == "cuda":
        index = device.index or 0
        props = torch.cuda.get_device_properties(index)
        memory_gib = props.total_memory / (1024**3)

        if memory_gib >= 120:
            return 64
        if memory_gib >= 80:
            return 56
        if memory_gib >= 48:
            return 40
        if memory_gib >= 24:
            return 32
        if memory_gib >= 16:
            return 24
        return 16

    if device.type == "mps":
        return 24

    return 16


def resolve_batch_size(device: torch.device) -> Tuple[int, bool]:
    raw = os.getenv("PAECTER_BATCH_SIZE")
    if raw is None:
        return recommend_batch_size(device), True

    cleaned = raw.strip().lower()
    if cleaned == "auto":
        return recommend_batch_size(device), True

    try:
        value = int(cleaned)
    except ValueError as exc:
        raise ValueError(
            f"PAECTER_BATCH_SIZE must be an integer or 'auto'; received {raw!r}."
        ) from exc

    if value <= 0:
        raise ValueError("PAECTER_BATCH_SIZE must be positive.")
    return value, False


def resolve_fetch_limit(batch_size: int) -> Tuple[int, bool]:
    raw = os.getenv("PAECTER_FETCH_LIMIT")
    multiplier_raw = os.getenv("PAECTER_AUTO_FETCH_MULTIPLIER", "8")
    try:
        multiplier = max(int(multiplier_raw), 1)
    except ValueError:
        LOGGER.warning(
            "Invalid PAECTER_AUTO_FETCH_MULTIPLIER value %r; defaulting to 8.",
            multiplier_raw,
        )
        multiplier = 8

    if raw is None:
        return max(batch_size * multiplier, batch_size), True

    cleaned = raw.strip().lower()
    if cleaned == "auto":
        return max(batch_size * multiplier, batch_size), True

    try:
        value = int(cleaned)
    except ValueError as exc:
        raise ValueError(
            f"PAECTER_FETCH_LIMIT must be an integer or 'auto'; received {raw!r}."
        ) from exc

    if value < batch_size:
        LOGGER.warning(
            "Fetch limit %d is smaller than batch size %d; increasing fetch limit to match.",
            value,
            batch_size,
        )
        value = batch_size
    return value, False


def resolve_model_dtype(device: torch.device) -> torch.dtype:
    raw = os.getenv("PAECTER_MODEL_DTYPE")
    if raw:
        cleaned = raw.strip().lower()
        dtype_map = {
            "fp32": torch.float32,
            "float32": torch.float32,
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
        }
        if cleaned not in dtype_map:
            raise ValueError(
                f"Unsupported PAECTER_MODEL_DTYPE value {raw!r}; "
                "expected one of fp32, fp16, bf16, float32, float16, bfloat16."
            )
        dtype = dtype_map[cleaned]
    else:
        dtype = torch.float16 if device.type == "cuda" else torch.float32

    if device.type != "cuda" and dtype == torch.float16:
        LOGGER.warning("Float16 requested on non-CUDA device; using float32 instead.")
        dtype = torch.float32

    if device.type == "mps" and dtype == torch.bfloat16:
        LOGGER.warning("bfloat16 not supported on MPS; using float32 instead.")
        dtype = torch.float32

    return dtype


def prepare_runtime_config() -> RuntimeConfig:
    device, description = detect_compute_device()
    batch_size, auto_batch = resolve_batch_size(device)
    fetch_limit, auto_fetch = resolve_fetch_limit(batch_size)
    dtype = resolve_model_dtype(device)

    LOGGER.info(
        "Runtime configuration resolved: device=%s, dtype=%s, batch_size=%d%s, fetch_limit=%d%s",
        description,
        str(dtype).replace("torch.", ""),
        batch_size,
        " (auto)" if auto_batch else "",
        fetch_limit,
        " (auto)" if auto_fetch else "",
    )

    return RuntimeConfig(
        device=device,
        device_description=description,
        batch_size=batch_size,
        fetch_limit=fetch_limit,
        dtype=dtype,
        auto_batch_size=auto_batch,
        auto_fetch_limit=auto_fetch,
    )


def load_env_file(env_path: Path) -> None:
    """Load environment variables from a simple ``.env`` style file.

    The parser understands ``KEY=value`` pairs, ignores comments and whitespace,
    and does not override values that are already defined in the current
    environment.
    """

    if not env_path.exists():
        LOGGER.warning("Environment file %s not found; relying on existing env vars.", env_path)
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)


def get_database_dsn() -> str:
    if (url := os.getenv("DATABASE_URL")):
        if "sslmode" not in url:
            delimiter = "&" if "?" in url else "?"
            url = f"{url}{delimiter}sslmode={os.getenv('SQL_SSLMODE', 'require')}"
        return url

    required_keys = ["SQL_HOST", "SQL_PORT", "SQL_DATABASE", "SQL_USER", "SQL_PASSWORD"]
    missing = [key for key in required_keys if not os.getenv(key)]
    if missing:
        raise RuntimeError(
            "Missing required database environment variables: " + ", ".join(missing)
        )

    return (
        f"host={os.environ['SQL_HOST']} "
        f"port={os.environ['SQL_PORT']} "
        f"dbname={os.environ['SQL_DATABASE']} "
        f"user={os.environ['SQL_USER']} "
        f"password={os.environ['SQL_PASSWORD']} "
        f"sslmode={os.getenv('SQL_SSLMODE', 'require')}"
    )

    if sslrootcert := os.getenv("SQL_SSLROOTCERT"):
        return dsn + f" sslrootcert={sslrootcert}"

    return dsn


PENDING_FAMILY_QUERY = """
WITH filtered AS (
    SELECT family_id::bigint AS family_id,
           title,
           abstract
    FROM epo_doc_db.mv_patent_family
    WHERE family_id ~ '^[0-9]+$'
      AND title IS NOT NULL
      AND btrim(title) <> ''
      AND abstract IS NOT NULL
      AND btrim(abstract) <> ''
)
SELECT f.family_id,
       f.title,
       f.abstract
FROM filtered f
WHERE NOT EXISTS (
    SELECT 1
    FROM export_embeddings e
    WHERE e.docdb_family_id = f.family_id
)
ORDER BY f.family_id
LIMIT %s
"""

DB_WRITE_MAX_RETRIES = int(os.getenv("PAECTER_DB_WRITE_RETRIES", "5"))
DB_WRITE_RETRY_BACKOFF = float(os.getenv("PAECTER_DB_WRITE_BACKOFF", "2.0"))


class PendingFamilyStream:
    """Background prefetcher that streams patent families awaiting embeddings."""

    def __init__(self, dsn: str, fetch_limit: int, batch_size: int):
        self._dsn = dsn
        self._fetch_limit = max(fetch_limit, 0)
        self._batch_size = max(batch_size, 1)
        self._queue: Queue = Queue(maxsize=4)
        self._sentinel = object()
        self._stop_event = Event()
        self._error: Optional[BaseException] = None
        self._thread = Thread(
            target=self._run,
            name="pending-family-fetcher",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        if self._fetch_limit == 0:
            self._put(self._sentinel)
            return

        produced = 0
        try:
            with psycopg2.connect(self._dsn) as conn:
                with conn.cursor(name="pending_families") as cursor:
                    cursor.itersize = self._batch_size
                    cursor.execute(PENDING_FAMILY_QUERY, (self._fetch_limit,))
                    while not self._stop_event.is_set():
                        rows = cursor.fetchmany(self._batch_size)
                        if not rows:
                            break
                        families = [
                            PatentFamily(family_id=row[0], title=row[1], abstract=row[2])
                            for row in rows
                        ]
                        self._put(families)
                        produced += len(families)
                        if produced >= self._fetch_limit:
                            break
        except BaseException as exc:  # noqa: BLE001 - propagate to main thread
            self._error = exc
        finally:
            self._put(self._sentinel)

    def _put(self, item) -> None:
        while True:
            try:
                self._queue.put(item, timeout=1)
                return
            except Full:
                if self._stop_event.is_set():
                    return

    def close(self) -> None:
        self._stop_event.set()
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                break
        self._put(self._sentinel)
        self._thread.join(timeout=5)

    def __iter__(self) -> "PendingFamilyStream":
        return self

    def __next__(self) -> List[PatentFamily]:
        chunk = self._queue.get()
        if chunk is self._sentinel:
            self._thread.join(timeout=5)
            if self._error:
                raise RuntimeError("Failed to fetch pending patent families") from self._error
            raise StopIteration
        return chunk

    def __enter__(self) -> "PendingFamilyStream":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def encode_families(
    families: Sequence[PatentFamily],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    max_length: int,
) -> List[np.ndarray]:
    texts = [family.combined_text() for family in families]
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=max_length,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        model_output = model(**encoded)

    pooled = mean_pooling(model_output, encoded["attention_mask"]).cpu()
    return [embedding.numpy().astype(np.float32) for embedding in pooled]


def insert_embeddings(cursor, families: Sequence[PatentFamily], embeddings: Sequence[np.ndarray]) -> None:
    records: List[Tuple[int, List[float]]] = [
        (family.family_id, embedding.tolist()) for family, embedding in zip(families, embeddings)
    ]
    LOGGER.info("Inserting embeddings for %d families", len(records))
    execute_values(
        cursor,
        """
        INSERT INTO export_embeddings (docdb_family_id, embedding)
        VALUES %s
        """,
        records,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    root_dir = Path(__file__).resolve().parent
    env_path = Path(os.getenv("PAECTER_ENV_FILE", root_dir / ".env"))
    LOGGER.info("Loading environment variables from %s", env_path)
    load_env_file(Path(env_path))

    max_length = int(os.getenv("PAECTER_MAX_LENGTH", "512"))
    runtime_config = prepare_runtime_config()
    batch_size = runtime_config.batch_size
    fetch_limit = runtime_config.fetch_limit

    LOGGER.info("Connecting to database")
    dsn = get_database_dsn()
    write_conn: Optional[psycopg2.extensions.connection] = None
    grand_total = 0
    chunk_counter = 0
    iteration_index = 0
    tokenizer: Optional[AutoTokenizer] = None
    model: Optional[AutoModel] = None
    try:
        LOGGER.info("Starting embedding loop with fetch limit %d and batch size %d", fetch_limit, batch_size)
        while True:
            iteration_index += 1
            iteration_processed = 0
            no_more_pending = False

            LOGGER.info(
                "Iteration %d: starting background fetcher (limit %d, batch size %d)",
                iteration_index,
                fetch_limit,
                batch_size,
            )

            with PendingFamilyStream(dsn, fetch_limit, batch_size) as stream:
                iterator = iter(stream)
                fetch_start = time.perf_counter()

                try:
                    first_chunk = next(iterator)
                except StopIteration:
                    if grand_total == 0:
                        LOGGER.info("No pending patent families found. Nothing to do.")
                    else:
                        LOGGER.info("No additional families found; stopping after iteration %d.", iteration_index)
                    no_more_pending = True
                else:
                    first_fetch_duration = time.perf_counter() - fetch_start
                    LOGGER.info(
                        "Iteration %d: first chunk contains %d families fetched in %.2fs",
                        iteration_index,
                        len(first_chunk),
                        first_fetch_duration,
                    )

                    if tokenizer is None or model is None:
                        encoder_load_start = time.perf_counter()
                        tokenizer = AutoTokenizer.from_pretrained("mpi-inno-comp/paecter")
                        model_kwargs = {}
                        if runtime_config.dtype is not None:
                            model_kwargs["torch_dtype"] = runtime_config.dtype
                        model = AutoModel.from_pretrained("mpi-inno-comp/paecter", **model_kwargs)
                        model.eval()
                        model.to(runtime_config.device)
                        if runtime_config.device.type == "cuda":
                            torch.cuda.set_device(runtime_config.device)
                            torch.backends.cuda.matmul.allow_tf32 = True
                            torch.backends.cudnn.benchmark = True
                        encoder_load_duration = time.perf_counter() - encoder_load_start
                        LOGGER.info(
                            "Encoder loaded on %s in %.2fs",
                            runtime_config.device_description,
                            encoder_load_duration,
                        )
                    else:
                        LOGGER.info(
                            "Encoder already loaded on %s; reusing.",
                            runtime_config.device_description,
                        )

                    fetch_start = time.perf_counter()
                    all_chunks = chain([first_chunk], iterator)

                    for local_index, chunk in enumerate(all_chunks, start=1):
                        if not chunk:
                            fetch_start = time.perf_counter()
                            continue

                        fetch_duration = (
                            first_fetch_duration
                            if (iteration_processed == 0 and local_index == 1)
                            else time.perf_counter() - fetch_start
                        )
                        global_chunk_number = chunk_counter + 1
                        LOGGER.info(
                            (
                                "Iteration %d, chunk %d (global %d): %d families "
                                "(IDs %s-%s) ready after %.2fs fetch wait"
                            ),
                            iteration_index,
                            local_index,
                            global_chunk_number,
                            len(chunk),
                            chunk[0].family_id,
                            chunk[-1].family_id,
                            fetch_duration,
                        )

                        encode_start = time.perf_counter()
                        embeddings = encode_families(
                            chunk,
                            tokenizer,
                            model,
                            runtime_config.device,
                            max_length,
                        )
                        encode_duration = time.perf_counter() - encode_start

                        db_start = time.perf_counter()
                        for attempt in range(1, DB_WRITE_MAX_RETRIES + 1):
                            try:
                                if write_conn is None or getattr(write_conn, "closed", 0):
                                    write_conn = psycopg2.connect(dsn)
                                    LOGGER.info(
                                        "Write connection (re)established on attempt %d",
                                        attempt,
                                    )
                                with write_conn.cursor() as cursor:
                                    insert_embeddings(cursor, chunk, embeddings)
                                write_conn.commit()
                                break
                            except (psycopg2.OperationalError, psycopg2.InterfaceError) as exc:
                                LOGGER.warning(
                                    "Database write failed on attempt %d/%d: %s",
                                    attempt,
                                    DB_WRITE_MAX_RETRIES,
                                    exc,
                                )
                                if write_conn and not getattr(write_conn, "closed", 0):
                                    try:
                                        write_conn.close()
                                    except Exception:
                                        LOGGER.debug("Error closing write connection after failure", exc_info=True)
                                write_conn = None
                                if attempt == DB_WRITE_MAX_RETRIES:
                                    raise
                                backoff = min(DB_WRITE_RETRY_BACKOFF ** attempt, 30)
                                LOGGER.info("Retrying in %.2fs", backoff)
                                time.sleep(backoff)
                        else:
                            raise RuntimeError("Exceeded maximum retries for database write")
                        db_duration = time.perf_counter() - db_start

                        iteration_processed += len(chunk)
                        chunk_counter += 1
                        grand_total += len(chunk)

                        LOGGER.info(
                            (
                                "Iteration %d, chunk %d (global %d) stored: encode %.2fs, "
                                "store %.2fs (iteration total %d, cumulative %d)"
                            ),
                            iteration_index,
                            local_index,
                            global_chunk_number,
                            encode_duration,
                            db_duration,
                            iteration_processed,
                            grand_total,
                        )

                        fetch_start = time.perf_counter()

            if no_more_pending:
                break

            if iteration_processed == 0:
                LOGGER.info("Iteration %d yielded no families; stopping.", iteration_index)
                break

            LOGGER.info(
                "Iteration %d completed; processed %d families (cumulative %d).",
                iteration_index,
                iteration_processed,
                grand_total,
            )

            if iteration_processed < fetch_limit:
                LOGGER.info(
                    "Iteration %d processed fewer than fetch limit (%d); assuming queue drained.",
                    iteration_index,
                    fetch_limit,
                )
                break
    except Exception:
        LOGGER.exception("Embedding generation failed")
        raise
    finally:
        if write_conn and not getattr(write_conn, "closed", 0):
            write_conn.close()

    LOGGER.info("Embedding generation finished; processed %d families in total.", grand_total)


def run_with_restarts() -> None:
    """Execute main with simple recovery behaviour for transient failures."""

    restart_delay = float(os.getenv("PAECTER_RESTART_DELAY", "60"))
    max_restarts_env = os.getenv("PAECTER_MAX_RESTARTS", 5)
    max_restarts: Optional[int] = int(max_restarts_env) if max_restarts_env else None

    attempt = 0
    while True:
        attempt += 1
        try:
            main()
        except KeyboardInterrupt:
            LOGGER.info("Interrupted by user; shutting down without restart.")
            break
        except Exception:
            LOGGER.exception("Run attempt %d failed; recoverable error detected.", attempt)
            if max_restarts is not None and attempt >= max_restarts:
                LOGGER.error(
                    "Reached maximum restarts (%d); aborting and propagating last error.",
                    max_restarts,
                )
                raise
            LOGGER.info(
                "Restarting main() after %.2fs delay (next attempt %d).",
                restart_delay,
                attempt + 1,
            )
            time.sleep(restart_delay)
            continue
        else:
            if attempt > 1:
                LOGGER.info("Run attempt %d succeeded after prior recoveries.", attempt)
            break


if __name__ == "__main__":
    run_with_restarts()
