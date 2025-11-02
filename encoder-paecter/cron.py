"""Cron job script for generating PAECTER embeddings for patent families."""
from __future__ import annotations

import logging
import os
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
        f"password={os.environ['SQL_PASSWORD']}"
    )


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

    fetch_limit = int(os.getenv("PAECTER_FETCH_LIMIT", "256"))
    batch_size = int(os.getenv("PAECTER_BATCH_SIZE", "16"))
    max_length = int(os.getenv("PAECTER_MAX_LENGTH", "512"))

    LOGGER.info("Connecting to database")
    dsn = get_database_dsn()
    total_processed = 0
    try:
        with psycopg2.connect(dsn) as conn:
            LOGGER.info("Database connection established")
            LOGGER.info(
                "Starting background fetcher (limit %d, batch size %d)",
                fetch_limit,
                batch_size,
            )
            with PendingFamilyStream(dsn, fetch_limit, batch_size) as stream:
                iterator = iter(stream)
                query_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                query_end = torch.cuda.Event(enable_timing=True) if query_start else None
                cpu_query_start = torch.cuda.Event(enable_timing=False) if query_start else None
                try:
                    first_chunk = next(iterator)
                except StopIteration:
                    LOGGER.info("No pending patent families found. Nothing to do.")
                    return

                LOGGER.info(
                    "Pending families detected (initial chunk size: %d); loading encoder",
                    len(first_chunk),
                )

                tokenizer = AutoTokenizer.from_pretrained("mpi-inno-comp/paecter")
                model = AutoModel.from_pretrained("mpi-inno-comp/paecter")
                model.eval()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                LOGGER.info("Encoder ready; using device: %s", device)

                all_chunks = chain([first_chunk], iterator)
                with conn.cursor() as cursor:
                    for index, chunk in enumerate(all_chunks, start=1):
                        if not chunk:
                            continue
                        LOGGER.info(
                            "Encoding chunk %d with %d families (IDs %s-%s)",
                            index,
                            len(chunk),
                            chunk[0].family_id,
                            chunk[-1].family_id,
                        )
                        embeddings = encode_families(chunk, tokenizer, model, device, max_length)
                        insert_embeddings(cursor, chunk, embeddings)
                        conn.commit()
                        total_processed += len(chunk)
                        LOGGER.info(
                            "Stored embeddings for chunk %d (processed %d families so far)",
                            index,
                            total_processed,
                        )
    except Exception:
        LOGGER.exception("Embedding generation failed")
        raise

    LOGGER.info("Embedding generation finished; processed %d families.", total_processed)


if __name__ == "__main__":
    main()
