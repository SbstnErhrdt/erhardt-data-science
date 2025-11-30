from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List

import psycopg2
import pyarrow as pa
import pyarrow.parquet as pq

from batch_utils import default_date_string, get_database_dsn, load_env_file, resolve_run_dirs


LOGGER = logging.getLogger(__name__)

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
"""


def approximate_row_size_bytes(title: str, abstract: str) -> int:
    # Lightweight size estimate to keep parquet shards near the target size.
    return len(title.encode("utf-8")) + len(abstract.encode("utf-8")) + 32


def write_parquet(rows: List[Dict[str, object]], file_index: int, output_dir: Path, schema: pa.schema) -> Path:
    table = pa.Table.from_pylist(rows, schema=schema)
    output_path = output_dir / f"{file_index}.parquet"
    pq.write_table(table, output_path, compression="snappy")
    return output_path


def stream_pending(cursor, fetch_size: int):
    while True:
        batch = cursor.fetchmany(fetch_size)
        if not batch:
            break
        yield batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Download patent families pending embedding into parquet shards.")
    parser.add_argument("--date", help="Date folder to write under (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--env-file", help="Path to .env file. Defaults to encoder-paecter/.env.")
    parser.add_argument("--output-root", help="Optional custom root path for to_process output.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    date_str = args.date or default_date_string()
    env_path = Path(args.env_file) if args.env_file else Path(__file__).resolve().parent / ".env"
    load_env_file(env_path)

    base_dir = Path(args.output_root) if args.output_root else Path.cwd()
    to_encode_dir, _ = resolve_run_dirs(date_str, base_dir=base_dir)
    to_encode_dir.mkdir(parents=True, exist_ok=True)

    target_bytes = int(float(os.getenv("PAECTER_PARQUET_TARGET_MB", "100")) * 1024 * 1024)
    fetch_size = int(os.getenv("PAECTER_DOWNLOAD_FETCH", "5000"))

    LOGGER.info(
        "Starting download for %s into %s (target shard %.1f MB, fetch size %d)",
        date_str,
        to_encode_dir,
        target_bytes / (1024 * 1024),
        fetch_size,
    )

    dsn = get_database_dsn()
    schema = pa.schema(
        [
            ("family_id", pa.int64()),
            ("title", pa.string()),
            ("abstract", pa.string()),
        ]
    )

    file_index = 0
    total_rows = 0
    shard_bytes = 0
    buffer: List[Dict[str, object]] = []

    with psycopg2.connect(dsn) as conn:
        with conn.cursor(name="pending_family_download") as cursor:
            cursor.itersize = fetch_size
            cursor.execute(PENDING_FAMILY_QUERY)

            for rows in stream_pending(cursor, fetch_size):
                for family_id, title, abstract in rows:
                    if title is None or abstract is None:
                        continue
                    row_size = approximate_row_size_bytes(title, abstract)
                    if buffer and shard_bytes + row_size >= target_bytes:
                        file_index += 1
                        output_path = write_parquet(buffer, file_index, to_encode_dir, schema)
                        LOGGER.info(
                            "Wrote shard %s with %d rows (approx %.1f MB)",
                            output_path.name,
                            len(buffer),
                            shard_bytes / (1024 * 1024),
                        )
                        buffer = []
                        shard_bytes = 0

                    buffer.append({"family_id": int(family_id), "title": title, "abstract": abstract})
                    shard_bytes += row_size
                    total_rows += 1

            if buffer:
                file_index += 1
                output_path = write_parquet(buffer, file_index, to_encode_dir, schema)
                LOGGER.info(
                    "Wrote shard %s with %d rows (approx %.1f MB)",
                    output_path.name,
                    len(buffer),
                    shard_bytes / (1024 * 1024),
                )

    if total_rows == 0:
        LOGGER.info("No pending families found.")
    else:
        LOGGER.info("Finished download: %d rows across %d shard(s) in %s", total_rows, file_index, to_encode_dir)


if __name__ == "__main__":
    main()
