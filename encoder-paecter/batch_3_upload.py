from __future__ import annotations

import argparse
import logging
from pathlib import Path

import psycopg2

from batch_utils import default_date_string, get_database_dsn, load_env_file, resolve_run_dirs


LOGGER = logging.getLogger(__name__)


def execute_sql_file(conn, sql_path: Path) -> int:
    sql_text = sql_path.read_text()
    with conn.cursor() as cursor:
        cursor.execute(sql_text)
    conn.commit()
    # Count statements by insert batches to give progress feedback.
    return sql_text.count("INSERT INTO")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload encoded embeddings from SQL files.")
    parser.add_argument("--date", help="Date folder to read from (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--env-file", help="Path to .env file. Defaults to encoder-paecter/.env.")
    parser.add_argument("--output-root", help="Optional custom root path for to_process output.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    date_str = args.date or default_date_string()
    env_path = Path(args.env_file) if args.env_file else Path(__file__).resolve().parent / ".env"
    load_env_file(env_path)

    base_dir = Path(args.output_root) if args.output_root else Path.cwd()
    _, to_upload_dir = resolve_run_dirs(date_str, base_dir=base_dir)

    sql_files = sorted(to_upload_dir.glob("*.sql"))
    if not sql_files:
        LOGGER.info("No SQL files found in %s", to_upload_dir)
        return

    dsn = get_database_dsn()
    LOGGER.info("Uploading %d SQL shard(s) from %s", len(sql_files), to_upload_dir)

    with psycopg2.connect(dsn) as conn:
        for sql_path in sql_files:
            LOGGER.info("Executing %s", sql_path.name)
            try:
                inserts = execute_sql_file(conn, sql_path)
            except Exception:
                LOGGER.exception("Failed to execute %s; leaving file in place for inspection.", sql_path.name)
                raise
            sql_path.unlink()
            LOGGER.info("Finished %s (%d insert batches). Removed source file.", sql_path.name, inserts)


if __name__ == "__main__":
    main()
