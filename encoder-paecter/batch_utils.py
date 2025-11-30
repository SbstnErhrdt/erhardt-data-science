from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple


LOGGER = logging.getLogger(__name__)


def load_env_file(env_path: Path) -> None:
    """Load env vars from a .env style file without overriding existing values."""

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
    """Build a PostgreSQL DSN from common environment variables."""

    if (url := os.getenv("DATABASE_URL")):
        if "sslmode" not in url:
            delimiter = "&" if "?" in url else "?"
            url = f"{url}{delimiter}sslmode={os.getenv('SQL_SSLMODE', 'require')}"
        return url

    required_keys = ["SQL_HOST", "SQL_PORT", "SQL_DATABASE", "SQL_USER", "SQL_PASSWORD"]
    missing = [key for key in required_keys if not os.getenv(key)]
    if missing:
        raise RuntimeError("Missing required database environment variables: " + ", ".join(missing))

    dsn = (
        f"host={os.environ['SQL_HOST']} "
        f"port={os.environ['SQL_PORT']} "
        f"dbname={os.environ['SQL_DATABASE']} "
        f"user={os.environ['SQL_USER']} "
        f"password={os.environ['SQL_PASSWORD']} "
        f"sslmode={os.getenv('SQL_SSLMODE', 'require')}"
    )
    if sslrootcert := os.getenv("SQL_SSLROOTCERT"):
        dsn = f"{dsn} sslrootcert={sslrootcert}"
    return dsn


def default_date_string() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def resolve_run_dirs(date_str: str, base_dir: Path | None = None) -> Tuple[Path, Path]:
    root = (base_dir or Path.cwd()) / "to_process" / date_str
    to_encode = root / "to_encode"
    to_upload = root / "to_upload"
    return to_encode, to_upload
