"""Cron job script for generating PAECTER embeddings for patent families."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

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


def _chunked(iterable: Sequence[PatentFamily], chunk_size: int) -> Iterator[Sequence[PatentFamily]]:
    for start in range(0, len(iterable), chunk_size):
        yield iterable[start:start + chunk_size]


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


def fetch_pending_patent_families(cursor, limit: int) -> List[PatentFamily]:
    cursor.execute(
        """
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
        """,
        (limit,),
    )

    rows = cursor.fetchall()
    return [PatentFamily(family_id=row[0], title=row[1], abstract=row[2]) for row in rows]


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
    try:
        with psycopg2.connect(dsn) as conn:
            LOGGER.info("Database connection established")
            LOGGER.info("Checking for pending families (limit %d)", fetch_limit)
            with conn.cursor() as cursor:
                families = fetch_pending_patent_families(cursor, fetch_limit)

            if not families:
                LOGGER.info("No pending patent families found. Nothing to do.")
                return

            LOGGER.info("Found %d pending patent families; loading encoder", len(families))

            tokenizer = AutoTokenizer.from_pretrained("mpi-inno-comp/paecter")
            model = AutoModel.from_pretrained("mpi-inno-comp/paecter")
            model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            LOGGER.info("Encoder ready; using device: %s", device)

            with conn.cursor() as cursor:
                for index, chunk in enumerate(_chunked(families, batch_size), start=1):
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
                    LOGGER.info("Stored embeddings for chunk %d", index)
    except Exception:
        LOGGER.exception("Embedding generation failed")
        raise

    LOGGER.info("Embedding generation finished")


if __name__ == "__main__":
    main()
