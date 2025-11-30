from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModel, AutoTokenizer

from batch_utils import default_date_string, load_env_file, resolve_run_dirs


LOGGER = logging.getLogger(__name__)


class PatentFamily:
    def __init__(self, family_id: int, title: str, abstract: str):
        self.family_id = int(family_id)
        self.title = title
        self.abstract = abstract

    def combined_text(self) -> str:
        return f"{self.title.strip()}\n\n{self.abstract.strip()}"


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        index = int(os.getenv("PAECTER_CUDA_DEVICE", "0"))
        index = max(min(index, torch.cuda.device_count() - 1), 0)
        return torch.device(f"cuda:{index}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def format_embedding(embedding: np.ndarray) -> str:
    return "[" + ",".join(f"{float(x):.9g}" for x in embedding.tolist()) + "]"


def flush_sql_rows(handle, pending_rows: List[str], chunk_size: int, *, force_all: bool = False) -> None:
    while len(pending_rows) >= chunk_size:
        chunk = pending_rows[:chunk_size]
        del pending_rows[:chunk_size]
        handle.write("INSERT INTO export_embeddings (docdb_family_id, embedding)\nVALUES\n    ")
        handle.write(",\n    ".join(chunk))
        handle.write("\nON CONFLICT (docdb_family_id) DO NOTHING;\n\n")

    if force_all and pending_rows:
        chunk = pending_rows[:]
        pending_rows.clear()
        handle.write("INSERT INTO export_embeddings (docdb_family_id, embedding)\nVALUES\n    ")
        handle.write(",\n    ".join(chunk))
        handle.write("\nON CONFLICT (docdb_family_id) DO NOTHING;\n\n")


def process_parquet_file(
    parquet_path: Path,
    output_path: Path,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    max_length: int,
    encode_batch_size: int,
    parquet_batch_size: int,
    insert_batch_size: int,
) -> int:
    pf = pq.ParquetFile(parquet_path)
    total_rows = 0

    with output_path.open("w") as handle:
        pending_rows: List[str] = []
        for record_batch in pf.iter_batches(batch_size=parquet_batch_size):
            data = record_batch.to_pydict()
            families: List[PatentFamily] = [
                PatentFamily(fid, title, abstract)
                for fid, title, abstract in zip(
                    data["family_id"],
                    data["title"],
                    data["abstract"],
                )
            ]

            for start in range(0, len(families), encode_batch_size):
                batch = families[start : start + encode_batch_size]
                embeddings = encode_families(batch, tokenizer, model, device, max_length)
                for family, embedding in zip(batch, embeddings):
                    pending_rows.append(f"({family.family_id}, '{format_embedding(embedding)}')")
                    total_rows += 1
                flush_sql_rows(handle, pending_rows, insert_batch_size)

        if pending_rows:
            flush_sql_rows(handle, pending_rows, insert_batch_size, force_all=True)

    return total_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode patent families from parquet shards and write SQL insert statements."
    )
    parser.add_argument("--date", help="Date folder to read from (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--env-file", help="Path to .env file. Defaults to encoder-paecter/.env.")
    parser.add_argument("--output-root", help="Optional custom root path for to_process output.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    date_str = args.date or default_date_string()
    env_path = Path(args.env_file) if args.env_file else Path(__file__).resolve().parent / ".env"
    load_env_file(env_path)

    base_dir = Path(args.output_root) if args.output_root else Path.cwd()
    to_encode_dir, to_upload_dir = resolve_run_dirs(date_str, base_dir=base_dir)
    to_upload_dir.mkdir(parents=True, exist_ok=True)

    device = detect_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    encode_batch_size = int(os.getenv("PAECTER_BATCH_SIZE", "32"))
    max_length = int(os.getenv("PAECTER_MAX_LENGTH", "512"))
    parquet_batch_size = int(os.getenv("PAECTER_PARQUET_BATCH_SIZE", "512"))
    insert_batch_size = int(os.getenv("PAECTER_SQL_INSERT_BATCH", "1000"))

    LOGGER.info(
        "Encoding from %s into %s (device=%s, encode batch=%d, parquet batch=%d, insert batch=%d)",
        to_encode_dir,
        to_upload_dir,
        device,
        encode_batch_size,
        parquet_batch_size,
        insert_batch_size,
    )

    tokenizer = AutoTokenizer.from_pretrained("mpi-inno-comp/paecter")
    model_kwargs = {"torch_dtype": dtype}
    model = AutoModel.from_pretrained("mpi-inno-comp/paecter", **model_kwargs)
    model.to(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    parquet_files = sorted(to_encode_dir.glob("*.parquet"))
    if not parquet_files:
        LOGGER.info("No parquet files found in %s", to_encode_dir)
        return

    total_rows = 0
    for parquet_path in parquet_files:
        output_path = to_upload_dir / (parquet_path.stem + ".sql")
        LOGGER.info("Encoding %s -> %s", parquet_path.name, output_path.name)
        rows_written = process_parquet_file(
            parquet_path,
            output_path,
            tokenizer,
            model,
            device,
            max_length,
            encode_batch_size,
            parquet_batch_size,
            insert_batch_size,
        )
        total_rows += rows_written
        parquet_path.unlink()
        LOGGER.info("Finished %s (%d rows). Removed source shard.", parquet_path.name, rows_written)

    LOGGER.info("Encoding complete: %d total rows written to %s", total_rows, to_upload_dir)


if __name__ == "__main__":
    main()
