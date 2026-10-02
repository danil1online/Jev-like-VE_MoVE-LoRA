"""
The base/pretraining dataset is a set of parquet files, each with a single 'text'
column. The last parquet file is the validation split, all others are train.
Prepared by jevelike/data/prep.py (ruwiki) or any compatible producer.
"""
import os
import pyarrow.parquet as pq


def list_parquet_files(data_dir):
    """Look into a data dir and return full paths to all parquet files (sorted)."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}. "
            f"Run `prepare-ruwiki --data-dir {data_dir}` first."
        )
    parquet_files = sorted(
        f for f in os.listdir(data_dir)
        if f.endswith(".parquet") and not f.endswith(".tmp")
    )
    return [os.path.join(data_dir, f) for f in parquet_files]


def parquets_iter_batched(data_dir, split="train", batch_rows=128):
    """
    Iterate through the dataset, yielding batches of row_group texts.
    - split: "train" (all but last parquet file) or "val" (last parquet file).
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files(data_dir)
    assert len(parquet_paths) > 0, f"No parquet files found in {data_dir}"
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column("text").to_pylist()
            for i in range(0, len(texts), batch_rows):
                yield texts[i:i + batch_rows]


def count_documents(data_dir, split="train"):
    n = 0
    for batch in parquets_iter_batched(data_dir, split, batch_rows=10**9):
        n += len(batch)
    return n
