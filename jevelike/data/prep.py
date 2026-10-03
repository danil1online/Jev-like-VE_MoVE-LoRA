"""
Prepare the ruwiki pretraining dataset: download (or read locally), split into
documents, shuffle, and write parquet shards. The LAST shard is the validation split.

Usage:
    prepare-ruwiki --data-dir <dir> [--num-shards 100] [--input local_file.txt]
"""
import os
import argparse
import random
import pyarrow as pa
import pyarrow.parquet as pq

from jevelike.common import download_file, print0

RUWIKI_URL = "https://huggingface.co/datasets/atBuba/ruwiki-dataset/resolve/main/ruwiki_full.txt"
SEP = "\n@@@\n"


def load_documents(input_path):
    print0(f"Reading {input_path} ...")
    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    docs = [d.strip() for d in content.split(SEP)]
    docs = [d for d in docs if d]
    print0(f"  {len(docs)} documents")
    return docs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=None, help="output dir (default: <base>/data)")
    ap.add_argument("--num-shards", type=int, default=100, help="number of TRAIN shards")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--input", default=None,
                    help="local ruwiki_full.txt (skips download)")
    ap.add_argument("--no-download", action="store_true",
                    help="fail instead of downloading when no input given")
    args = ap.parse_args()

    data_dir = args.data_dir or os.path.join(os.getcwd(), "data")
    os.makedirs(data_dir, exist_ok=True)

    input_path = args.input
    if input_path is None:
        if args.no_download:
            raise SystemExit("No --input given and --no-download set")
        input_path = os.path.join(data_dir, "ruwiki_full.txt")
        if not os.path.exists(input_path):
            print0(f"Downloading ruwiki dataset to {input_path} ...")
            download_file(RUWIKI_URL, input_path)
        else:
            print0(f"Using existing {input_path}")

    docs = load_documents(input_path)

    rng = random.Random(args.seed)
    rng.shuffle(docs)

    # ~1% held out for validation: n_val shards, at least 1
    n_val = max(1, args.num_shards // 100)
    total_shards = args.num_shards + n_val
    shard_size = len(docs) // total_shards
    print0(f"Writing {args.num_shards} train + {n_val} val shards "
           f"({shard_size} docs each) to {data_dir}")

    for i in range(total_shards):
        chunk = docs[i * shard_size:(i + 1) * shard_size]
        if i == total_shards - 1:  # last shard takes the remainder and is the val split
            chunk = docs[i * shard_size:]
        prefix = "val" if i >= args.num_shards else "train"
        path = os.path.join(data_dir, f"ruwiki_{prefix}_{i:04d}.parquet")
        table = pa.table({"text": chunk})
        pq.write_table(table, path)
    print0("Done.")


if __name__ == "__main__":
    main()
