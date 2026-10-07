"""
Prepare a pretraining dataset from a HuggingFace parquet-hosted dataset
(e.g. HuggingFaceFW/fineweb-edu) into the jevelike shard layout:
<data_dir>/hf_train_####.parquet with the trailing hf_val_####.parquet files as
the validation split (single 'text' column; sorted order puts val last).

Flow: enumerate parquet files via the HF datasets-server API (fallback: Hub tree
API), download them into a local cache (resumable, stops at --max-tokens using
a chars/token estimate), then rewrite the cached files into balanced shards with
a seeded shuffle buffer. A VAL_SHA256.txt manifest of the val shards is written
so it can be verified later that validation data was never modified.

Usage:
    prepare-hf --dataset HuggingFaceFW/fineweb-edu --subsample sample-10BT \
        --max-tokens 4000000000 --num-shards 200
"""
import os
import argparse
import hashlib

import requests
import pyarrow as pa
import pyarrow.parquet as pq

from jevelike.common import download_file, print0

DATASETS_SERVER = "https://datasets-server.huggingface.co"
HUB_API = "https://huggingface.co/api"


def _auth_headers():
    token = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def parse_parquet_api_response(payload, config=None, split="train"):
    """datasets-server /parquet response -> list of file URLs for config/split."""
    urls = []
    for entry in payload.get("parquet_files", []):
        if entry.get("split") != split:
            continue
        if config is not None and entry.get("config") != config:
            continue
        url = entry.get("url")
        if url:
            urls.append(url)
    return sorted(urls)


def parse_tree_api_response(entries, needle=None):
    """Hub tree API entries -> parquet URLs under an optional path substring."""
    urls = []
    for e in entries:
        path = e.get("path", "")
        if e.get("type") != "file" or not path.endswith(".parquet"):
            continue
        if needle is not None and needle not in path:
            continue
        urls.append(f"https://huggingface.co/datasets/{e['dataset']}/resolve/main/{path}"
                    if "dataset" in e else path)
    return sorted(urls)


def list_parquet_urls(dataset, config=None, split="train"):
    """Parquet file URLs of an HF dataset: datasets-server first, Hub tree API fallback."""
    try:
        params = {"dataset": dataset, "split": split}
        if config:
            params["config"] = config
        r = requests.get(f"{DATASETS_SERVER}/parquet", params=params,
                         headers=_auth_headers(), timeout=60)
        r.raise_for_status()
        urls = parse_parquet_api_response(r.json(), config=config, split=split)
        if urls:
            return urls
        print0("datasets-server returned no files; falling back to the Hub tree API")
    except Exception as e:
        print0(f"datasets-server lookup failed ({e}); falling back to the Hub tree API")
    r = requests.get(f"{HUB_API}/datasets/{dataset}/tree/main",
                     params={"recursive": "true"}, headers=_auth_headers(), timeout=60)
    r.raise_for_status()
    entries = r.json()
    for e in entries:
        e.setdefault("dataset", dataset)
    urls = parse_tree_api_response(entries, needle=config)
    if not urls:
        raise SystemExit(f"No parquet files found for {dataset!r} config={config!r}")
    return urls


def cache_path_for(url, cache_dir):
    """Stable local filename for a URL: <md5[:8]>_<basename>."""
    base = os.path.basename(url.split("?")[0]) or "part.parquet"
    h = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
    return os.path.join(cache_dir, f"{h}_{base}")


def download_parquets(urls, cache_dir, max_chars, chars_per_token=4.0, text_field="text"):
    """Download URLs into cache_dir until the accumulated text reaches max_chars.
    Returns (cached_paths, total_chars). Resumable: existing files are not re-downloaded."""
    os.makedirs(cache_dir, exist_ok=True)
    target = int(max_chars)
    cached, total_chars = [], 0
    for i, url in enumerate(urls):
        path = cache_path_for(url, cache_dir)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print0(f"[{i + 1}/{len(urls)}] downloading {url}")
            download_file(url, path, headers=_auth_headers())
        chars = count_text_chars(path, text_field)
        cached.append(path)
        total_chars += chars
        est_tokens = int(total_chars / chars_per_token)
        print0(f"  +{chars:,} chars (total {total_chars:,} ~{est_tokens:,} tokens)")
        if total_chars >= target:
            print0("Reached the token budget; stopping the download")
            break
    return cached, total_chars


def count_text_chars(path, text_field="text"):
    pf = pq.ParquetFile(path)
    check_schema(pf, path, text_field)
    n = 0
    for rg_idx in range(pf.num_row_groups):
        col = pf.read_row_group(rg_idx, columns=[text_field]).column(text_field)
        n += sum(len(s) for s in col.to_pylist())
    return n


def check_schema(pf, path, text_field):
    names = pf.schema_arrow.names
    if text_field not in names:
        raise SystemExit(f"{path}: column {text_field!r} not found; available: {names}. "
                         f"Pass --text-field <name>.")


def shard_of(doc, total_shards, seed):
    """Deterministic global assignment of a doc to a shard (seeded content hash)."""
    h = hashlib.md5(f"{seed}:{doc}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % total_shards


def write_shards(cached_paths, out_dir, num_shards, seed=42, min_chars=200,
                 flush_docs=2048, text_field="text", n_val=None):
    """Rewrite cached parquet files into hf_train_/hf_val_ shards.
    Docs are assigned to shards by a seeded content hash (deterministic global
    shuffle: the val tail carries no stream-order bias). Returns (train, val) files."""
    if n_val is None:
        n_val = max(1, num_shards // 100)
    total_shards = num_shards + n_val
    os.makedirs(out_dir, exist_ok=True)

    def doc_iter():
        for path in cached_paths:
            pf = pq.ParquetFile(path)
            check_schema(pf, path, text_field)
            for rg_idx in range(pf.num_row_groups):
                for doc in pf.read_row_group(rg_idx, columns=[text_field]).column(text_field).to_pylist():
                    if len(doc) >= min_chars:
                        yield doc

    schema = pa.schema([("text", pa.string())])
    buffers = [[] for _ in range(total_shards)]
    writers, paths = {}, []

    def writer_for(i):
        if i not in writers:
            name = f"hf_val_{i:04d}.parquet" if i >= num_shards else f"hf_train_{i:04d}.parquet"
            path = os.path.join(out_dir, name)
            paths.append(path)
            writers[i] = pq.ParquetWriter(path, schema)
        return writers[i]

    n_written_total = 0
    for doc in doc_iter():
        k = shard_of(doc, total_shards, seed)
        buffers[k].append(doc)
        n_written_total += 1
        if len(buffers[k]) >= flush_docs:
            writer_for(k).write_table(pa.table({"text": buffers[k]}))
            buffers[k] = []
    for k in range(total_shards):
        if buffers[k]:
            writer_for(k).write_table(pa.table({"text": buffers[k]}))
            buffers[k] = []
    for w in writers.values():
        w.close()

    train_files = [p for p in paths if "/hf_train_" in p]
    val_files = [p for p in paths if "/hf_val_" in p]
    manifest = os.path.join(out_dir, "VAL_SHA256.txt")
    with open(manifest, "w", encoding="utf-8") as f:
        for p in sorted(val_files):
            h = hashlib.sha256()
            with open(p, "rb") as vf:
                for chunk in iter(lambda: vf.read(1 << 20), b""):
                    h.update(chunk)
            f.write(f"{h.hexdigest()}  {os.path.basename(p)}\n")
    print0(f"Wrote {len(train_files)} train + {len(val_files)} val shards "
           f"({n_written_total:,} docs) to {out_dir}; manifest: {manifest}")
    return train_files, val_files


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="HF dataset id, e.g. HuggingFaceFW/fineweb-edu")
    ap.add_argument("--subsample", default=None, help="dataset config name, e.g. sample-10BT")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--data-dir", default=None, help="output shard dir (default: <base>/data)")
    ap.add_argument("--cache-dir", default=None, help="download cache (default: <data-dir>/hf_cache)")
    ap.add_argument("--num-shards", type=int, default=200, help="number of TRAIN shards")
    ap.add_argument("--val-shards", type=int, default=None, help="val shards (default: num_shards//100, min 1)")
    ap.add_argument("--max-tokens", type=int, required=True, help="token budget to download")
    ap.add_argument("--chars-per-token", type=float, default=4.0)
    ap.add_argument("--min-chars", type=int, default=200, help="drop docs shorter than this")
    ap.add_argument("--flush-docs", type=int, default=2048,
                    help="rows per written row-group (per-shard buffer size)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-download", action="store_true",
                    help="use the existing cache dir as-is (re-shard only)")
    args = ap.parse_args()

    data_dir = args.data_dir or os.path.join(os.getcwd(), "data")
    cache_dir = args.cache_dir or os.path.join(data_dir, "hf_cache")

    if args.skip_download:
        cached = sorted(
            os.path.join(cache_dir, f) for f in os.listdir(cache_dir)
            if f.endswith(".parquet")
        )
        total_chars = sum(count_text_chars(p, args.text_field) for p in cached)
        print(f"Using {len(cached)} cached files ({total_chars:,} chars)")
    else:
        urls = list_parquet_urls(args.dataset, config=args.subsample, split=args.split)
        print0(f"{len(urls)} parquet files listed for {args.dataset}"
               + (f" [{args.subsample}]" if args.subsample else ""))
        cached, total_chars = download_parquets(
            urls, cache_dir, max_chars=args.max_tokens * args.chars_per_token,
            chars_per_token=args.chars_per_token, text_field=args.text_field)

    if not cached:
        raise SystemExit("No data downloaded/cached; nothing to shard.")
    write_shards(cached, data_dir, num_shards=args.num_shards, seed=args.seed,
                 min_chars=args.min_chars, flush_docs=args.flush_docs,
                 text_field=args.text_field, n_val=args.val_shards)


if __name__ == "__main__":
    main()
