import hashlib
import os

import pyarrow as pa
import pyarrow.parquet as pq

from jevelike.dataset import list_parquet_files
from jevelike.data.prep_hf import (
    parse_parquet_api_response, parse_tree_api_response, cache_path_for,
    shard_of, write_shards,
)


def _make_cache(tmp_path, files):
    """files: list of lists of docs -> returns list of parquet paths."""
    paths = []
    for i, docs in enumerate(files):
        p = str(tmp_path / f"cache_{i}.parquet")
        pq.write_table(pa.table({"text": docs}), p)
        paths.append(p)
    return paths


def _read_all(out_dir):
    docs = []
    for p in list_parquet_files(out_dir):
        docs += pq.read_table(p).column("text").to_pylist()
    return docs


def test_parse_parquet_api_response():
    payload = {"parquet_files": [
        {"config": "sample-10BT", "split": "train", "url": "u2"},
        {"config": "sample-10BT", "split": "train", "url": "u1"},
        {"config": "sample-10BT", "split": "val", "url": "uv"},
        {"config": "sample-100BT", "split": "train", "url": "u3"},
    ]}
    assert parse_parquet_api_response(payload, config="sample-10BT") == ["u1", "u2"]
    assert parse_parquet_api_response(payload) == ["u1", "u2", "u3"]  # any config, train split


def test_parse_tree_api_response():
    entries = [
        {"type": "file", "path": "data/sample-10BT/a.parquet", "dataset": "d"},
        {"type": "file", "path": "data/sample-100BT/b.parquet", "dataset": "d"},
        {"type": "directory", "path": "data/c.parquet", "dataset": "d"},
    ]
    urls = parse_tree_api_response(entries, needle="sample-10BT")
    assert urls == ["https://huggingface.co/datasets/d/resolve/main/data/sample-10BT/a.parquet"]


def test_cache_path_for_stable_and_unique():
    a = cache_path_for("https://x/a/part_0000.parquet", "/c")
    b = cache_path_for("https://y/b/part_0000.parquet", "/c")
    assert a == cache_path_for("https://x/a/part_0000.parquet", "/c")
    assert a != b and a.endswith("_part_0000.parquet")


def test_shard_of_deterministic_and_balanced():
    docs = [f"doc {i} " + "x" * 50 for i in range(2000)]
    a = [shard_of(d, 8, 42) for d in docs]
    b = [shard_of(d, 8, 42) for d in docs]
    assert a == b
    counts = [a.count(k) for k in range(8)]
    assert all(150 < c < 350 for c in counts)  # roughly uniform
    assert set(a) == set(range(8))
    # different seed -> different assignment (same docs, shard-by-shard)
    assert [shard_of(d, 8, 1) for d in docs[:50]] != [shard_of(d, 8, 42) for d in docs[:50]]


def test_write_shards_layout_and_coverage(tmp_path):
    docs = [f"document number {i} " + "lorem ipsum " * 20 for i in range(500)]
    short = ["tiny"] * 30  # below min_chars -> dropped
    cache = _make_cache(tmp_path, [docs[:250], docs[250:] + short])
    out = str(tmp_path / "out")
    train, val = write_shards(cache, out, num_shards=9, seed=7)
    assert len(val) == 1 and len(train) == 9
    names = sorted(os.path.basename(p) for p in list_parquet_files(out))
    assert names[-1].startswith("hf_val_")  # val sorts last -> dataloader split works
    got = _read_all(out)
    assert sorted(got) == sorted(docs)  # every long doc exactly once, shorts dropped


def test_write_shards_manifest_matches(tmp_path):
    docs = [f"d{i} " + "y" * 300 for i in range(120)]
    cache = _make_cache(tmp_path, [docs])
    out = str(tmp_path / "out")
    _, val = write_shards(cache, out, num_shards=5, seed=3)
    manifest = open(os.path.join(out, "VAL_SHA256.txt")).read().splitlines()
    assert len(manifest) == len(val)
    for line in manifest:
        h, name = line.split()
        p = os.path.join(out, name)
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == h


def test_write_shards_deterministic(tmp_path):
    docs = [f"doc-{i}-" + "z" * 100 for i in range(300)]
    cache = _make_cache(tmp_path, [docs])
    o1, o2 = str(tmp_path / "o1"), str(tmp_path / "o2")
    write_shards(cache, o1, num_shards=4, seed=5)
    write_shards(cache, o2, num_shards=4, seed=5)
    assert sorted(_read_all(o1)) == sorted(_read_all(o2))
    n1 = sorted(os.listdir(o1))
    n2 = sorted(os.listdir(o2))
    assert n1 == n2
