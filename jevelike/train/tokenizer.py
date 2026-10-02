"""
Train a BPE tokenizer in the style of GPT-4 on the ruwiki parquet dataset.

Usage:
    tok-train [--data-dir data] [--max-chars 2000000000] [--vocab-size 65536]
"""
import os
import time
import argparse
import torch

from jevelike.tokenizer import RustBPETokenizer, build_token_bytes
from jevelike.common import get_base_dir, get_data_dir
from jevelike.dataset import parquets_iter_batched


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=str, default=None, help="parquet data dir (default: <base>/data)")
    ap.add_argument("--max-chars", type=int, default=2_000_000_000,
                    help="maximum characters to train on (default: 2B)")
    ap.add_argument("--doc-cap", type=int, default=10_000,
                    help="maximum characters per document (default: 10,000)")
    ap.add_argument("--vocab-size", type=int, default=65536,
                    help="vocabulary size including special tokens (default: 65536)")
    args = ap.parse_args()
    print(f"max_chars: {args.max_chars:,}")
    print(f"doc_cap: {args.doc_cap:,}")
    print(f"vocab_size: {args.vocab_size:,}")
    data_dir = args.data_dir or get_data_dir()

    def text_iterator():
        nchars = 0
        for batch in parquets_iter_batched(data_dir, split="train"):
            for doc in batch:
                doc_text = doc
                if len(doc_text) > args.doc_cap:
                    doc_text = doc_text[:args.doc_cap]
                nchars += len(doc_text)
                yield doc_text
                if nchars > args.max_chars:
                    return

    t0 = time.time()
    tokenizer = RustBPETokenizer.train_from_iterator(text_iterator(), args.vocab_size)
    train_time = time.time() - t0
    print(f"Training time: {train_time:.2f}s")

    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    tokenizer.save(tokenizer_dir)
    build_token_bytes(tokenizer, tokenizer_dir)

    # quick inline sanity check
    test_text = "Привет, мир! Это тест.\nЧисла: 123, 4567, 89\nОсобые: да нет А Б В"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    assert decoded == test_text, f"tokenizer roundtrip failed: {decoded!r}"
    for word in ["да", "нет", "А", "Я", "0", "9", "<|bos|>", "<|ctx|>", "<|q|>", "<|a|>"]:
        tok_id = tokenizer.encode_single_token(word)
        print(f"  {word!r} -> id {tok_id}")
    print("Tokenizer sanity check passed.")


if __name__ == "__main__":
    main()
