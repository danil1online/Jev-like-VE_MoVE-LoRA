"""
Train a BPE tokenizer in the style of GPT-4 on a parquet dataset and verify that
all Jev answer candidates are single tokens (a hard requirement of JevAdapter).

Usage:
    tok-train [--data-dir data] [--max-chars 2000000000] [--vocab-size 65536]
              [--lang ru|en|both] [--extra-data-dir dir ...]

--extra-data-dir corpora are consumed before the main --data-dir (e.g. a small
ruwiki fraction added to an English corpus so Cyrillic gets real merges).
"""
import os
import time
import argparse
import torch

from jevelike.configs import DEFAULT_LETTERS
from jevelike.tokenizer import RustBPETokenizer, build_token_bytes
from jevelike.common import get_base_dir, get_data_dir
from jevelike.dataset import parquets_iter_batched

SPECIAL_TOKENS = ["<|bos|>", "<|ctx|>", "<|q|>", "<|a|>"]
DIGITS = "0123456789"
EN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def candidate_words(lang: str):
    """Jev answer-candidate words that must be single tokens for the given language."""
    assert lang in ("ru", "en", "both"), f"unknown lang {lang!r}"
    words = []
    if lang in ("ru", "both"):
        words += ["да", "нет"] + list(DEFAULT_LETTERS) + list(DIGITS)
    if lang in ("en", "both"):
        words += ["yes", "no"] + list(EN_LETTERS) + list(DIGITS)
    return sorted(set(words))


def find_multi_token_candidates(tokenizer, words):
    """Words the tokenizer cannot encode as a single token (Jev would fail on them)."""
    bad = []
    for w in words:
        try:
            tokenizer.encode_single_token(w)
        except Exception:
            bad.append(w)
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=str, default=None, help="parquet data dir (default: <base>/data)")
    ap.add_argument("--extra-data-dir", type=str, action="append", default=[],
                    help="additional corpus dir consumed first (repeatable), e.g. a ru fraction")
    ap.add_argument("--max-chars", type=int, default=2_000_000_000,
                    help="maximum characters to train on (default: 2B)")
    ap.add_argument("--doc-cap", type=int, default=10_000,
                    help="maximum characters per document (default: 10,000)")
    ap.add_argument("--vocab-size", type=int, default=65536,
                    help="vocabulary size including special tokens (default: 65536)")
    ap.add_argument("--lang", type=str, default="ru", choices=("ru", "en", "both"),
                    help="which Jev candidate words to sanity-check (default: ru)")
    ap.add_argument("--allow-multi-token", action="store_true",
                    help="warn instead of failing when a candidate word is not a single token")
    args = ap.parse_args()
    print(f"max_chars: {args.max_chars:,}")
    print(f"doc_cap: {args.doc_cap:,}")
    print(f"vocab_size: {args.vocab_size:,}")
    data_dir = args.data_dir or get_data_dir()

    def iter_dir_texts(directory):
        for batch in parquets_iter_batched(directory, split="train"):
            for doc in batch:
                yield doc

    def text_iterator():
        nchars = 0
        for directory in list(args.extra_data_dir) + [data_dir]:
            for doc_text in iter_dir_texts(directory):
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
    for word in SPECIAL_TOKENS:
        tok_id = tokenizer.encode_single_token(word)
        print(f"  {word!r} -> id {tok_id}")

    candidates = candidate_words(args.lang)
    bad = find_multi_token_candidates(tokenizer, candidates)
    if bad:
        msg = (f"{len(bad)}/{len(candidates)} Jev candidate words are NOT single tokens "
               f"for --lang {args.lang}: {bad[:20]}{'...' if len(bad) > 20 else ''}. "
               f"JevAdapter would fail on them.")
        if not args.allow_multi_token:
            raise SystemExit(msg + " Retrain with more data (or --allow-multi-token to ignore).")
        print(f"WARNING: {msg}")
    else:
        print(f"Tokenizer sanity check passed: all {len(candidates)} candidate words "
              f"(lang={args.lang}) are single tokens.")


if __name__ == "__main__":
    main()
