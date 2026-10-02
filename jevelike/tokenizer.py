"""
BPE tokenizer in the style of GPT-4: train with rustbpe, inference with tiktoken.
Ported from nanochat (karpathy/nanochat).

Difference from nanochat: the special-token set is the Jev prompt set:
    <|bos|>  beginning of sequence (document delimiter)
    <|ctx|>  marks the start of the context (article / document)
    <|q|>    marks the start of the question
    <|a|>    marks the answer position (the model emits ONE answer token after it)

Answer words (да / нет / А..Я / 0..9) are deliberately NOT special tokens:
rustbpe learns ordinary ranks for them, and encode_ordinary never emits special
ids. Answer token ids are resolved at runtime via encode_single_token()
(see jevelike/jev.py:JevRenderer).
"""
import os
import pickle
from functools import lru_cache

import rustbpe
import tiktoken

SPECIAL_TOKENS = [
    "<|bos|>",
    "<|ctx|>",
    "<|q|>",
    "<|a|>",
]

# Same split pattern as nanochat (GPT-4 style, tuned for ~32K+ vocab).
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RustBPETokenizer:
    """Light wrapper around tiktoken (for efficient inference) but train with rustbpe."""

    def __init__(self, enc, bos_token):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # 1) train using rustbpe
        tokenizer = rustbpe.Tokenizer()
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, \
            f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
        # 2) construct the associated tiktoken encoding for inference
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
        return cls(enc, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir):
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        enc = tiktoken.get_encoding(tiktoken_name)
        return cls(enc, "<|bos|>")

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    def id_to_token(self, id):
        return self.enc.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        try:
            return self.enc.encode_single_token(text)
        except Exception as e:
            raise ValueError(
                f"Special token {text!r} is not in the vocabulary. "
                f"Valid special tokens: {SPECIAL_TOKENS}. ({e})"
            ) from e

    def encode_single_token(self, text):
        """Encode a word that MUST be a single token; raises a clear error otherwise."""
        try:
            return self.enc.encode_single_token(text)
        except Exception as e:
            raise ValueError(
                f"Word {text!r} is not a single token in this tokenizer. "
                f"Jev answer words (да/нет, А..Я, 0..9) must be single tokens. "
                f"Train the tokenizer on enough text, or pick another word. ({e})"
            )

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id)
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.enc.decode(ids)

    def decode_single_token_bytes(self, token_id):
        return self.enc.decode_single_token_bytes(token_id)

    def save(self, tokenizer_dir):
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer encoding to {pickle_path}")


# -----------------------------------------------------------------------------
# Convenience functions

def get_tokenizer(tokenizer_dir=None):
    from jevelike.common import get_base_dir
    if tokenizer_dir is None:
        tokenizer_dir = os.path.join(get_base_dir(), "tokenizer")
    return RustBPETokenizer.from_directory(tokenizer_dir)


def get_token_bytes(device="cpu", tokenizer_dir=None):
    import torch
    from jevelike.common import get_base_dir
    if tokenizer_dir is None:
        tokenizer_dir = os.path.join(get_base_dir(), "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), \
        f"Token bytes not found at {token_bytes_path}? It gets written by tok_train."
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes


def build_token_bytes(tokenizer, tokenizer_dir):
    """Cache token id -> byte length for BPB evaluation (see tok_train)."""
    import torch
    vocab_size = tokenizer.get_vocab_size()
    special_ids = set(tokenizer.encode_special(s) for s in tokenizer.get_special_tokens())
    token_bytes = []
    for token_id in range(vocab_size):
        if token_id in special_ids:
            token_bytes.append(0)  # special tokens are not counted
        else:
            token_bytes.append(len(tokenizer.decode_single_token_bytes(token_id)))
    token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device="cpu")
    os.makedirs(tokenizer_dir, exist_ok=True)
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    with open(token_bytes_path, "wb") as f:
        torch.save(token_bytes, f)
    print(f"Saved token_bytes to {token_bytes_path}")
    return token_bytes
