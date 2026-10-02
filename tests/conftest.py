"""
Shared fixtures: a deterministic fake tokenizer (so Jev tests never depend on
rustbpe merge luck) and tiny model configs.
"""
import hashlib
import pickle
import sys
import types

import pytest
import torch

from jevelike.configs import ModelConfig, MoveConfig, LoraConfig, JevConfig
from jevelike.move import resolve_move
from jevelike.gpt import GPT

LETTERS = "АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЭЮЯ"
DIGITS = "0123456789"
SPECIALS = ["<|bos|>", "<|ctx|>", "<|q|>", "<|a|>"]
CANDIDATES = ["да", "нет"] + list(LETTERS) + list(DIGITS)

FAKE_VOCAB = 4096


class FakeTokenizer:
    """Deterministic stand-in for RustBPETokenizer with the exact API surface
    used by jevelike (encode, decode, encode_single_token, special ids).
    Every Jev candidate word is a single token by construction."""

    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        for i, w in enumerate(SPECIALS):
            self.word_to_id[w] = i
            self.id_to_word[i] = w
        for i, w in enumerate(CANDIDATES):
            tid = len(SPECIALS) + i
            self.word_to_id[w] = tid
            self.id_to_word[tid] = w

    def get_vocab_size(self):
        return FAKE_VOCAB

    def get_special_tokens(self):
        return list(SPECIALS)

    def get_bos_token_id(self):
        return self.word_to_id["<|bos|>"]

    def encode_special(self, text):
        return self.word_to_id[text]

    def encode_single_token(self, text):
        if text in self.word_to_id:
            return self.word_to_id[text]
        raise ValueError(f"Word {text!r} is not a single token in the fake tokenizer")

    def _pseudo_id(self, word):
        h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
        return 200 + h % 1500  # stays below FAKE_VOCAB

    def encode(self, text, prepend=None, append=None, num_threads=8):
        if isinstance(text, list):
            raise NotImplementedError
        words = text.split()
        ids = [self.word_to_id[w] if w in self.word_to_id else self._pseudo_id(w) for w in words]
        if prepend is not None:
            p = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids = [p] + ids
        if append is not None:
            a = append if isinstance(append, int) else self.encode_special(append)
            ids = ids + [a]
        return ids

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return " ".join(self.id_to_word.get(i, f"<unk{i}>") for i in ids)

    def decode_single_token_bytes(self, token_id):
        return self.id_to_word[token_id].encode("utf-8") if token_id in self.id_to_word else b"x"

    def save(self, tokenizer_dir):
        import os
        os.makedirs(tokenizer_dir, exist_ok=True)
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "wb") as f:
            pickle.dump(self, f)


@pytest.fixture(scope="session")
def fake_tokenizer():
    return FakeTokenizer()


SMALL_CORPUS = (
    "да нет А Б В Г Д Е Ж З И К Л М Н О П Р С Т У Ф Х Ц Ч Ш Щ Э Ю Я "
    "0 1 2 3 4 5 6 7 8 9 "
    "Привет мир это тестовый текст на русском языке для обучения токенизатора "
    "Слова предложений должны хорошо сегментироваться по байтам "
)


@pytest.fixture(scope="session")
def small_tok():
    """A real rustbpe-trained tokenizer on a tiny corpus (for the tokenizer,
    checkpoint and dataloader tests). NOT guaranteed to make every candidate
    word a single token -- that is the job of the full 2e9-char training run."""
    from jevelike.tokenizer import RustBPETokenizer
    return RustBPETokenizer.train_from_iterator(iter([SMALL_CORPUS] * 60), 512)


def tiny_model_config(**overrides):
    cfg = dict(sequence_len=32, vocab_size=FAKE_VOCAB, padding_multiple=16,
               n_layer=2, n_head=2, n_kv_head=2, n_embd=32, window_pattern="SL", softcap=15.0)
    cfg.update(overrides)
    return ModelConfig(**cfg)


@pytest.fixture
def model_config():
    return tiny_model_config()


def build_tiny(model_config, mode="off", **move_kwargs):
    move_cfg = MoveConfig(mode=mode, **move_kwargs)
    move = resolve_move(move_cfg, model_config)
    torch.manual_seed(0)
    model = GPT(model_config, move=move)
    model.init_weights()  # GPT.__init__ does not init (see checkpoint.build_model)
    return model


@pytest.fixture
def lora_config():
    return LoraConfig(rank=2, alpha=4.0, target_modules=["q", "v"])


@pytest.fixture
def jev_config():
    return JevConfig()
