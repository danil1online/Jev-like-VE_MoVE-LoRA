import pytest

from jevelike.tokenizer import RustBPETokenizer, build_token_bytes, SPECIAL_TOKENS
from conftest import DIGITS


def test_train_and_roundtrip(small_tok):
    text = "да это тест 42 нет мир привет"
    ids = small_tok.encode(text)
    assert len(ids) > 0
    assert small_tok.decode(ids) == text
    # empty text
    assert small_tok.encode("") == []


def test_special_tokens(small_tok):
    for s in SPECIAL_TOKENS:
        tid = small_tok.encode_single_token(s)
        assert isinstance(tid, int) and tid >= 0
    assert small_tok.get_bos_token_id() == small_tok.encode_single_token("<|bos|>")
    assert set(SPECIAL_TOKENS) <= set(small_tok.get_special_tokens())
    # specials are the top of the vocab
    n = small_tok.get_vocab_size()
    assert max(small_tok.encode_single_token(s) for s in SPECIAL_TOKENS) == n - 1


def test_single_byte_words_are_single_tokens(small_tok):
    # digits are single bytes -> always a single token in any BPE
    for d in DIGITS:
        assert small_tok.encode_single_token(d) is not None


def test_encode_prepend_append(small_tok):
    ids = small_tok.encode("мир", prepend="<|bos|>")
    assert ids[0] == small_tok.get_bos_token_id()
    ids = small_tok.encode(["мир", "да"], prepend="<|bos|>", append="<|a|>")
    assert ids[0][0] == small_tok.get_bos_token_id()
    assert all(row[-1] == small_tok.encode_single_token("<|a|>") for row in ids)
    with pytest.raises(ValueError):
        small_tok.encode("мир", prepend="не-спец-токен-12345")


def test_build_token_bytes(small_tok, tmp_path):
    token_bytes = build_token_bytes(small_tok, str(tmp_path))
    assert token_bytes.numel() == small_tok.get_vocab_size()
    assert (token_bytes >= 0).all()
    # specials are masked with 0
    for s in SPECIAL_TOKENS:
        assert token_bytes[small_tok.encode_single_token(s)].item() == 0
    # byte tokens decode to exactly their byte length
    tid = small_tok.encode_single_token("0")
    assert token_bytes[tid].item() == 1
    assert small_tok.decode_single_token_bytes(tid) == b"0"
