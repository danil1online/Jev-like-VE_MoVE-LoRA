import pytest

from jevelike.configs import DEFAULT_LETTERS
from jevelike.train.tokenizer import candidate_words, find_multi_token_candidates


def test_candidate_words_ru():
    words = candidate_words("ru")
    assert "да" in words and "нет" in words
    assert set(DEFAULT_LETTERS) <= set(words)
    assert set("0123456789") <= set(words)
    assert "yes" not in words and "A" not in words


def test_candidate_words_en():
    words = candidate_words("en")
    assert "yes" in words and "no" in words
    assert set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") <= set(words)
    assert set("0123456789") <= set(words)
    assert "да" not in words


def test_candidate_words_both_and_unique():
    both = candidate_words("both")
    ru, en = candidate_words("ru"), candidate_words("en")
    assert set(both) == set(ru) | set(en)
    assert len(both) == len(set(both))


def test_candidate_words_bad_lang():
    with pytest.raises(AssertionError):
        candidate_words("de")


def test_find_multi_token_candidates_ok(fake_tokenizer):
    # the fake tokenizer has every RU candidate as a single token by construction
    assert find_multi_token_candidates(fake_tokenizer, candidate_words("ru")) == []


def test_find_multi_token_candidates_flags_missing():
    class OnlyRu:
        def encode_single_token(self, w):
            if w in ("да", "нет"):
                return 0
            raise ValueError("not a single token")

    bad = find_multi_token_candidates(OnlyRu(), ["да", "yes", "no", "Q"])
    assert bad == ["yes", "no", "Q"]
