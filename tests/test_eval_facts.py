import json

import torch

from conftest import build_tiny
from jevelike.train.eval_facts import extract_clozes, load_items, score_model


def test_extract_clozes_basic():
    texts = [
        "Paris is the capital of France. Rome is the capital of Italy.",
        "Paris is the capital of France.",  # duplicate -> deduped
        "nothing here matches the pattern",
    ]
    items = extract_clozes(texts)
    assert {"prefix": "The capital of France is", "answer": "Paris"} in items
    assert {"prefix": "The capital of Italy is", "answer": "Rome"} in items
    assert sum(1 for i in items if i["answer"] == "Paris") == 1


def test_extract_clozes_subject_predicate():
    items = extract_clozes(["Newton was English."])
    assert {"prefix": "Newton was", "answer": "English"} in items


def test_extract_clozes_tokenizer_filter():
    class OnlySingle:
        def encode_single_token(self, w):
            if w == "Paris":
                return 1
            raise ValueError("multi-token")

    texts = ["Paris is the capital of France. New York is the largest city of America."]
    unfiltered = extract_clozes(texts)
    assert {"prefix": "The largest city of America is", "answer": "New York"} in unfiltered
    filtered = extract_clozes(texts, tokenizer=OnlySingle())
    assert all(i["answer"] == "Paris" for i in filtered) and filtered


def test_extract_clozes_limit():
    texts = [f"City{i} is the capital of Region{i}." for i in range(100)]
    items = extract_clozes(texts, limit=10)
    assert len(items) == 10


def test_load_items(tmp_path):
    p = tmp_path / "facts.jsonl"
    p.write_text("\n".join([
        json.dumps({"text": "Paris is the capital of France."}),
        json.dumps({"prefix": "Paris is", "answer": "the capital of France"}),
        "",
    ]), encoding="utf-8")
    docs, clozes = load_items(str(p))
    assert len(docs) == 1 and len(clozes) == 1
    assert clozes[0]["answer"] == "the capital of France"


def test_score_model_smoke(fake_tokenizer, model_config):
    model = build_tiny(model_config, mode="off")
    token_bytes = torch.ones(model_config.vocab_size)
    docs = ["да нет А Б В Г да нет", "слово другое слово третье"]
    clozes = [
        {"prefix": "вопрос один", "answer": "да"},      # single-token answer -> evaluated
        {"prefix": "вопрос два", "answer": "two words"},  # multi-token -> skipped
    ]
    m = score_model(model, fake_tokenizer, docs, clozes, torch.device("cpu"),
                    token_bytes=token_bytes)
    assert m["fact_bpb"] > 0 and m["fact_bpb"] < 100
    assert m["n_docs"] == 2
    assert m["n_clozes"] == 1 and m["n_clozes_skipped"] == 1
    assert m["cloze_acc"] in (0.0, 1.0)


def test_score_model_empty_clozes(fake_tokenizer, model_config):
    model = build_tiny(model_config, mode="off")
    token_bytes = torch.ones(model_config.vocab_size)
    m = score_model(model, fake_tokenizer, ["да нет"], [], torch.device("cpu"),
                    token_bytes=token_bytes)
    assert m["cloze_acc"] is None and m["n_clozes"] == 0
