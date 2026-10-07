import json

import pytest

from jevelike.data.contrastive import (
    build_pair_prompt, parse_pair_response, pair_rows, split_rows,
    load_input_texts, make_pair_id,
)


def test_build_pair_prompt():
    system, user = build_pair_prompt("текст один")
    assert "контрастивные" in system
    assert "текст один" in user
    assert "true_statement" in user and "false_statement" in user


def test_build_pair_prompt_en():
    system, user = build_pair_prompt("some text", lang="en")
    assert "contrastive" in system.lower()
    assert "some text" in user
    assert "true_statement" in user and "false_statement" in user
    # the JSON example braces must survive .format (regression guard)
    assert '{\n  "true_statement"' in user


def test_pair_rows_en():
    pair = {"true_statement": "t", "false_statement": "f"}
    rows = pair_rows("text", pair, "pid", lang="en")
    assert rows[0]["answer"] == "yes" and rows[1]["answer"] == "no"
    assert rows[0]["task"] == "noul" and rows[0]["pair_id"] == "pid"


@pytest.mark.parametrize("raw,expected", [
    ('{"true_statement": "а", "false_statement": "б"}', {"true_statement": "а", "false_statement": "б"}),
    ('{"false_statement": "б", "true_statement": "а"}', {"true_statement": "а", "false_statement": "б"}),
    ('```json\n{"true_statement": "а", "false_statement": "б"}\n```', {"true_statement": "а", "false_statement": "б"}),
    ('Вот ответ:\n{"true_statement": "а", "false_statement": "б"}\nУдачи!', {"true_statement": "а", "false_statement": "б"}),
])
def test_parse_pair_response_ok(raw, expected):
    assert parse_pair_response(raw) == expected


@pytest.mark.parametrize("raw", [
    "",
    None,
    '{"true_statement": "а"}',
    '{"true_statement": "а", "false_statement": "а"}',
    '{"true_statement": "", "false_statement": "б"}',
    '[1, 2, 3]',
    'не json вообще',
    '```python\nprint(1)\n```',
])
def test_parse_pair_response_rejects(raw):
    assert parse_pair_response(raw) is None


def test_pair_rows():
    pair = {"true_statement": "т", "false_statement": "ф"}
    rows = pair_rows("текст", pair, "pid123")
    assert len(rows) == 2
    assert rows[0] == {"text": "текст", "question": "т", "answer": "да", "task": "noul", "pair_id": "pid123"}
    assert rows[1] == {"text": "текст", "question": "ф", "answer": "нет", "task": "noul", "pair_id": "pid123"}
    assert rows[0]["pair_id"] == rows[1]["pair_id"]


def test_split_rows_pairs_stay_together():
    rows = []
    for pid in range(20):
        rows.append({"pair_id": pid, "i": 0})
        rows.append({"pair_id": pid, "i": 1})
    train, val = split_rows(rows, val_frac=0.25, seed=7)
    assert len(train) + len(val) == len(rows)
    assert len(val) == 10  # 5 pairs * 2 rows
    train_pids = {r["pair_id"] for r in train}
    val_pids = {r["pair_id"] for r in val}
    assert not train_pids & val_pids
    # both rows of every pair land in the same split
    for r in train + val:
        assert (r["pair_id"] in train_pids) == (r in train)


def test_split_rows_deterministic():
    rows = [{"pair_id": i, "i": j} for i in range(10) for j in range(2)]
    t1, v1 = split_rows(rows, 0.3, seed=42)
    t2, v2 = split_rows(rows, 0.3, seed=42)
    t3, v3 = split_rows(rows, 0.3, seed=43)
    assert t1 == t2 and v1 == v2
    assert t3 != t1  # different seed -> different assignment


def test_split_rows_extremes():
    rows = [{"pair_id": i} for i in range(5)]
    train, val = split_rows(rows, 0.0)
    assert train and not val
    train, val = split_rows(rows, 1.0)
    assert val and not train


def test_load_input_texts(tmp_path):
    jsonl = tmp_path / "in.jsonl"
    jsonl.write_text(
        json.dumps({"text": "один"}) + "\n"
        + "\n"
        + json.dumps({"text": "два"}) + "\n",
        encoding="utf-8",
    )
    assert load_input_texts(str(jsonl)) == ["один", "два"]
    txt = tmp_path / "in.txt"
    txt.write_text("линия одна\nлиния два\n\n", encoding="utf-8")
    assert load_input_texts(str(txt)) == ["линия одна", "линия два"]


def test_make_pair_id():
    assert make_pair_id("один") == make_pair_id("один")
    assert make_pair_id("один") != make_pair_id("два")
    assert len(make_pair_id("один")) == 12
