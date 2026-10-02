import pytest
import torch

from jevelike.jev import (
    JevRenderer, JevAdapter, jev_ce_loss, answer_probs, decide,
    calibration_metrics, IGNORE_INDEX,
)
from conftest import build_tiny


def test_renderer_layout(fake_tokenizer, jev_config):
    r = JevRenderer(fake_tokenizer, jev_config)
    ids, target = r.render_one("слово два три", "вопрос?", "да")
    assert ids[0] == r.bos_id
    assert ids[1] == r.ctx_id
    assert ids[-1] == r.a_id
    assert ids.count(r.q_id) == 1
    q_pos = ids.index(r.q_id)
    assert 2 < q_pos < len(ids) - 1
    assert target == fake_tokenizer.encode_single_token("да")
    # answer position is the last one
    assert r.render_one("a", "b", "да")[0][-1] == r.a_id
    # candidate ordering: noul(2) + choice(28) + score(10)
    assert len(r.cand_ids) == 40
    assert r.word_to_cand["да"] == 0
    assert r.word_to_cand["нет"] == 1
    assert r.word_to_cand["А"] == 2
    assert r.word_to_cand["Я"] == 29
    assert r.word_to_cand["0"] == 30
    assert r.word_to_cand["9"] == 39
    words, idxs = r.candidates_for("score")
    assert words == [str(d) for d in range(10)]
    assert idxs == list(range(30, 40))
    with pytest.raises(AssertionError):
        r.render_one("t", "q", "некандидат")
    with pytest.raises(AssertionError):
        r.candidates_for("bogus")


def test_make_batch(fake_tokenizer, jev_config):
    r = JevRenderer(fake_tokenizer, jev_config)
    items = [
        {"text": "a b c", "question": "q", "answer": "да", "task": "noul"},
        {"text": "x", "question": "short", "answer": "А", "task": "choice"},
    ]
    x, labels, meta = r.make_batch(items)
    assert x.shape[0] == 2
    L = x.shape[1]
    # exactly two valid label positions
    assert (labels == IGNORE_INDEX).sum().item() == x.numel() - 2
    assert labels[0, meta[0]["pos"]].item() == fake_tokenizer.encode_single_token("да")
    assert labels[1, meta[1]["pos"]].item() == fake_tokenizer.encode_single_token("А")
    # label sits on the <|a|> token
    assert x[0, meta[0]["pos"]].item() == r.a_id
    assert x[1, meta[1]["pos"]].item() == r.a_id
    # the longer row is the first one; it is not right-padded
    assert meta[0]["pos"] == L - 1
    assert meta[1]["pos"] < L - 1
    assert meta[0]["task"] == "noul" and meta[1]["task"] == "choice"
    with pytest.raises(AssertionError):
        r.make_batch(items, max_len=2)


def test_jev_ce_loss_matches_manual():
    cand = [4, 5, 6, 7]
    B, T = 2, 3
    torch.manual_seed(0)
    logits_full = torch.randn(B, T, 100)
    labels = torch.full((B, T), IGNORE_INDEX)
    labels[0, 1] = cand[3]
    labels[1, 2] = cand[0]
    loss = jev_ce_loss(logits_full, labels, cand)
    sel = logits_full.index_select(-1, torch.tensor(cand))
    logp = torch.log_softmax(sel.float(), dim=-1)
    expected = torch.stack([-logp[0, 1, 3], -logp[1, 2, 0]]).mean()
    torch.testing.assert_close(loss, expected)
    # pre-restricted logits give the same loss
    torch.testing.assert_close(jev_ce_loss(sel, labels, cand), expected)
    # fully masked -> zero
    assert jev_ce_loss(logits_full, torch.full((B, T), IGNORE_INDEX), cand).item() == 0.0
    # temperature
    logp_t = torch.log_softmax(sel.float() / 0.5, dim=-1)
    expected_t = torch.stack([-logp_t[0, 1, 3], -logp_t[1, 2, 0]]).mean()
    torch.testing.assert_close(jev_ce_loss(logits_full, labels, cand, temperature=0.5), expected_t)


def test_answer_probs_and_decide():
    cand = [4, 5, 6]
    logits = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 5.0, 0.0, 0.0, 0.0]]])
    p = answer_probs(logits, cand)
    assert p.shape == (1, 1, 3)
    torch.testing.assert_close(p.sum(-1), torch.ones(1, 1))
    # matches manual softmax over selected columns
    expected = torch.softmax(torch.tensor([1.0, 2.0, 5.0]), dim=-1)
    torch.testing.assert_close(p[0, 0], expected)
    # temperature flattens
    p_t = answer_probs(logits, cand, temperature=100.0)
    torch.testing.assert_close(p_t, torch.full((1, 1, 3), 1 / 3), rtol=0.0, atol=0.02)
    d = decide(p, ["a", "b", "c"])
    assert d.item() == 2


def test_calibration_metrics():
    K = 4
    targets = torch.tensor([0, 1, 2, 3, 0, 1])
    probs = torch.zeros(6, K)
    for n, t in enumerate(targets.tolist()):
        probs[n, t] = 1.0
    m = calibration_metrics(probs, targets)
    assert m["n"] == 6
    assert m["accuracy"] == 1.0
    assert m["ece"] < 1e-9
    assert m["brier"] < 1e-9
    assert m["logloss"] < 1e-6
    # uniform, wrong confidence
    m2 = calibration_metrics(torch.full((4, K), 1.0 / K), targets[:4])
    assert 0.0 < m2["accuracy"] <= 1.0
    assert 0.0 <= m2["ece"] <= 1.0
    assert m2["brier"] > 0
    assert m2["logloss"] > 0


def _make_adapter(model_config, lora_config, jev_config, fake_tokenizer):
    model = build_tiny(model_config, mode="off")
    adapter = JevAdapter(model, lora_config, jev_config, fake_tokenizer)
    return model, adapter


def test_adapter_identity_at_init(fake_tokenizer, model_config, lora_config, jev_config):
    model, adapter = _make_adapter(model_config, lora_config, jev_config, fake_tokenizer)
    torch.manual_seed(1)
    idx = torch.randint(0, model_config.vocab_size, (2, 8))
    for sid in ("<|ctx|>", "<|q|>", "<|a|>"):
        idx[0, [1, 4, 7][("<|ctx|>", "<|q|>", "<|a|>").index(sid)]] = \
            fake_tokenizer.encode_single_token(sid)
    with torch.no_grad():
        base_logits = model(idx)
        h = model(idx, return_hidden=True)
        raw = adapter.extra_out(h)
    cand = adapter.cand_ids
    base_restricted = base_logits.index_select(-1, cand)
    # at init: extra_out == lm_head rows, LoRA delta == 0 -> same logits (up to softcap)
    torch.testing.assert_close(raw, base_restricted, rtol=1e-3, atol=1e-3)


def test_adapter_freeze_and_param_count(fake_tokenizer, model_config, lora_config, jev_config):
    model, adapter = _make_adapter(model_config, lora_config, jev_config, fake_tokenizer)
    trainable = adapter.trainable_params()
    # 2 layers * 2 targets * (A + B) + extra_in + extra_out
    n_lora = len(lora_config.target_modules) * model_config.n_layer * 2
    assert len(trainable) == n_lora + 2
    expected_numel = (n_lora // 2) * 2 * lora_config.rank * model_config.n_embd \
        + 3 * model_config.n_embd + 40 * model_config.n_embd
    assert sum(p.numel() for p in trainable) == expected_numel
    grad_ids = {id(p) for p in adapter.parameters() if p.requires_grad}
    assert grad_ids == {id(p) for p in trainable}
    assert not model.transformer.wte.weight.requires_grad
    assert not model.lm_head.weight.requires_grad


def test_adapter_forward_backward(fake_tokenizer, model_config, lora_config, jev_config):
    model, adapter = _make_adapter(model_config, lora_config, jev_config, fake_tokenizer)
    items = [
        {"text": "первое слово второе", "question": "вопрос один", "answer": "да", "task": "noul"},
        {"text": "a", "question": "b c d e f", "answer": "5", "task": "score"},
        {"text": "x y", "question": "z", "answer": "К", "task": "choice"},
    ]
    x, labels, meta = adapter.renderer.make_batch(items)
    loss = adapter(x, labels)
    assert loss.ndim == 0
    assert loss.item() > 0
    loss.backward()
    for p in adapter.trainable_params():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    # probs at every position, softmax over candidates
    probs = adapter.probs(x)
    assert probs.shape == (x.shape[0], x.shape[1], 40)
    torch.testing.assert_close(probs.sum(-1), torch.ones(probs.shape[:-1]), rtol=0.0, atol=1e-5)
    # per-position calibration metrics are computable from the adapter probs
    tidx = torch.tensor([adapter.renderer.word_to_cand[it["answer"]] for it in items])
    m = calibration_metrics(probs[torch.arange(3), [m2["pos"] for m2 in meta]], tidx)
    assert 0.0 <= m["accuracy"] <= 1.0
    assert m["n"] == 3
