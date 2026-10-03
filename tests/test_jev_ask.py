import pytest
import torch

from conftest import build_tiny
from jevelike.jev import JevAdapter
from jevelike.train.jev_ask import ask


def test_ask_single_forward_pass(fake_tokenizer, model_config, lora_config, jev_config):
    model = build_tiny(model_config, mode="off")
    adapter = JevAdapter(model, lora_config, jev_config, fake_tokenizer)
    adapter.eval()

    calls = {"n": 0}
    orig_hidden = adapter.hidden

    def counting_hidden(idx):
        calls["n"] += 1
        return orig_hidden(idx)

    adapter.hidden = counting_hidden

    questions = ["Вопрос один?", "Вопрос два?"]
    results = ask(adapter, "Контекст текста", questions, "noul", torch.device("cpu"))
    assert calls["n"] == 1  # all questions share one forward pass
    assert len(results) == 2
    noul_words = adapter.renderer.words_by_task["noul"]
    for word, p, probs in results:
        assert word in noul_words
        assert 0.0 < p <= 1.0
        torch.testing.assert_close(probs.sum(), torch.ones(()), rtol=0, atol=1e-6)
        assert probs[adapter.renderer.word_to_cand[word]].item() == p
    # temperature > 1 flattens the reported distribution
    _, _, probs1 = ask(adapter, "ctx", ["q"], "noul", torch.device("cpu"), temperature=1.0)[0]
    _, _, probsT = ask(adapter, "ctx", ["q"], "noul", torch.device("cpu"), temperature=5.0)[0]
    assert probsT.max().item() <= probs1.max().item() + 1e-6

    with pytest.raises(ValueError, match="Unknown task"):
        ask(adapter, "t", ["q"], "bogus", torch.device("cpu"))
