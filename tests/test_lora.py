import pytest
import torch
import torch.nn

from jevelike.configs import LoraConfig
from jevelike.lora import apply_lora, freeze_base, LoRALinear, lora_num_params, trainable_params
from conftest import build_tiny, tiny_model_config


def test_lora_linear_math():
    torch.manual_seed(0)
    base = torch.nn.Linear(8, 5, bias=False)
    lora = LoRALinear(base, rank=3, alpha=6.0)
    assert lora.scale == 2.0
    x = torch.randn(2, 7, 8)
    y = lora(x)
    delta = (x.float() @ lora.lora_A.t() @ lora.lora_B.t()) * lora.scale
    torch.testing.assert_close(y, base(x) + delta.to(x.dtype))


def test_lora_init_is_exact_noop():
    torch.manual_seed(0)
    base = torch.nn.Linear(8, 5, bias=False)
    lora = LoRALinear(base, rank=3)
    x = torch.randn(2, 7, 8)
    torch.testing.assert_close(lora(x), base(x))


def test_lora_dropout_identity_at_eval():
    lora = LoRALinear(torch.nn.Linear(8, 5, bias=False), rank=3, dropout=0.5)
    lora.eval()
    x = torch.randn(2, 7, 8)
    delta = (x.float() @ lora.lora_A.t() @ lora.lora_B.t()) * lora.scale
    torch.testing.assert_close(lora(x), lora.base(x) + delta.to(x.dtype))


def test_apply_lora_replaces_targets():
    cfg = tiny_model_config(n_layer=2)
    model = build_tiny(cfg, mode="off")
    lora_cfg = LoraConfig(rank=2, alpha=4.0, target_modules=["q", "v", "proj"])
    adapters = apply_lora(model, lora_cfg)
    assert len(adapters) == 2 * 3
    assert all(isinstance(a, LoRALinear) for a in adapters)
    assert isinstance(model.transformer.h[0].attn.c_q, LoRALinear)
    assert isinstance(model.transformer.h[1].attn.c_v, LoRALinear)
    assert isinstance(model.transformer.h[1].attn.c_proj, LoRALinear)
    # non-target linears untouched
    assert not isinstance(model.transformer.h[0].attn.c_k, LoRALinear)
    assert not isinstance(model.transformer.h[0].mlp.c_fc, LoRALinear)
    # forward still works end to end
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    logits = model(idx)
    assert logits.shape == (2, 8, cfg.vocab_size)


def test_apply_lora_invalid_target():
    cfg = tiny_model_config(n_layer=2)
    with pytest.raises(AssertionError):
        apply_lora(build_tiny(cfg, mode="off"), LoraConfig(target_modules=["bogus"]))


def test_freeze_base_and_param_counts():
    cfg = tiny_model_config(n_layer=2)
    model = build_tiny(cfg, mode="off")
    lora_cfg = LoraConfig(rank=2, alpha=4.0, target_modules=["q", "v"])
    adapters = apply_lora(model, lora_cfg)
    freeze_base(model)
    tp = trainable_params(model)
    assert len(tp) == len(adapters) * 2
    # per adapter: A (rank, n_embd) + B (n_embd, rank) = 2 * rank * n_embd
    assert lora_num_params(model) == len(adapters) * 2 * lora_cfg.rank * cfg.n_embd
    assert sum(p.numel() for p in tp) == lora_num_params(model)
    # everything else frozen
    frozen = [p for p in model.parameters() if not p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    assert total - sum(p.numel() for p in tp) == sum(p.numel() for p in frozen)
