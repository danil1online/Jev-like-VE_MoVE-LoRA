import pytest
import torch

from jevelike.move import MoveBank, mix_value, resolve_move
from jevelike.configs import ModelConfig, MoveConfig
from conftest import tiny_model_config, build_tiny

torch.manual_seed(0)


def _manual_move_mix(v, ve, z, scale, gated):
    g = scale * torch.sigmoid(z.float())
    if gated:
        g0, gm = g[..., :1], g[..., 1:]
        out = v * g0
    else:
        gm = g
        out = v
    return out + (gm.unsqueeze(-1) * ve.float()).sum(dim=3)


def _manual_lave_mix(v, ve, z, scale, gated):
    g = scale * torch.sigmoid(z.float())
    if gated:
        g0, g1 = g[..., :1], g[..., 1:]
        return v * g0 + ve * g1
    return v + ve * g.unsqueeze(-1)


def test_move_bank_shapes():
    bank = MoveBank(vocab_size=100, num_slots=3, kv_dim=16)
    bank.init_weights()  # torch.empty leaves garbage bits (incl. NaN patterns)
    idx = torch.randint(0, 100, (4, 7))
    ve = bank(idx)
    assert ve.shape == (4, 7, 3, 16)
    # lookup matches direct indexing
    torch.testing.assert_close(ve, bank.weight[idx])
    bank.init_weights(s=0.5)
    assert bank.weight.abs().max() <= 0.5 + 1e-6


def test_mix_value_move_gated_matches_manual():
    B, T, H, M, D = 2, 3, 4, 3, 8
    v = torch.randn(B, T, H, D)
    ve = torch.randn(B, T, H, M, D)
    z = torch.randn(B, T, H, M + 1)
    out = mix_value(v, ve, z, 2.0, gated_standard=True)
    torch.testing.assert_close(out, _manual_move_mix(v, ve, z, 2.0, True))
    assert out.shape == v.shape


def test_mix_value_move_ungated_matches_manual():
    B, T, H, M, D = 2, 3, 4, 3, 8
    v = torch.randn(B, T, H, D)
    ve = torch.randn(B, T, H, M, D)
    z = torch.randn(B, T, H, M)
    out = mix_value(v, ve, z, 2.0, gated_standard=False)
    torch.testing.assert_close(out, _manual_move_mix(v, ve, z, 2.0, False))


def test_mix_value_lave_ungated_matches_manual():
    B, T, H, D = 2, 3, 4, 8
    v = torch.randn(B, T, H, D)
    ve = torch.randn(B, T, H, D)
    z = torch.randn(B, T, H)
    out = mix_value(v, ve, z, 2.0, gated_standard=False)
    torch.testing.assert_close(out, _manual_lave_mix(v, ve, z, 2.0, False))


def test_mix_value_lave_gated_matches_manual():
    B, T, H, D = 2, 3, 4, 8
    v = torch.randn(B, T, H, D)
    ve = torch.randn(B, T, H, D)
    z = torch.randn(B, T, H, 2)
    out = mix_value(v, ve, z, 2.0, gated_standard=True)
    torch.testing.assert_close(out, _manual_lave_mix(v, ve, z, 2.0, True))


def test_mix_value_limits():
    # gate -> -inf: no extra value injected; if gated, standard path -> 0
    B, T, H, M, D = 1, 1, 2, 2, 4
    v = torch.randn(B, T, H, D)
    ve = torch.randn(B, T, H, M, D)
    z = torch.full((B, T, H, M + 1), -30.0)
    out_gated = mix_value(v, ve, z, 2.0, gated_standard=True)
    torch.testing.assert_close(out_gated, torch.zeros_like(v), rtol=0.0, atol=1e-4)
    z = torch.full((B, T, H, M), -30.0)
    out_ung = mix_value(v, ve, z, 2.0, gated_standard=False)
    torch.testing.assert_close(out_ung, v, rtol=0.0, atol=1e-4)
    # gate -> +inf, scale=2: g0 = 2 -> V = 2v; slots: sum of ve * 2
    z = torch.full((B, T, H, M + 1), 30.0)
    out = mix_value(v, ve, z, 2.0, gated_standard=True)
    expected = 2.0 * v + (2.0 * ve).sum(dim=3)
    torch.testing.assert_close(out, expected, rtol=0.0, atol=1e-4)


def test_resolve_move_gate_dims():
    m = tiny_model_config(n_layer=4, n_head=2, n_kv_head=2, n_embd=32)
    r = resolve_move(MoveConfig(mode="move", num_slots=2), m)
    assert r.gate_in_dim == 32          # "full" gate
    assert r.gate_out_dim(2) == 6       # (M+1)*H
    r12 = resolve_move(MoveConfig(mode="move", num_slots=2, gate_input="12"), m)
    assert r12.gate_in_dim == 12
    with pytest.raises(AssertionError):
        resolve_move(MoveConfig(mode="move", num_slots=2, gate_input="12"),
                     tiny_model_config(n_embd=8))
    # lave gate dims
    rl = resolve_move(MoveConfig(mode="lave"), m)
    assert rl.gated_standard is False
    assert rl.gate_out_dim(2) == 2      # 1 slot * H
    rlg = resolve_move(MoveConfig(mode="lave", gated_standard=True), m)
    assert rlg.gate_out_dim(2) == 4     # 2 slots * H


def test_value_health_stats(fake_tokenizer, model_config):
    from jevelike.train.base import value_health_stats
    off = build_tiny(model_config, mode="off")
    assert value_health_stats(off) == {}
    mv = build_tiny(model_config, mode="move", num_slots=2)
    s = value_health_stats(mv)
    assert s["ve/bank_rows"] > 0 and s["ve/bank_norm_mean"] >= 0.0
    assert 0.0 <= s["ve/bank_dead_frac"] <= 1.0
    assert "ve/gate_norm_mean" in s and s["ve/gate_norm_max"] >= s["ve/gate_norm_mean"]
    lv = build_tiny(model_config, mode="lave", lave_layers="deep")
    sl = value_health_stats(lv)
    assert sl["ve/bank_rows"] > 0 and "ve/gate_norm_mean" in sl
