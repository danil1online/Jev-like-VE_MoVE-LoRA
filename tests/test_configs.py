import os
import pytest

from jevelike.configs import (
    ModelConfig, MoveConfig, LoraConfig, JevConfig, TrainConfig,
    load_config, save_config,
)
from jevelike.move import resolve_move

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")


def test_repo_configs_load():
    names = ["base_d12_off.yaml", "base_d12_move.yaml", "base_d12_lave.yaml",
             "base_d20_move.yaml", "jev_lora_d12.yaml"]
    for name in names:
        cfg = load_config(os.path.join(CONFIGS_DIR, name))
        cfg.validate()
    d12 = load_config(os.path.join(CONFIGS_DIR, "base_d12_off.yaml"))
    assert d12.model.n_layer == 12
    assert d12.model.kv_dim == 768


def test_model_config_validate():
    ModelConfig(n_layer=12, n_head=6, n_kv_head=6, n_embd=768).validate()
    with pytest.raises(AssertionError):
        ModelConfig(n_embd=100, n_head=7).validate()
    with pytest.raises(AssertionError):
        ModelConfig(n_head=6, n_kv_head=4).validate()
    with pytest.raises(AssertionError):
        ModelConfig(window_pattern="SXL").validate()
    # pattern tiles, any length ok
    ModelConfig(n_layer=7, window_pattern="SSSL").validate()


def test_move_config_validate():
    with pytest.raises(AssertionError):
        MoveConfig(mode="bogus").validate()
    with pytest.raises(AssertionError):
        MoveConfig(gate_input="half").validate()
    MoveConfig(mode="lave", lave_layers="all").validate()
    MoveConfig(mode="lave", lave_layers="deep").validate()
    MoveConfig(mode="lave", lave_layers=[6, 7, 8]).validate()
    with pytest.raises(AssertionError):
        MoveConfig(mode="lave", lave_layers="bogus").validate()
    with pytest.raises(AssertionError):
        MoveConfig(mode="lave", lave_layers=[]).validate()
    with pytest.raises(AssertionError):
        MoveConfig(mode="lave", lave_layers=[-1, 3]).validate()


def test_resolve_move_off():
    r = resolve_move(MoveConfig(mode="off"), ModelConfig())
    assert r.is_off and r.num_slots == 0
    assert r.gate_out_dim(n_kv_head=6) == 0
    assert r.bank_params(65536, 768) == 0


def test_resolve_move_move():
    m = ModelConfig(n_layer=12, n_head=6, n_kv_head=6, n_embd=768)
    r = resolve_move(MoveConfig(mode="move", num_slots=6, gate_input="full", gated_standard=True), m)
    assert r.is_move and r.num_slots == 6
    assert r.gated_standard
    assert r.gate_in_dim == 768
    assert r.gate_out_dim(6) == 7 * 6  # (M+1) * n_kv_head
    assert r.bank_params(65536, 768) == 65536 * 6 * 768
    assert all(r.has_bank(i) for i in range(12))
    # auto slots = n_layer // 2
    r_auto = resolve_move(MoveConfig(mode="move"), m)
    assert r_auto.num_slots == 6
    # gate_input "12"
    r12 = resolve_move(MoveConfig(mode="move", gate_input="12"), m)
    assert r12.gate_in_dim == 12


def test_resolve_move_lave():
    m = ModelConfig(n_layer=12, n_head=6, n_kv_head=6, n_embd=768)
    r = resolve_move(MoveConfig(mode="lave", lave_slots=1, lave_layers="alt", gated_standard=False), m)
    assert r.is_lave
    assert r.lave_layer_indices == (1, 3, 5, 7, 9, 11)
    assert not r.gated_standard
    assert r.gate_out_dim(6) == 1 * 6
    assert r.bank_params(65536, 768) == 6 * 65536 * 1 * 768
    r_all = resolve_move(MoveConfig(mode="lave", lave_layers="all"), m)
    assert r_all.lave_layer_indices == tuple(range(12))
    # "deep": deepest half (6..11 for d12)
    r_deep = resolve_move(MoveConfig(mode="lave", lave_layers="deep"), m)
    assert r_deep.lave_layer_indices == (6, 7, 8, 9, 10, 11)
    # explicit list: sorted, deduplicated
    r_list = resolve_move(MoveConfig(mode="lave", lave_layers=[5, 3, 3, 9]), m)
    assert r_list.lave_layer_indices == (3, 5, 9)
    with pytest.raises(AssertionError):
        resolve_move(MoveConfig(mode="lave", lave_layers=[0, 12]), m)
    # odd n_layer: deep = range(n//2, n)
    r_odd = resolve_move(MoveConfig(mode="lave", lave_layers="deep"), ModelConfig(n_layer=7))
    assert r_odd.lave_layer_indices == (3, 4, 5, 6)
    r_gated = resolve_move(MoveConfig(mode="lave", gated_standard=True), m)
    assert r_gated.gate_out_dim(6) == 2 * 6


def test_lora_config():
    LoraConfig(rank=4, target_modules=["q", "k", "v", "proj", "fc", "mlp_proj"]).validate()
    with pytest.raises(AssertionError):
        LoraConfig(rank=0).validate()
    with pytest.raises(AssertionError):
        LoraConfig(target_modules=["attn"]).validate()


def test_jev_config_candidates():
    j = JevConfig()
    cands = j.candidate_words()
    assert cands["noul"] == ["да", "нет"]
    assert len(cands["choice"]) == 28  # DEFAULT_LETTERS (33-letter alphabet minus Й, Ё, Ъ, Ы, Ь)
    assert cands["score"] == [str(d) for d in range(10)]
    with pytest.raises(AssertionError):
        JevConfig(letter_tokens="А").validate()


def test_train_config_from_dict_roundtrip(tmp_path):
    raw = {
        "model": {"n_layer": 4, "n_embd": 128, "n_head": 4, "n_kv_head": 4},
        "move": {"mode": "move", "num_slots": 2},
        "max_steps": 10,
        "device_batch_size": 2,
    }
    cfg = TrainConfig.from_dict(raw)
    assert cfg.model.n_layer == 4
    assert cfg.move.mode == "move"
    p = str(tmp_path / "c.yaml")
    save_config(cfg, p)
    cfg2 = load_config(p)
    assert cfg2.as_dict() == cfg.as_dict()


def test_train_config_unknown_keys():
    with pytest.raises(ValueError, match="unknown top-level keys"):
        TrainConfig.from_dict({"bogus_key": 1})
    with pytest.raises(ValueError, match="unknown config keys"):
        TrainConfig.from_dict({"model": {"bogus": 1}})
