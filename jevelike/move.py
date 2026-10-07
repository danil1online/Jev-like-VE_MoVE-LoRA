"""
MoVE (Value Memory with Mixture of Value Experts, arXiv:2601.22887) and LaVE
value-embedding machinery:

- MoveBank: a single shared (vocab x M x kv_dim) parameter bank. Looked up ONCE
  per forward with the input token ids and shared by all layers.
- ResolvedMove: a fully-resolved view of MoveConfig + ModelConfig (slot count,
  gate in/out dims, which layers have a bank).
- mix_value: the gated value-mixing formula, factorized for unit testing.

Conventions (B=batch, T=seq, H=n_kv_head, M=slots, D=head_dim):
  move: v (B,T,H,D), ve (B,T,H,M,D), gate_logits (B,T,H,M+1) [gated] or (B,T,H,M)
    V  = g0 * V_std + sum_m g_m * M_m
  lave: v (B,T,H,D), ve (B,T,H,D), gate_logits (B,T,H) [ungated] or (B,T,H,2)
    ungated: V = V_std + g * M_1
    gated:   V = g0 * V_std + g1 * M_1
"""
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from jevelike.common import COMPUTE_DTYPE


# -----------------------------------------------------------------------------
# Resolved move config
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedMove:
    mode: str                       # "off" | "lave" | "move"
    num_slots: int                  # M: total slots (move) / slots per layer (lave)
    gate_scale: float
    gate_in_dim: int
    gated_standard: bool
    lave_layer_indices: Tuple[int, ...]

    @property
    def is_off(self):
        return self.mode == "off"

    @property
    def is_move(self):
        return self.mode == "move"

    @property
    def is_lave(self):
        return self.mode == "lave"

    def has_bank(self, layer_idx: int) -> bool:
        """Whether a layer gets value embeddings."""
        if self.is_move:
            return True
        if self.is_lave:
            return layer_idx in self.lave_layer_indices
        return False

    def gate_out_dim(self, n_kv_head: int) -> int:
        """Number of gate logits per layer (across all kv heads)."""
        if self.is_move:
            return (self.num_slots + 1) * n_kv_head
        if self.is_lave:
            return (2 if self.gated_standard else 1) * n_kv_head
        return 0

    def bank_params(self, vocab_padded: int, kv_dim: int) -> int:
        if self.is_move:
            return vocab_padded * self.num_slots * kv_dim
        if self.is_lave:
            return len(self.lave_layer_indices) * vocab_padded * self.num_slots * kv_dim
        return 0


def resolve_move(move_cfg, model_cfg) -> ResolvedMove:
    """Resolve a MoveConfig against a ModelConfig into concrete values."""
    move_cfg.validate()
    if move_cfg.mode == "off":
        return ResolvedMove(mode="off", num_slots=0, gate_scale=1.0,
                            gate_in_dim=0, gated_standard=False, lave_layer_indices=())

    if move_cfg.gate_input == "full":
        gate_in_dim = model_cfg.n_embd
    else:  # "12"
        gate_in_dim = 12
        assert 12 <= model_cfg.n_embd, "gate_input='12' requires n_embd >= 12"

    gated_standard = move_cfg.gated_standard
    if gated_standard is None:
        gated_standard = move_cfg.mode == "move"

    if move_cfg.mode == "move":
        if move_cfg.num_slots > 0:
            m = move_cfg.num_slots
        else:
            m = model_cfg.n_layer // 2
        assert m >= 1, "move mode requires at least 1 slot"
        return ResolvedMove(mode="move", num_slots=m, gate_scale=move_cfg.gate_scale,
                            gate_in_dim=gate_in_dim, gated_standard=gated_standard,
                            lave_layer_indices=())

    # lave
    slots = move_cfg.lave_slots if move_cfg.lave_slots > 0 else 1
    ll = move_cfg.lave_layers
    n_layer = model_cfg.n_layer
    if ll == "all":
        layers = tuple(range(n_layer))
    elif ll == "deep":
        # deepest half of the layers (e.g. 6..11 for d12), per nanochat discussion #463
        layers = tuple(range(n_layer // 2, n_layer))
    elif isinstance(ll, (list, tuple)):
        layers = tuple(sorted(set(int(i) for i in ll)))
        assert layers and min(layers) >= 0 and max(layers) < n_layer, \
            f"lave_layers {tuple(ll)} out of range for n_layer={n_layer}"
    else:  # "alt": same pattern as nanochat value_embeds (L-1, L-3, ...)
        layers = tuple(i for i in range(n_layer) if i % 2 == (n_layer - 1) % 2)
    return ResolvedMove(mode="lave", num_slots=slots, gate_scale=move_cfg.gate_scale,
                        gate_in_dim=gate_in_dim, gated_standard=gated_standard,
                        lave_layer_indices=layers)


# -----------------------------------------------------------------------------
# Value bank
# -----------------------------------------------------------------------------

class MoveBank(nn.Module):
    """Shared value-embedding bank E: (vocab_padded x num_slots x kv_dim).

    A single lookup per forward:  bank(idx) -> (B, T, M, kv_dim).
    """

    def __init__(self, vocab_size: int, num_slots: int, kv_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_slots = num_slots
        self.kv_dim = kv_dim
        self.weight = nn.Parameter(torch.empty(vocab_size, num_slots, kv_dim))
        if COMPUTE_DTYPE != torch.float16:
            self.to(dtype=COMPUTE_DTYPE)

    def forward(self, idx):
        return self.weight[idx]  # (B, T, M, kv_dim)

    def init_weights(self, s=1.0):
        # same init convention as value embeddings: uniform with std ~ s (bound = s)
        nn.init.uniform_(self.weight, -s, s)


# -----------------------------------------------------------------------------
# Value mixing
# -----------------------------------------------------------------------------

def mix_value(v, ve, gate_logits, gate_scale, gated_standard):
    """Gated mixing of the standard value path v with value-embedding slots.

    Shapes (see module docstring):
      move: v (B,T,H,D), ve (B,T,H,M,D), gate_logits (B,T,H,M+1) or (B,T,H,M)
      lave: v (B,T,H,D), ve (B,T,H,D), gate_logits (B,T,H) or (B,T,H,2)

    Returns the mixed values with the same shape as v.
    """
    z = gate_logits.float()
    g = gate_scale * torch.sigmoid(z)
    if ve.dim() == 5:
        # move: M slots
        if gated_standard:
            g0, gm = g[..., :1], g[..., 1:]
            v = v * g0
        else:
            gm = g
        v = v + (gm.unsqueeze(-1) * ve).sum(dim=3)
    else:
        # lave: single slot
        if gated_standard:
            assert g.shape[-1] == 2, f"gated lave expects 2 gate channels, got {g.shape[-1]}"
            g0, g1 = g[..., :1], g[..., 1:]
            v = v * g0 + ve * g1
        else:
            v = v + ve * g.unsqueeze(-1)
    return v.to(dtype=v.dtype)
