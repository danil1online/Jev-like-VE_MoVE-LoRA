"""
LoRA (Low-Rank Adaptation) for Jev adapters.

Only the low-rank branches are trainable; the base model is frozen.
Targets are the attention projections (q/k/v/proj) and optionally MLP layers.
"""
import math
import torch
import torch.nn as nn

from jevelike.configs import LoraConfig

# target name -> (block_attr, module_attr)
TARGET_MAP = {
    "q": ("attn", "c_q"),
    "k": ("attn", "c_k"),
    "v": ("attn", "c_v"),
    "proj": ("attn", "c_proj"),
    "fc": ("mlp", "c_fc"),
    "mlp_proj": ("mlp", "c_proj"),
}


class LoRALinear(nn.Module):
    """
    Wraps a base linear layer: y = W x + (alpha/rank) * B A x.
    A: (rank, in_f), kaiming init; B: (out_f, rank), zeros -> adapter is
    an exact no-op at initialization.
    """

    def __init__(self, base: nn.Module, rank: int = 16, alpha: float = 32.0,
                 dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear), "base must be a linear layer"
        self.base = base
        in_f, out_f = base.in_features, base.out_features
        assert rank > 0
        self.rank = rank
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B = nn.Parameter(torch.empty(out_f, rank))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def extra_repr(self):
        return f"in_features={self.base.in_features}, out_features={self.base.out_features}, rank={self.rank}, scale={self.scale}"

    def forward(self, x):
        y = self.base(x)
        x32 = x.float()
        delta = x32 @ self.lora_A.t()
        delta = self.lora_dropout(delta)
        delta = delta @ self.lora_B.t()
        return y + delta.to(y.dtype) * self.scale


def apply_lora(model, lora_cfg: LoraConfig):
    """Replace target linears inside model.transformer.h with LoRALinear. Returns list of adapters."""
    lora_cfg.validate()
    adapters = []
    for block in model.transformer.h:
        for target in lora_cfg.target_modules:
            if target not in TARGET_MAP:
                raise ValueError(f"Unknown LoRA target {target!r}. Valid: {sorted(TARGET_MAP)}")
            block_attr, module_attr = TARGET_MAP[target]
            module = getattr(block, block_attr)
            linear = getattr(module, module_attr)
            adapter = LoRALinear(linear, rank=lora_cfg.rank, alpha=lora_cfg.alpha,
                                 dropout=lora_cfg.dropout)
            setattr(module, module_attr, adapter)
            adapters.append(adapter)
    return adapters


def freeze_base(model):
    """Freeze all base model parameters (LoRA A/B stay trainable)."""
    for p in model.parameters():
        p.requires_grad = False
    for adapter in model.modules():
        if isinstance(adapter, LoRALinear):
            adapter.lora_A.requires_grad = True
            adapter.lora_B.requires_grad = True


def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def lora_num_params(model):
    n = 0
    for adapter in model.modules():
        if isinstance(adapter, LoRALinear):
            n += adapter.lora_A.numel() + adapter.lora_B.numel()
    return n
