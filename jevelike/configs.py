"""
Configuration dataclasses for jevelike: model architecture, MoVE/LaVE variants,
LoRA adapter, Jev task, and training. Loadable from YAML.
"""
from dataclasses import dataclass, field, fields, asdict
from typing import Optional, List

import yaml

VALID_MOVE_MODES = ("off", "lave", "move")
VALID_GATE_INPUTS = ("full", "12")
VALID_LAVE_LAYERS = ("alt", "all")


def _from_dict(dc_cls, raw):
    if raw is None:
        return None
    known = {f.name for f in fields(dc_cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{dc_cls.__name__}: unknown config keys: {sorted(unknown)}")
    return dc_cls(**raw)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

@dataclass
class ModelConfig:
    sequence_len: int = 2048
    vocab_size: int = 65536
    padding_multiple: int = 64
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"
    softcap: float = 15.0

    @classmethod
    def from_dict(cls, raw):
        return _from_dict(cls, raw)

    @property
    def head_dim(self):
        return self.n_embd // self.n_head

    @property
    def kv_dim(self):
        return self.head_dim * self.n_kv_head

    @property
    def mlp_dim(self):
        return 4 * self.n_embd

    def validate(self):
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head (GQA)"
        # pattern is tiled across layers (e.g. "SSSL" for any n_layer), so no length check
        assert set(self.window_pattern) <= set("SL"), \
            f"window_pattern may only contain 'S' or 'L': {self.window_pattern!r}"

    def as_dict(self):
        return asdict(self)


# -----------------------------------------------------------------------------
# MoVE / LaVE
# -----------------------------------------------------------------------------

@dataclass
class MoveConfig:
    """
    mode:
      - "off":  no value embeddings (vanilla nanochat-style transformer)
      - "lave": per-layer value-embedding bank (LaVE), on alternating or all layers,
                standard path ungated by default (gate only on the extra term)
      - "move": a single shared (VOCAB x M x kv_dim) bank looked up once per forward,
                routed per-layer by a learned gate, both standard and extra paths gated
    num_slots:
      - for "move": total number of slots M (e.g. x1 = n_layer // 2, x2 = n_layer,
        x4 = 2 * n_layer). -1 means "auto" (n_layer // 2).
      - for "lave": slots per layer (default 1, as in LaVE). -1 means auto (1).
    gate_input:
      - "full": gate reads the full post-norm attention input (d -> H*(M+1))
      - "12":   gate reads the first 12 dims (as in nanochat's ve_gate)
    gate_scale:
      multiplier on sigmoid outputs (MoVE paper uses 2.0).
    gated_standard:
      whether the standard value path is also multiplied by a gate.
      None -> default per mode: True for "move" (paper), False for "lave".
    lave_layers:
      which layers get a bank in "lave" mode: "alt" (L-1, L-3, ...) or "all".
    """
    mode: str = "off"
    num_slots: int = -1
    lave_slots: int = 1
    gate_scale: float = 2.0
    gate_input: str = "full"
    gated_standard: Optional[bool] = None
    lave_layers: str = "alt"

    @classmethod
    def from_dict(cls, raw):
        return _from_dict(cls, raw)

    def validate(self):
        assert self.mode in VALID_MOVE_MODES, f"mode must be one of {VALID_MOVE_MODES}, got {self.mode!r}"
        assert self.gate_input in VALID_GATE_INPUTS, f"gate_input must be one of {VALID_GATE_INPUTS}"
        assert self.lave_layers in VALID_LAVE_LAYERS, f"lave_layers must be one of {VALID_LAVE_LAYERS}"
        assert self.gate_scale > 0

    def as_dict(self):
        return asdict(self)


# -----------------------------------------------------------------------------
# LoRA adapter (Jev-Like)
# -----------------------------------------------------------------------------

VALID_LORA_TARGETS = ("q", "k", "v", "proj", "fc", "mlp_proj")


@dataclass
class LoraConfig:
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: ["q", "k", "v"])

    @classmethod
    def from_dict(cls, raw):
        return _from_dict(cls, raw)

    def validate(self):
        assert self.rank > 0
        for t in self.target_modules:
            assert t in VALID_LORA_TARGETS, f"target must be one of {VALID_LORA_TARGETS}, got {t!r}"

    def as_dict(self):
        return asdict(self)


# -----------------------------------------------------------------------------
# Jev task
# -----------------------------------------------------------------------------

VALID_JEV_TASKS = ("noul", "choice", "score")

DEFAULT_LETTERS = "АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЭЮЯ"


@dataclass
class JevConfig:
    """Typed-decision task: prompt ends with <|a|>, the model emits ONE answer token
    from a small candidate set (da/net, letters A..Ya, digits 0..9).
    Answer words are ordinary vocabulary tokens (NOT special tokens), resolved
    at runtime via tokenizer.encode_single_token."""
    yes_token: str = "да"
    no_token: str = "нет"
    letter_tokens: str = DEFAULT_LETTERS
    digit_tokens: str = "0123456789"
    max_ctx_len: int = 1024
    max_q_len: int = 256
    max_examples_len: int = 1280
    temperature: float = 1.0
    ece_bins: int = 15

    @classmethod
    def from_dict(cls, raw):
        return _from_dict(cls, raw)

    def candidate_words(self):
        return {
            "noul": [self.yes_token, self.no_token],
            "choice": list(self.letter_tokens),
            "score": list(self.digit_tokens),
        }

    def validate(self):
        assert len(self.letter_tokens) >= 2, "need at least 2 choice candidates"
        assert len(self.digit_tokens) >= 2, "need at least 2 score candidates"

    def as_dict(self):
        return asdict(self)


# -----------------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------------

@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    move: MoveConfig = field(default_factory=MoveConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    jev: JevConfig = field(default_factory=JevConfig)

    # --- base pretraining ---
    model_tag: str = "d12"
    seed: int = 42
    max_steps: int = 50_000
    warmup_steps: int = 40
    warmdown_ratio: float = 0.65
    final_lr_frac: float = 0.05
    start_from_scratch: bool = True
    resume_from_step: int = 0
    save_every: int = 0
    eval_every: int = 250
    sample_every: int = 0

    device_batch_size: int = 32
    total_batch_size: int = -1      # -1 = auto (param-data scaling)
    target_param_data_ratio: float = 12.0

    matrix_lr: float = 0.02
    unembedding_lr: float = 0.004
    embedding_lr: float = 0.2
    scalar_lr: float = 0.5
    move_gate_lr: float = 0.005
    weight_decay: float = 0.28
    grad_clip: float = 1.0

    # --- jev adapter training ---
    jev_base_checkpoint: str = ""   # path to base checkpoint dir (model_XXXXXX.pt)
    jev_data_dir: str = ""          # dir with train.jsonl / val.jsonl
    jev_num_iterations: int = 2000
    jev_warmup_steps: int = 50
    jev_warmdown_ratio: float = 0.5
    jev_final_lr_frac: float = 0.1
    jev_batch_size: int = 16
    jev_lr: float = 1e-4
    jev_weight_decay: float = 0.0
    jev_warmup: float = 0.02
    jev_eval_every: int = 200
    jev_save_every: int = 1000

    @classmethod
    def from_dict(cls, raw):
        if not isinstance(raw, dict):
            raise ValueError("Top-level config must be a mapping")
        raw = dict(raw)
        for key, dc_cls in (("model", ModelConfig), ("move", MoveConfig),
                            ("lora", LoraConfig), ("jev", JevConfig)):
            if key in raw:
                raw[key] = dc_cls.from_dict(raw[key])
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"TrainConfig: unknown top-level keys: {sorted(unknown)}")
        cfg = cls(**raw)
        cfg.validate()
        return cfg

    def validate(self):
        self.model.validate()
        self.move.validate()
        self.lora.validate()
        self.jev.validate()

    def as_dict(self):
        return {
            "model": self.model.as_dict(),
            "move": self.move.as_dict(),
            "lora": self.lora.as_dict(),
            "jev": self.jev.as_dict(),
            **{f.name: getattr(self, f.name) for f in fields(self)
               if f.name not in ("model", "move", "lora", "jev")},
        }


def load_config(path) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return TrainConfig.from_dict(raw)


def save_config(cfg: TrainConfig, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.as_dict(), f, allow_unicode=True, sort_keys=False)
