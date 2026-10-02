"""
Utilities for saving and loading model/optim/adapter checkpoints.
Simplified port of nanochat's checkpoint_manager: single rank, no legacy patching.
"""
import os
import re
import json
import torch

from jevelike.common import get_base_dir, get_checkpoints_dir, get_logger
from jevelike.configs import ModelConfig, MoveConfig
from jevelike.move import resolve_move
from jevelike.gpt import GPT
from jevelike.tokenizer import get_tokenizer

logger = get_logger(__name__)


def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    torch.save(model_data, model_path)
    logger.info(f"Saved model parameters to: {model_path}")
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=2)
    logger.info(f"Saved metadata to: {meta_path}")
    if optimizer_data is not None:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}.pt")
        torch.save(optimizer_data, optimizer_path)
        logger.info(f"Saved optimizer state to: {optimizer_path}")


def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False):
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data


def build_model(checkpoint_dir, step, device, phase, tokenizer_dir=None):
    """
    Build a model from a checkpoint directory. Returns:
    - model (uncompiled), tokenizer, meta data saved during training.
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    model_data, optimizer_data, meta_data = load_checkpoint(
        checkpoint_dir, step, device, load_optimizer=False
    )
    if device.type in {"cpu", "mps"}:
        # Convert bf16 tensors to fp32 for CPU
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

    model_config = ModelConfig.from_dict(meta_data["model_config"])
    move_config = MoveConfig.from_dict(meta_data.get("move_config", {"mode": "off"}))
    logger.info(f"Building model with config: {model_config.as_dict()}, move: {move_config.as_dict()}")
    move = resolve_move(move_config, model_config)

    with torch.device("meta"):
        model = GPT(model_config, move=move)
    model.to_empty(device=device)
    model.init_weights()  # needed to init the rotary embeddings
    model.load_state_dict(model_data, strict=True, assign=True)
    if phase == "eval":
        model.eval()
    else:
        model.train()

    tokenizer = get_tokenizer(tokenizer_dir)
    assert tokenizer.get_vocab_size() == model_config.vocab_size, (
        f"Tokenizer vocab size {tokenizer.get_vocab_size()} does not match "
        f"model config vocab size {model_config.vocab_size}"
    )
    return model, tokenizer, meta_data


def find_largest_model(checkpoints_dir):
    """Guess the model tag: take the biggest model available (d<number>), else most recent."""
    model_tags = [f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            candidates.append((int(match.group(1)), model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    """Look into checkpoint_dir and find model_<step>.pt with the highest step."""
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if re.search(r'model_(\d+)\.pt$', f)]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = max(int(f.split("_")[-1].split(".")[0]) for f in checkpoint_files)
    return last_step


def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None, tokenizer_dir=None):
    if model_tag is None:
        model_tag = find_largest_model(checkpoints_dir)
        logger.info(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        step = find_last_step(checkpoint_dir)
    logger.info(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase, tokenizer_dir)
    return model, tokenizer, meta_data


def load_model(device, phase, model_tag=None, step=None, tokenizer_dir=None):
    """Load a base model from the default checkpoints dir (<base_dir>/checkpoints)."""
    return load_model_from_dir(get_checkpoints_dir(), device, phase, model_tag, step, tokenizer_dir)


def load_optimizer_state(checkpoint_dir, step, device):
    optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}.pt")
    if not os.path.exists(optimizer_path):
        logger.info(f"Optimizer checkpoint not found: {optimizer_path}")
        return None
    logger.info(f"Loading optimizer state from {optimizer_path}")
    return torch.load(optimizer_path, map_location=device)


# -----------------------------------------------------------------------------
# Jev LoRA adapter checkpoints (trainable adapter only, no base weights)

def save_adapter(path, step, adapter_state, meta_data=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"step": step, "adapter": adapter_state, "meta": meta_data or {}}
    torch.save(payload, path)
    logger.info(f"Saved adapter to: {path}")


def load_adapter(path, device):
    payload = torch.load(path, map_location=device)
    return payload["adapter"], payload.get("step", None), payload.get("meta", {})


def find_last_adapter(adi_dir):
    """Look in a dir for adapter_<step>.pt with the highest step."""
    files = [f for f in os.listdir(adi_dir) if re.search(r'adapter_(\d+)\.pt$', f)]
    if not files:
        raise FileNotFoundError(f"No adapter checkpoints found in {adi_dir}")
    return max(int(f.split("_")[-1].split(".")[0]) for f in files)
