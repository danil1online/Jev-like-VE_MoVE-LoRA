"""
Train the Jev-Like LoRA adapter on top of a frozen base model.

Usage:
    jev-train --config configs/jev_lora_d12.yaml [--set key=value ...]
    jev-eval  --config configs/jev_lora_d12.yaml [--adapter adapter_001000.pt]

Jev data: JSONL files <jev_data_dir>/train.jsonl and val.jsonl with
{"text": ..., "question": ..., "answer": ..., "task": "noul"|"choice"|"score"}.
Tokenization is cached next to the data as train_cache.pt / val_cache.pt.
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import time

import torch

from jevelike.common import (
    COMPUTE_DTYPE, compute_init, compute_cleanup, get_base_dir,
    get_runs_dir, print0, setup_default_logging, JsonlLogger,
)
from jevelike.configs import load_config, save_config
from jevelike.checkpoint import find_last_step, build_model, save_adapter, load_adapter
from jevelike.jev import (
    JevAdapter, JevRenderer, IGNORE_INDEX, calibration_metrics,
    calibrate_probs, fit_temperature,
)
from jevelike.tokenizer import get_tokenizer

TASK_ORDER = ("noul", "choice", "score")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--device-type", type=str, default="")
    ap.add_argument("--adapter", type=str, default=None,
                    help="adapter checkpoint to evaluate (for jev-eval)")
    return ap.parse_args()


def apply_overrides(cfg, pairs):
    for pair in pairs:
        key, _, value = pair.partition("=")
        value = json.loads(value)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = getattr(node, part)
        setattr(node, parts[-1], value)


def load_base_from_path(path, device, tokenizer_dir=None):
    """path: checkpoint dir (uses last step) or direct model_XXXXXX.pt file."""
    if path.endswith(".pt"):
        checkpoint_dir = os.path.dirname(os.path.abspath(path))
        step = int(re.search(r"model_(\d+)\.pt$", path).group(1))
    else:
        checkpoint_dir = path
        step = find_last_step(checkpoint_dir)
    print0(f"Loading base model from {checkpoint_dir} step {step}")
    return build_model(checkpoint_dir, step, device, "eval", tokenizer_dir)


def load_jev_items(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def cache_key(items):
    h = hashlib.md5()
    for it in items:
        h.update(json.dumps([it["text"], it["question"], it["answer"], it.get("task", "noul")],
                            ensure_ascii=False).encode("utf-8"))
    return h.hexdigest()[:12]


def render_cache(renderer, items, cache_path):
    """Pre-tokenize items to (ids, target, task) and cache to disk."""
    key = cache_key(items)
    ckpt = os.path.join(os.path.dirname(cache_path), "cache")
    os.makedirs(ckpt, exist_ok=True)
    path = os.path.join(ckpt, os.path.basename(cache_path) + f".{key}.pt")
    if os.path.exists(path):
        print0(f"Using cached rendering: {path}")
        return torch.load(path, weights_only=True)
    print0(f"Tokenizing {len(items)} items -> {path}")
    rendered = []
    for it in items:
        ids, target = renderer.render_one(it["text"], it["question"], it["answer"])
        rendered.append({"ids": ids, "target": int(target), "task": it.get("task", "noul"),
                         "answer": it["answer"]})
    torch.save(rendered, path)
    return rendered


def make_cached_batch(rendered):
    L = max(len(r["ids"]) for r in rendered)
    B = len(rendered)
    x = torch.zeros(B, L, dtype=torch.long)
    labels = torch.full((B, L), IGNORE_INDEX, dtype=torch.long)
    meta = []
    for i, r in enumerate(rendered):
        x[i, :len(r["ids"])] = torch.tensor(r["ids"], dtype=torch.long)
        labels[i, len(r["ids"]) - 1] = r["target"]
        meta.append({"task": r["task"], "answer": r["answer"], "pos": len(r["ids"]) - 1})
    return x, labels, meta


def task_slices(renderer):
    slices, start = {}, 0
    for task in TASK_ORDER:
        words = renderer.words_by_task[task]
        slices[task] = (start, start + len(words))
        start += len(words)
    return slices


def check_adapter_meta(a_meta, cfg, path):
    """Warn if the adapter was trained on a different base than the config points to."""
    saved_base = a_meta.get("base_checkpoint")
    if saved_base and cfg.jev_base_checkpoint and \
            os.path.normpath(saved_base) != os.path.normpath(cfg.jev_base_checkpoint):
        print0(f"WARNING: adapter {path} was trained on base {saved_base!r}, "
               f"but config points to {cfg.jev_base_checkpoint!r}; the config base is used.")


def collect_probs(adapter, rendered, device, batch_size=32):
    """Per-task (probs, target) tensors over rendered items (answer positions only).
    Both rows of a batch share one forward pass; adapter is left in eval mode."""
    renderer = adapter.renderer
    slices = task_slices(renderer)
    per_task = {t: {"probs": [], "target": []} for t in TASK_ORDER}
    adapter.eval()
    with torch.inference_mode():
        for i in range(0, len(rendered), batch_size):
            chunk = rendered[i:i + batch_size]
            x, labels, meta = make_cached_batch(chunk)
            x = x.to(device)
            probs = adapter.probs(x)
            for j, m in enumerate(meta):
                task = m["task"]
                s, e = slices[task]
                # restricted softmax over this task's candidates (the global
                # adapter softmax spreads mass over all 40 candidates)
                p = probs[j, m["pos"], s:e]
                p = p / p.sum(-1, keepdim=True)
                cand_idx = renderer.word_to_cand[m["answer"]]
                per_task[task]["probs"].append(p.cpu())
                per_task[task]["target"].append(cand_idx - s)
    out = {}
    for task in TASK_ORDER:
        if per_task[task]["probs"]:
            out[task] = (torch.stack(per_task[task]["probs"]),
                         torch.tensor(per_task[task]["target"], dtype=torch.long))
    return out


def evaluate_adapter(adapter, rendered, device, batch_size=32):
    """Per-task accuracy/ECE/Brier/logloss on rendered items."""
    bins = adapter.jev_cfg.ece_bins
    return {task: calibration_metrics(probs, target, bins=bins)
            for task, (probs, target) in collect_probs(adapter, rendered, device, batch_size).items()}


def main():
    args = parse_args()
    setup_default_logging()
    cfg = load_config(args.config)
    apply_overrides(cfg, args.set)
    cfg.lora.validate()
    cfg.jev.validate()

    device_type = args.device_type if args.device_type else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_type)
    compute_init(device_type)
    torch.manual_seed(cfg.seed)
    print0(f"Device: {device_type} | COMPUTE_DTYPE: {COMPUTE_DTYPE}")

    run_name = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(get_runs_dir(), f"jev_{cfg.model_tag}", run_name)
    os.makedirs(run_dir, exist_ok=True)
    logger = JsonlLogger(os.path.join(run_dir, "log.jsonl"))
    save_config(cfg, os.path.join(run_dir, "config.yaml"))

    tokenizer = get_tokenizer()
    model, _, meta_data = load_base_from_path(cfg.jev_base_checkpoint, device)
    print0(f"Base model config: {json.dumps(meta_data['model_config'])}")
    print0(f"Base move config: {json.dumps(meta_data.get('move_config', {}))}")

    adapter = JevAdapter(model, cfg.lora, cfg.jev, tokenizer)
    adapter.train()  # base is frozen anyway; enables LoRA dropout in train mode
    n_train = sum(p.numel() for p in adapter.trainable_params())
    print0(f"Adapter trainable params: {n_train:,}")

    # data
    data_dir = cfg.jev_data_dir
    train_path = os.path.join(data_dir, "train.jsonl")
    val_path = os.path.join(data_dir, "val.jsonl")
    train_items = load_jev_items(train_path)
    val_items = load_jev_items(val_path)
    print0(f"Jev data: {len(train_items)} train / {len(val_items)} val items")
    train_rendered = render_cache(adapter.renderer, train_items, "train_cache.pt")
    val_rendered = render_cache(adapter.renderer, val_items, "val_cache.pt")

    # resume
    resume_step = cfg.resume_from_step
    if resume_step > 0:
        adapter_path = os.path.join(run_dir, f"adapter_{resume_step:06d}.pt")
        if not os.path.exists(adapter_path):
            adapter_path = os.path.join(get_base_dir(), "jev_checkpoints", cfg.model_tag,
                                        f"adapter_{resume_step:06d}.pt")
        if os.path.exists(adapter_path):
            adapter_state, step, a_meta = load_adapter(adapter_path, device)
            adapter.load_adapter_state(adapter_state)
            check_adapter_meta(a_meta, cfg, adapter_path)
            adapter.train()
            print0(f"Resumed adapter from {adapter_path} step {step}")

    # optimizer: single AdamW group over all trainable adapter params
    optimizer = torch.optim.AdamW(adapter.trainable_params(), lr=cfg.jev_lr,
                                  betas=(0.8, 0.95), weight_decay=cfg.jev_weight_decay)

    num_iterations = cfg.jev_num_iterations
    warmup_iters = max(1, int(round(cfg.jev_warmup * num_iterations)))

    def get_lr_multiplier(it):
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        progress = (it - warmup_iters) / max(1, num_iterations - warmup_iters)
        progress = min(1.0, progress)
        return cfg.jev_final_lr_frac + (1 - cfg.jev_final_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress))

    rng = random.Random(cfg.seed)
    n = len(train_rendered)
    print0(f"Training adapter for {num_iterations:,} steps (batch {cfg.jev_batch_size}, lr {cfg.jev_lr})")

    step = resume_step
    while step < num_iterations:
        # batch
        batch_idx = [rng.randrange(n) for _ in range(cfg.jev_batch_size)]
        batch_rendered = [train_rendered[i] for i in batch_idx]
        x, labels, meta = make_cached_batch(batch_rendered)
        x, labels = x.to(device), labels.to(device)

        lrm = get_lr_multiplier(step)
        for group in optimizer.param_groups:
            group["lr"] = cfg.jev_lr * lrm

        t0 = time.time()
        loss = adapter(x, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_f = loss.item()
        dt = time.time() - t0

        log_payload = {"step": step, "train_loss": loss_f, "lrm": lrm, "dt": dt}
        if (cfg.jev_eval_every > 0 and (step == num_iterations - 1 or
                step > 0 and step % cfg.jev_eval_every == 0)) or step == 0:
            results = evaluate_adapter(adapter, val_rendered, device)
            adapter.train()  # evaluate_adapter switches to eval mode
            for task, m in results.items():
                log_payload[f"val/{task}/acc"] = m["accuracy"]
                log_payload[f"val/{task}/ece"] = m["ece"]
                log_payload[f"val/{task}/brier"] = m["brier"]
                log_payload[f"val/{task}/logloss"] = m["logloss"]
            accs = " | ".join(f"{t}: {results[t]['accuracy']:.4f}" for t in TASK_ORDER if t in results)
            print0(f"step {step:05d}/{num_iterations:05d} | loss: {loss_f:.6f} | {accs}")
        else:
            if step % 10 == 0:
                print0(f"step {step:05d}/{num_iterations:05d} | loss: {loss_f:.6f} | lrm: {lrm:.3f} | dt: {dt * 1000:.0f}ms")
        logger.log(**log_payload)

        if (cfg.jev_save_every > 0 and step > 0 and step % cfg.jev_save_every == 0) or step == num_iterations - 1:
            out_dir = os.path.join(get_base_dir(), "jev_checkpoints", cfg.model_tag)
            save_adapter(os.path.join(out_dir, f"adapter_{step:06d}.pt"), step,
                         adapter.adapter_state_dict(),
                         meta_data={
                             "base_checkpoint": cfg.jev_base_checkpoint,
                             "base_meta": {k: meta_data[k] for k in ("model_config", "move_config") if k in meta_data},
                             "lora_config": cfg.lora.as_dict(),
                             "jev_config": cfg.jev.as_dict(),
                         })
        step += 1

    # post-hoc per-task temperature calibration on val
    out_dir = os.path.join(get_base_dir(), "jev_checkpoints", cfg.model_tag)
    calib = {}
    calib_payload = {"done": True, "total_training_time": time.time()}
    bins = cfg.jev.ece_bins
    for task, (probs, target) in collect_probs(adapter, val_rendered, device).items():
        t, _ = fit_temperature(probs, target)
        before = calibration_metrics(probs, target, bins=bins)
        after = calibration_metrics(calibrate_probs(probs, t), target, bins=bins)
        calib[task] = t
        calib_payload[f"calib/{task}/T"] = t
        calib_payload[f"calib/{task}/logloss_before"] = before["logloss"]
        calib_payload[f"calib/{task}/logloss_after"] = after["logloss"]
        calib_payload[f"calib/{task}/ece_before"] = before["ece"]
        calib_payload[f"calib/{task}/ece_after"] = after["ece"]
        print0(f"calibration[{task}]: T={t:.3f} | logloss {before['logloss']:.4f} -> {after['logloss']:.4f} | "
               f"ece {before['ece']:.4f} -> {after['ece']:.4f} | n={target.numel()}")
    calib_path = os.path.join(out_dir, "calibration.json")
    with open(calib_path, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2)
    print0(f"Saved calibration to {calib_path}")
    logger.log(**calib_payload)
    logger.close()
    compute_cleanup(device_type)


def eval_main():
    args = parse_args()
    setup_default_logging()
    cfg = load_config(args.config)
    apply_overrides(cfg, args.set)

    device_type = args.device_type if args.device_type else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_type)
    compute_init(device_type)

    tokenizer = get_tokenizer()
    model, _, meta_data = load_base_from_path(cfg.jev_base_checkpoint, device)
    adapter = JevAdapter(model, cfg.lora, cfg.jev, tokenizer)

    adapter_dir = os.path.join(get_base_dir(), "jev_checkpoints", cfg.model_tag)
    if args.adapter is not None:
        adapter_path = args.adapter
    else:
        files = [f for f in os.listdir(adapter_dir) if re.search(r"adapter_(\d+)\.pt$", f)]
        if not files:
            raise FileNotFoundError(f"No adapter checkpoints in {adapter_dir}")
        best = max(files, key=lambda f: int(f.split("_")[-1].split(".")[0]))
        adapter_path = os.path.join(adapter_dir, best)
    print0(f"Loading adapter from {adapter_path}")
    adapter_state, step, a_meta = load_adapter(adapter_path, device)
    adapter.load_adapter_state(adapter_state)
    check_adapter_meta(a_meta, cfg, adapter_path)
    adapter.eval()
    print0(f"Adapter step: {step}")

    data_dir = cfg.jev_data_dir
    val_items = load_jev_items(os.path.join(data_dir, "val.jsonl"))
    val_rendered = render_cache(adapter.renderer, val_items, "val_cache.pt")
    raw = collect_probs(adapter, val_rendered, device)
    calib_path = os.path.join(adapter_dir, "calibration.json")
    calib = {}
    if os.path.exists(calib_path):
        with open(calib_path, "r", encoding="utf-8") as f:
            calib = json.load(f)
        print0(f"Using calibration from {calib_path}: {calib}")
    bins = cfg.jev.ece_bins
    print0(f"\nStep {step} | Jev evaluation:")
    for task in TASK_ORDER:
        if task not in raw:
            continue
        probs, target = raw[task]
        m = calibration_metrics(probs, target, bins=bins)
        line = (f"  {task:8s} | acc: {m['accuracy']:.4f} | ece: {m['ece']:.4f} | "
                f"brier: {m['brier']:.4f} | logloss: {m['logloss']:.4f} | n: {m['n']}")
        if task in calib:
            m2 = calibration_metrics(calibrate_probs(probs, calib[task]), target, bins=bins)
            line += (f" | T={calib[task]:.3f}: logloss {m['logloss']:.4f}->{m2['logloss']:.4f}, "
                     f"ece {m['ece']:.4f}->{m2['ece']:.4f}")
        print0(line)
    compute_cleanup(device_type)


if __name__ == "__main__":
    main()
