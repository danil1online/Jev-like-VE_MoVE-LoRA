"""
Pretrain a base VE/MoVE model from scratch on the ruwiki dataset.

Usage:
    base-train --config configs/base_d12_off.yaml [--set key=value ...]
    base-eval  [--model-tag d12] [--step 10000]

Config is a YAML file (see configs/*.yaml); individual keys can be overridden
on the command line: --set model.n_layer=4 --set max_steps=100
"""
import argparse
import gc
import json
import math
import os
import time

import torch

from jevelike.common import (
    COMPUTE_DTYPE, compute_init, compute_cleanup, get_base_dir,
    get_checkpoints_dir, get_data_dir, get_runs_dir, print0,
    setup_default_logging, JsonlLogger,
)
from jevelike.configs import load_config, save_config
from jevelike.gpt import GPT
from jevelike.move import resolve_move
from jevelike.tokenizer import get_tokenizer, get_token_bytes
from jevelike.dataloader import (
    tokenizing_distributed_data_loader_bos_bestfit,
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
)
from jevelike.loss_eval import evaluate_bpb
from jevelike.checkpoint import (
    save_checkpoint, load_checkpoint, load_model_from_dir,
    find_largest_model, find_last_step,
)

SAMPLE_PROMPTS = [
    "Столица Франции —",
    "Химический символ золота —",
    "Если вчера был понедельник, то завтра будет",
    "Планеты Солнечной системы:",
    "Если 5*x + 3 = 13, то x равно",
    "Самый большой океан —",
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=str, required=True, help="path to YAML config")
    ap.add_argument("--set", action="append", default=[],
                    help="override config key, e.g. --set model.n_layer=4 (repeatable)")
    ap.add_argument("--device-type", type=str, default="", help="cuda|cpu (empty = autodetect)")
    ap.add_argument("--data-dir", type=str, default=None, help="override data dir")
    ap.add_argument("--run-name", type=str, default=None, help="run name for logging dir")
    return ap.parse_args()


def apply_overrides(cfg, pairs):
    for pair in pairs:
        key, _, value = pair.partition("=")
        value = json.loads(value)  # ints/floats/bools/strings
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = getattr(node, part)
        old = getattr(node, parts[-1])
        if isinstance(old, bool) and not isinstance(value, bool):
            value = value == "true" or value is True
        setattr(node, parts[-1], value)


def build_model(cfg, tokenizer, device):
    cfg.model.vocab_size = tokenizer.get_vocab_size()
    cfg.model.validate()
    cfg.move.validate()
    move = resolve_move(cfg.move, cfg.model)
    with torch.device("meta"):
        model = GPT(cfg.model, move=move)
    model.to_empty(device=device)
    model.init_weights()
    return model, move


def main():
    args = parse_args()
    setup_default_logging()
    cfg = load_config(args.config)
    apply_overrides(cfg, args.set)

    device_type = args.device_type if args.device_type else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_type)
    compute_init(device_type)
    torch.manual_seed(cfg.seed)
    synchronize = torch.cuda.synchronize if device_type == "cuda" else (lambda: None)
    get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else (lambda: 0)
    data_dir = args.data_dir or get_data_dir()
    print0(f"Device: {device_type} | COMPUTE_DTYPE: {COMPUTE_DTYPE}")

    # run logging
    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(get_runs_dir(), cfg.model_tag, run_name)
    os.makedirs(run_dir, exist_ok=True)
    logger = JsonlLogger(os.path.join(run_dir, "log.jsonl"))
    save_config(cfg, os.path.join(run_dir, "config.yaml"))

    # tokenizer
    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device)
    vocab_size = tokenizer.get_vocab_size()
    print0(f"Vocab size: {vocab_size:,}")

    # model
    checkpoint_dir = os.path.join(get_checkpoints_dir(), cfg.model_tag)
    model, move = build_model(cfg, tokenizer, device)
    print0(f"Model config: {json.dumps(cfg.model.as_dict())}")
    print0(f"Move config: {json.dumps(cfg.move.as_dict())}")
    print0(f"Parameter counts:")
    for key, value in model.num_scaling_params().items():
        print0(f"  {key:24s}: {value:,}")
    num_flops_per_token = model.estimate_flops()
    print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

    # resume
    resuming = cfg.resume_from_step > 0
    meta_data = {}
    dataloader_state_dict = None
    if resuming:
        print0(f"Resuming from step {cfg.resume_from_step}")
        model_data, optimizer_data, meta_data = load_checkpoint(
            checkpoint_dir, cfg.resume_from_step, device, load_optimizer=True
        )
        model.load_state_dict(model_data, strict=True, assign=True)
        del model_data

    # compile (orig_model kept for sampling/state_dict, which dislike changing shapes)
    orig_model = model
    if device_type == "cuda":
        model = torch.compile(model, dynamic=False)

    # ------------------------------------------------------------------
    # Scaling: horizon, batch size, LR/WD scaling (nanochat-style, ref d12)
    # ------------------------------------------------------------------
    param_counts = orig_model.num_scaling_params()
    num_scaling_params = param_counts["transformer_matrices"] + param_counts["lm_head"]
    # reference d12 scaling params for the ratio math (D12 = ModelConfig defaults)
    from jevelike.configs import ModelConfig
    ref_cfg = ModelConfig()
    with torch.device("meta"):
        ref_model = GPT(ref_cfg, move=None)
    ref_counts = ref_model.num_scaling_params()
    D_REF = cfg.target_param_data_ratio * (ref_counts["transformer_matrices"] + ref_counts["lm_head"])
    B_REF = 2 ** 19
    target_tokens = int(cfg.target_param_data_ratio * num_scaling_params)

    total_batch_size = cfg.total_batch_size
    if total_batch_size == -1:
        predicted = B_REF * (target_tokens / D_REF) ** 0.383
        total_batch_size = 2 ** round(math.log2(predicted))
        print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

    batch_lr_scale = (total_batch_size / B_REF) ** 0.5
    if batch_lr_scale != 1.0:
        print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,}")
    weight_decay_scaled = cfg.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
    if weight_decay_scaled != cfg.weight_decay:
        print0(f"Scaling weight decay from {cfg.weight_decay:.6f} to {weight_decay_scaled:.6f}")

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    optimizer = model.setup_optimizer(
        unembedding_lr=cfg.unembedding_lr * batch_lr_scale,
        embedding_lr=cfg.embedding_lr * batch_lr_scale,
        scalar_lr=cfg.scalar_lr * batch_lr_scale,
        matrix_lr=cfg.matrix_lr * batch_lr_scale,
        weight_decay=weight_decay_scaled,
        move_gate_lr=cfg.move_gate_lr,
    )
    if resuming:
        optimizer.load_state_dict(optimizer_data)
        del optimizer_data

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    dataloader_resume_state_dict = None if not resuming else meta_data.get("dataloader_state_dict")
    train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer, cfg.device_batch_size, cfg.model.sequence_len,
        split="train", data_dir=data_dir, device=device,
        resume_state_dict=dataloader_resume_state_dict,
    )
    build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, cfg.device_batch_size, cfg.model.sequence_len,
        split="val", data_dir=data_dir, device=device,
    )
    x, y, dataloader_state_dict = next(train_loader)

    # ------------------------------------------------------------------
    # Horizon and schedulers
    # ------------------------------------------------------------------
    if cfg.max_steps > 0:
        num_iterations = cfg.max_steps
        print0(f"Using user-provided number of iterations: {num_iterations:,}")
    else:
        num_iterations = target_tokens // total_batch_size
        print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
    total_tokens = total_batch_size * num_iterations
    print0(f"Total number of training tokens: {total_tokens:,}")
    print0(f"Tokens : scaling params ratio: {total_tokens / num_scaling_params:.2f}")

    def get_lr_multiplier(it):
        warmup_iters = cfg.warmup_steps
        warmdown_iters = round(cfg.warmdown_ratio * num_iterations)
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        elif it <= num_iterations - warmdown_iters:
            return 1.0
        else:
            progress = (num_iterations - it) / warmdown_iters
            return progress * 1.0 + (1 - progress) * cfg.final_lr_frac

    def get_muon_momentum(it):
        warmdown_iters = round(cfg.warmdown_ratio * num_iterations)
        warmdown_start = num_iterations - warmdown_iters
        if it < 400:
            frac = it / 400
            return (1 - frac) * 0.85 + frac * 0.97
        elif it >= warmdown_start:
            progress = (it - warmdown_start) / warmdown_iters
            return 0.97 * (1 - progress) + 0.90 * progress
        else:
            return 0.97

    def get_weight_decay(it):
        return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / max(num_iterations, 1)))

    # ------------------------------------------------------------------
    # Loop state
    # ------------------------------------------------------------------
    if not resuming:
        step = 0
        val_bpb = None
        min_val_bpb = float("inf")
        smooth_train_loss = 0.0
        total_training_time = 0.0
    else:
        step = meta_data["step"]
        val_bpb = meta_data.get("val_bpb")
        loop_state = meta_data.get("loop_state", {})
        min_val_bpb = loop_state.get("min_val_bpb", float("inf"))
        smooth_train_loss = loop_state.get("smooth_train_loss", 0.0)
        total_training_time = loop_state.get("total_training_time", 0.0)

    tokens_per_fwdbwd = cfg.device_batch_size * cfg.model.sequence_len
    assert total_batch_size % tokens_per_fwdbwd == 0, \
        f"total_batch_size ({total_batch_size}) must be a multiple of {tokens_per_fwdbwd}"
    grad_accum_steps = total_batch_size // tokens_per_fwdbwd
    print0(f"Tokens / micro-batch: {tokens_per_fwdbwd:,}")
    print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

    scaler = torch.amp.GradScaler("cuda") if COMPUTE_DTYPE == torch.float16 and device_type == "cuda" else None

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    eval_tokens = 4 * 2 ** 20  # ~4.2M tokens of val evaluation
    while True:
        last_step = step == num_iterations

        if cfg.eval_every > 0 and (last_step or step % cfg.eval_every == 0):
            model.eval()
            val_loader = build_val_loader()
            eval_steps = max(1, eval_tokens // (cfg.device_batch_size * cfg.model.sequence_len))
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
            print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
            if val_bpb < min_val_bpb:
                min_val_bpb = val_bpb
            logger.log(step=step, val_bpb=val_bpb)
            model.train()

        if cfg.sample_every > 0 and (last_step or (step > 0 and step % cfg.sample_every == 0)):
            model.eval()
            for prompt in SAMPLE_PROMPTS:
                tokens = tokenizer.encode(prompt, prepend="<|bos|>")
                gen_tokens = []
                with torch.inference_mode():
                    for tid in orig_model.generate(tokens, max_tokens=16, temperature=0.0):
                        gen_tokens.append(tid)
                sample_text = tokenizer.decode(tokens + gen_tokens)
                print0(f"  {prompt!r} -> {sample_text!r}")
                logger.log(step=step, sample=prompt, completion=sample_text)
            model.train()

        if last_step or (step > 0 and step != cfg.resume_from_step and cfg.save_every > 0 and step % cfg.save_every == 0):
            save_checkpoint(
                checkpoint_dir,
                step,
                orig_model.state_dict(),
                optimizer.state_dict(),
                {
                    "step": step,
                    "val_bpb": val_bpb,
                    "model_config": cfg.model.as_dict(),
                    "move_config": cfg.move.as_dict(),
                    "user_config": cfg.as_dict(),
                    "device_batch_size": cfg.device_batch_size,
                    "total_batch_size": total_batch_size,
                    "dataloader_state_dict": dataloader_state_dict,
                    "loop_state": {
                        "min_val_bpb": min_val_bpb,
                        "smooth_train_loss": smooth_train_loss,
                        "total_training_time": total_training_time,
                    },
                },
            )

        if last_step:
            break

        synchronize()
        t0 = time.time()
        train_loss = None
        for micro_step in range(grad_accum_steps):
            loss = model(x, y)
            train_loss = loss.detach()
            loss = loss / grad_accum_steps
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            x, y, dataloader_state_dict = next(train_loader)
        lrm = get_lr_multiplier(step)
        muon_momentum = get_muon_momentum(step)
        muon_weight_decay = get_weight_decay(step)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_weight_decay
        if scaler is not None:
            scaler.unscale_(optimizer)
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        model.zero_grad(set_to_none=True)
        train_loss_f = train_loss.item()
        synchronize()
        dt = time.time() - t0

        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
        pct_done = 100 * step / num_iterations
        tok_per_sec = int(total_batch_size / dt)
        flops_per_sec = num_flops_per_token * total_batch_size / dt
        mfu = flops_per_sec / 1e12  # just for logging on CPU; not a real MFU
        if step > 10:
            total_training_time += dt
        epoch = f"{dataloader_state_dict['epoch']} pq:{dataloader_state_dict['pq_idx']} rg:{dataloader_state_dict['rg_idx']}"
        print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | "
               f"lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | epoch: {epoch}")
        if step % 100 == 0:
            logger.log(step=step, train_loss=debiased_smooth_loss, lrm=lrm, dt=dt,
                       tok_per_sec=tok_per_sec, flops_per_sec=flops_per_sec)

        first_step_of_run = (step == 0) or (resuming and step == cfg.resume_from_step)
        step += 1
        if first_step_of_run:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif step % 5000 == 0:
            gc.collect()

    print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
    print0(f"Total training time: {total_training_time / 60:.2f}m")
    if val_bpb is not None:
        print0(f"Minimum validation bpb: {min_val_bpb:.6f}")
    logger.log(done=True, min_val_bpb=min_val_bpb, total_training_time=total_training_time)
    logger.close()
    compute_cleanup(device_type)


def eval_main():
    ap = argparse.ArgumentParser(description="Evaluate val bpb of a base model checkpoint")
    ap.add_argument("--model-tag", type=str, default=None)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--device-type", type=str, default="")
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--eval-tokens", type=int, default=4 * 2 ** 20)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    setup_default_logging()

    device_type = args.device_type if args.device_type else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_type)
    compute_init(device_type)
    data_dir = args.data_dir or get_data_dir()

    checkpoints_dir = get_checkpoints_dir()
    if args.model_tag is None:
        args.model_tag = find_largest_model(checkpoints_dir)
    checkpoint_dir = os.path.join(checkpoints_dir, args.model_tag)
    if args.step is None:
        args.step = find_last_step(checkpoint_dir)
    print0(f"Evaluating {checkpoint_dir} step {args.step}")

    model, tokenizer, meta_data = load_model_from_dir(
        checkpoints_dir, device, "eval", args.model_tag, args.step
    )
    token_bytes = get_token_bytes(device=device)
    val_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, args.batch_size, model.config.sequence_len,
        split="val", data_dir=data_dir, device=device,
    )
    eval_steps = max(1, args.eval_tokens // (args.batch_size * model.config.sequence_len))
    bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
    print0(f"Step {args.step:06d} | Validation bpb: {bpb:.6f}")
    compute_cleanup(device_type)


if __name__ == "__main__":
    main()
