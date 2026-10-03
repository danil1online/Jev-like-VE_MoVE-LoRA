"""
Ask a trained Jev adapter a question (System One style): one forward pass over
(text, question) -> answer word + probability over the task candidates.

Usage:
    jev-ask --config configs/jev_lora_d12.yaml --task noul \
        --text "Иван Иванов оформил возврат товара." \
        --question "Иван Иванов оформил возврат товара?"

Multiple questions share one context and one forward pass (repeat --question).
If --text is omitted the context is read from stdin.
If <jev_checkpoints>/<model_tag>/calibration.json exists (written by jev-train),
its per-task temperature is applied to the probabilities; --temperature overrides it.
"""
import argparse
import json
import os
import sys

import torch

from jevelike.common import (
    compute_init, compute_cleanup, get_base_dir, print0, setup_default_logging,
)
from jevelike.configs import load_config
from jevelike.checkpoint import load_adapter, find_last_adapter
from jevelike.jev import JevAdapter, calibrate_probs
from jevelike.tokenizer import get_tokenizer
from jevelike.train.jev_lora import (
    apply_overrides, check_adapter_meta, load_base_from_path, task_slices,
)


def ask(adapter, text, questions, task, device, temperature=1.0):
    """One forward pass over all questions sharing the same context.
    Returns a list of (answer_word, answer_prob, prob_vector_over_task_candidates)."""
    renderer = adapter.renderer
    if task not in renderer.words_by_task:
        raise ValueError(f"Unknown task {task!r}; valid: {list(renderer.words_by_task)}")
    placeholder = renderer.words_by_task[task][0]
    items = [{"text": text, "question": q, "answer": placeholder, "task": task} for q in questions]
    x, labels, meta = renderer.make_batch(items)
    x = x.to(device)
    s, e = task_slices(renderer)[task]
    with torch.inference_mode():
        probs = adapter.probs(x)
    words = renderer.words_by_task[task]
    out = []
    for i in range(len(questions)):
        # restricted softmax over this task's candidates
        p = probs[i, meta[i]["pos"], s:e]
        p = calibrate_probs(p / p.sum(-1, keepdim=True), temperature)
        idx = p.argmax().item()
        out.append((words[idx], p[idx].item(), p))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--device-type", type=str, default="")
    ap.add_argument("--adapter", type=str, default=None,
                    help="adapter checkpoint (default: highest step in jev_checkpoints/<model_tag>)")
    ap.add_argument("--task", type=str, default="noul", choices=("noul", "choice", "score"))
    ap.add_argument("--text", type=str, default=None, help="context (stdin if omitted)")
    ap.add_argument("--question", action="append", required=True,
                    help="question to answer; repeatable (one forward pass for all)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="post-hoc temperature override (default: calibration.json, else 1.0)")
    args = ap.parse_args()

    setup_default_logging()
    cfg = load_config(args.config)
    apply_overrides(cfg, args.set)

    device_type = args.device_type if args.device_type else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_type)
    compute_init(device_type)

    tokenizer = get_tokenizer()
    model, _, meta_data = load_base_from_path(cfg.jev_base_checkpoint, device)
    adapter = JevAdapter(model, cfg.lora, cfg.jev, tokenizer)
    adapter.eval()

    adapter_dir = os.path.join(get_base_dir(), "jev_checkpoints", cfg.model_tag)
    if args.adapter is not None:
        adapter_path = args.adapter
    else:
        step = find_last_adapter(adapter_dir)
        adapter_path = os.path.join(adapter_dir, f"adapter_{step:06d}.pt")
    adapter_state, step, a_meta = load_adapter(adapter_path, device)
    adapter.load_adapter_state(adapter_state)
    check_adapter_meta(a_meta, cfg, adapter_path)
    print0(f"Adapter step: {step}")

    temperature = 1.0
    if args.temperature is not None:
        temperature = args.temperature
    else:
        calib_path = os.path.join(adapter_dir, "calibration.json")
        if os.path.exists(calib_path):
            with open(calib_path, "r", encoding="utf-8") as f:
                calib = json.load(f)
            if args.task in calib:
                temperature = calib[args.task]
                print0(f"Calibrated temperature ({args.task}): {temperature}")

    text = args.text if args.text is not None else sys.stdin.read()
    results = ask(adapter, text, args.question, args.task, device, temperature)
    for q, (word, p, _probs) in zip(args.question, results):
        print0(f"Q: {q}\nA: {word}  (p={p:.4f})")
    compute_cleanup(device_type)


if __name__ == "__main__":
    main()
