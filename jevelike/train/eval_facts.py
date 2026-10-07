"""
Fact-memory probe for base checkpoints (H1 in docs/experiments.md): fact-bpb and
entity-cloze top-1 accuracy on a small held-out set, complementary to val-bpb.

Build the eval set from the held-out val shards (never from train data):
    eval-facts --build --data-dir /home/jev/data --data facts_eval.jsonl [--limit 2000]

Score checkpoints:
    eval-facts --model-tag d12_move_x1_en --steps 1000,4767 --data facts_eval.jsonl

Eval-set JSONL rows are either {"text": ...} (fact-bpb) or {"prefix": ..., "answer": ...}
(cloze; answers that are not a single token in the model tokenizer are skipped).
"""
import os
import re
import json
import math
import argparse

import torch
import torch.nn.functional as F

from jevelike.common import compute_init, compute_cleanup, get_checkpoints_dir, print0, setup_default_logging
from jevelike.checkpoint import load_model_from_dir
from jevelike.dataset import parquets_iter_batched
from jevelike.tokenizer import get_tokenizer, get_token_bytes

# "Newton was English."  ->  prefix "Newton was", answer "English"
CLOZE_RE = re.compile(
    r"([A-Z][A-Za-z0-9 ,\-\.]{2,60}?)\s(is|was)\s([A-Z][A-Za-z0-9 \-\.]{2,40}?)(?=[\.,;!\n]|$)"
)
# "Paris is the capital of France."  ->  prefix "The capital of France is", answer "Paris"
CLOZE_INVERT_RE = re.compile(
    r"([A-Z][A-Za-z0-9 \-\.]{1,40}?)\s(?:is|was)\sthe\s([a-z][A-Za-z0-9 \-]{1,40}?)\sof\s"
    r"([A-Z][A-Za-z0-9 \-\.]{1,30}?)(?=[\.,;!\n]|$)"
)


def extract_clozes(texts, limit=2000, tokenizer=None):
    """Heuristic entity-fact cloze probes extracted from plain text.
    With a tokenizer given, only answers that are a single token are kept."""
    items, seen = [], set()

    def add(prefix, answer):
        prefix, answer = prefix.strip(), answer.strip().rstrip(".").strip()
        if len(answer) < 3:
            return
        if tokenizer is not None:
            try:
                tokenizer.encode_single_token(answer)
            except Exception:
                return
        key = (prefix, answer)
        if key in seen:
            return
        seen.add(key)
        items.append({"prefix": prefix, "answer": answer})

    for text in texts:
        for m in CLOZE_RE.finditer(text):
            subj, verb, obj = m.group(1).strip(), m.group(2), m.group(3)
            if subj.lower() != obj.lower():
                add(f"{subj} {verb}", obj)
                if len(items) >= limit:
                    return items
        for m in CLOZE_INVERT_RE.finditer(text):
            ent, noun, of_ent = m.group(1), m.group(2), m.group(3)
            add(f"The {noun} of {of_ent} is", ent)
            if len(items) >= limit:
                return items
    return items


def load_items(path):
    docs, clozes = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "text" in row:
                docs.append(row["text"])
            elif "prefix" in row and "answer" in row:
                clozes.append({"prefix": row["prefix"], "answer": row["answer"]})
    return docs, clozes


@torch.no_grad()
def score_model(model, tokenizer, docs, clozes, device, token_bytes=None, max_doc_chars=2048):
    """Returns fact_bpb (bpb over the fact docs) and cloze top-1 accuracy."""
    model.eval()
    if token_bytes is None:
        token_bytes = get_token_bytes().to(device)
    bos = tokenizer.get_bos_token_id()

    nats, nbytes = 0.0, 0
    for doc in docs:
        ids = tokenizer.encode(doc[:max_doc_chars], prepend=bos)
        if len(ids) < 2:
            continue
        x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
        y = torch.tensor(ids[1:], dtype=torch.long, device=device)
        logits = model(x)[0]
        loss = F.cross_entropy(logits.float(), y, reduction="none")
        b = token_bytes[y].to(torch.float32)
        nats += (loss * (b > 0)).sum().item()
        nbytes += int((b > 0).sum().item())
    fact_bpb = nats / (math.log(2) * nbytes) if nbytes else float("inf")

    correct, evaluated, skipped = 0, 0, 0
    for item in clozes:
        try:
            answer_id = tokenizer.encode_single_token(item["answer"])
        except Exception:
            skipped += 1
            continue
        ids = tokenizer.encode(item["prefix"], prepend=bos)
        if not ids:
            skipped += 1
            continue
        x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(x)[0, -1]
        evaluated += 1
        if int(logits.argmax().item()) == answer_id:
            correct += 1
    return {
        "fact_bpb": fact_bpb,
        "cloze_acc": (correct / evaluated) if evaluated else None,
        "n_docs": len(docs), "n_clozes": evaluated, "n_clozes_skipped": skipped,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true", help="build the eval set from val shards")
    ap.add_argument("--data-dir", default=None, help="dataset dir (for --build; default <base>/data)")
    ap.add_argument("--data", default="facts_eval.jsonl", help="eval set path (built or read)")
    ap.add_argument("--limit", type=int, default=2000, help="max cloze items to build")
    ap.add_argument("--doc-limit", type=int, default=500, help="max fact docs to build")
    ap.add_argument("--model-tag", default=None)
    ap.add_argument("--steps", default="", help="comma-separated steps (default: latest)")
    ap.add_argument("--device-type", default="")
    args = ap.parse_args()

    if args.build:
        from jevelike.common import get_data_dir
        data_dir = args.data_dir or get_data_dir()
        texts = []
        for batch in parquets_iter_batched(data_dir, split="val"):
            texts.extend(batch)
        clozes = extract_clozes(texts, limit=args.limit, tokenizer=get_tokenizer())
        docs = [t[:4000] for t in texts[:args.doc_limit]]
        with open(args.data, "w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps({"text": d}, ensure_ascii=False) + "\n")
            for c in clozes:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"Wrote {args.data}: {len(docs)} fact docs + {len(clozes)} cloze items")
        return

    if not args.model_tag:
        raise SystemExit("either --build or --model-tag is required")
    setup_default_logging()
    device_type = args.device_type or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_type)
    compute_init(device_type)
    docs, clozes = load_items(args.data)
    print0(f"Eval set: {len(docs)} fact docs, {len(clozes)} cloze items")

    steps = [int(s) for s in args.steps.split(",") if s.strip()]
    ckpt_dir = get_checkpoints_dir()
    results = []
    for step in steps or [None]:
        model, tokenizer, meta = load_model_from_dir(ckpt_dir, device, "eval",
                                                     model_tag=args.model_tag, step=step)
        m = score_model(model, tokenizer, docs, clozes, device)
        m["model_tag"] = args.model_tag
        m["step"] = meta.get("step", step)
        results.append(m)
        acc = "n/a" if m["cloze_acc"] is None else f"{m['cloze_acc']:.4f}"
        print0(f"step {m['step']} | fact_bpb: {m['fact_bpb']:.4f} | cloze_acc: {acc} "
               f"(evaluated {m['n_clozes']}, skipped {m['n_clozes_skipped']})")
        del model
    compute_cleanup(device_type)


if __name__ == "__main__":
    main()
