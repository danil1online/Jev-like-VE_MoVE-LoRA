"""
Generate Jev-Like NOUL (yes/no) contrastive data with an OpenAI-compatible LLM API.

LLM-as-a-Teacher (docs/gemini_talk.md, steps 1-2): for each source text the LLM
produces a `true_statement` and a `false_statement` that differs in exactly ONE
fact. Each text yields two training rows:

    {"text": T, "question": true_statement,  "answer": "да",  "task": "noul", "pair_id": ...}
    {"text": T, "question": false_statement, "answer": "нет", "task": "noul", "pair_id": ...}

With --lang en the prompts are English and the answer words are "yes"/"no".

Input:  JSONL with {"text": ...} rows, or a plain .txt file (one text per line).
Output: <output>/raw.jsonl (resumable, one row per source text), then a
        deterministic split into <output>/train.jsonl and <output>/val.jsonl;
        both rows of a pair always land in the same split.

Env: JEV_API_BASE (default http://localhost:8000/v1), JEV_API_KEY (default ""),
     JEV_MODEL (default gpt-4o-mini).
The script is resumable: source texts already present in raw.jsonl are skipped.
"""
import os
import json
import argparse
import hashlib
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

API_BASE = os.environ.get("JEV_API_BASE", "http://localhost:8000/v1")
API_KEY = os.environ.get("JEV_API_KEY", "")
MODEL = os.environ.get("JEV_MODEL", "gpt-4o-mini")

SYSTEM_PROMPTS = {
    "ru": (
        'Ты — эксперт по анализу данных. Твоя задача — создавать контрастивные пары '
        'утверждений для обучения классификатора "System One".'
    ),
    "en": (
        'You are an expert data analyst. Your task is to create contrastive pairs of '
        'statements for training a "System One" classifier.'
    ),
}

USER_PROMPT_TEMPLATES = {
    "ru": (
        'Входной текст:\n"""\n{text}\n"""\n\n'
        'Задание: Внимательно прочитай текст и сгенерируй строго в формате JSON:\n'
        '1. "true_statement": Одно короткое и емкое утверждение на русском языке, '
        'которое абсолютно истинно на основе текста.\n'
        '2. "false_statement": Точно такое же утверждение, но в котором изменено '
        'ровно ОДНО ключевое слово (сущность, число, действие), что делает его '
        'абсолютно ложным на основе текста.\n\n'
        'Формат ответа:\n'
        '{{\n  "true_statement": "строка",\n  "false_statement": "строка"\n}}'
    ),
    "en": (
        'Input text:\n"""\n{text}\n"""\n\n'
        'Task: Read the text carefully and produce strictly valid JSON:\n'
        '1. "true_statement": One short, precise statement in English that is '
        'absolutely true based on the text.\n'
        '2. "false_statement": The exact same statement, but with exactly ONE key '
        'word changed (entity, number, action), making it absolutely false based on '
        'the text.\n\n'
        'Answer format:\n'
        '{{\n  "true_statement": "string",\n  "false_statement": "string"\n}}'
    ),
}

# answer words per language (must match jev.yes_token/no_token of the training config)
LANG_ANSWERS = {"ru": ("да", "нет"), "en": ("yes", "no")}


def build_pair_prompt(text, lang="ru"):
    return SYSTEM_PROMPTS[lang], USER_PROMPT_TEMPLATES[lang].format(text=text)


def parse_pair_response(content):
    """Parse the LLM answer into {"true_statement", "false_statement"} or None."""
    if not content:
        return None
    content = content.strip()
    if content.startswith("```"):  # strip a markdown code fence
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:]
        content = content.strip()
    try:
        data = json.loads(content)
    except Exception:
        i, j = content.find("{"), content.rfind("}")  # tolerate prose around the JSON
        if i == -1 or j <= i:
            return None
        try:
            data = json.loads(content[i:j + 1])
        except Exception:
            return None
    if not isinstance(data, dict):
        return None
    true_s = data.get("true_statement")
    false_s = data.get("false_statement")
    if not isinstance(true_s, str) or not isinstance(false_s, str):
        return None
    true_s, false_s = true_s.strip(), false_s.strip()
    if not true_s or not false_s or true_s == false_s:
        return None
    return {"true_statement": true_s, "false_statement": false_s}


def pair_rows(text, pair, pair_id, lang="ru"):
    """The two training rows for one contrastive pair (both share pair_id)."""
    yes, no = LANG_ANSWERS[lang]
    return [
        {"text": text, "question": pair["true_statement"],
         "answer": yes, "task": "noul", "pair_id": pair_id},
        {"text": text, "question": pair["false_statement"],
         "answer": no, "task": "noul", "pair_id": pair_id},
    ]


def split_rows(rows, val_frac, seed=42):
    """Deterministic train/val split that keeps both rows of a pair together."""
    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["pair_id"], []).append(r)
    pairs = list(by_pair)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_val = max(0, min(len(pairs), round(len(pairs) * val_frac)))
    val_pairs, train_pairs = set(pairs[:n_val]), set(pairs[n_val:])
    train, val = [], []
    for pid in pairs:
        (val if pid in val_pairs else train).extend(by_pair[pid])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def ask_llm_pair(text, max_retries=3, timeout=120, json_mode=True, lang="ru"):
    """Ask the LLM for a contrastive pair. Returns the pair dict or None."""
    system, user = build_pair_prompt(text, lang)
    url = f"{API_BASE.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0.0,
        "max_tokens": 300,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    for attempt in range(max_retries):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if r.status_code == 400 and json_mode and attempt == 0:
                # server does not support response_format: retry in plain mode
                return ask_llm_pair(text, max_retries, timeout, json_mode=False, lang=lang)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            pair = parse_pair_response(content)
            if pair is not None:
                return pair
        except Exception:
            pass
    return None


def load_input_texts(path):
    """JSONL with {"text": ...} rows, or a plain text file (one text per line)."""
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if path.endswith((".jsonl", ".json")):
                row = json.loads(line)
                text = row["text"]
            else:
                text = line
            if text:
                texts.append(text)
    return texts


def make_pair_id(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True,
                    help='JSONL with {"text": ...} rows, or a .txt file (one text per line)')
    ap.add_argument("--output", required=True, help="output directory")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-only", action="store_true",
                    help="skip the API; rebuild train/val from an existing raw.jsonl")
    ap.add_argument("--lang", choices=("ru", "en"), default="ru",
                    help='language of prompts and answer words ("да"/"нет" or "yes"/"no")')
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    raw_path = os.path.join(args.output, "raw.jsonl")

    if args.split_only:
        texts, done = [], set()
    else:
        texts = load_input_texts(args.input)
        done = set()
        if os.path.exists(raw_path):
            with open(raw_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        done.add(json.loads(line)["text"])
        if done:
            print(f"Resuming: {len(done)} texts already done")
    print(f"Loaded {len(texts)} source texts")

    if not args.split_only:
        todo = [t for t in texts if t not in done]
        print(f"To do: {len(todo)}")
        lock = threading.Lock()
        n_ok, n_fail = 0, 0
        with open(raw_path, "a", encoding="utf-8") as out_f:
            def work(text):
                pair = ask_llm_pair(text, lang=args.lang)
                if pair is None:
                    return None
                row = {"text": text, "pair": pair, "pair_id": make_pair_id(text),
                       "lang": args.lang}
                with lock:
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out_f.flush()
                return row

            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = {ex.submit(work, t): t for t in todo}
                for fut in as_completed(futures):
                    if fut.result() is None:
                        n_fail += 1
                    else:
                        n_ok += 1
                    total = n_ok + n_fail
                    if total % 100 == 0:
                        print(f"  {total}/{len(todo)} (ok={n_ok}, fail={n_fail})")
        print(f"Done: {n_ok} ok, {n_fail} failed")

    # split raw.jsonl -> train.jsonl / val.jsonl (pairs stay together)
    raw = []
    if os.path.exists(raw_path):
        with open(raw_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw.append(json.loads(line))
    rows = []
    for r in raw:
        rows.extend(pair_rows(r["text"], r["pair"], r["pair_id"], r.get("lang", "ru")))
    train, val = split_rows(rows, args.val_frac, seed=args.seed)
    for name, part in (("train", train), ("val", val)):
        path = os.path.join(args.output, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {path}: {len(part)} rows")


if __name__ == "__main__":
    main()
