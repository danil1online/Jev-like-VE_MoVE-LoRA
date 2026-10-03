# Jev-like-VE / MoVE-LoRA

Training a value-embedding (VE) / MoVE base model from scratch on
`atBuba/ruwiki-dataset` (nanochat-style pipeline), plus a **Jev-Like-LoRA
adapter** on top of a frozen base for typed decisions:

- **NOUL** — yes/no (`да` / `нет`) over a context + question
- **choice** — single-letter answers (28 Cyrillic letters)
- **score** — 0..9

The adapter is a low-rank LoRA (q/k/v by default) plus learnable rows for the
`<|ctx|>`, `<|q|>`, `<|a|>` special tokens and a candidate-restricted head
(`extra_in` / `extra_out`), so inference is a single forward pass with
`softmax` over the 40 answer candidates.

Design spec: `docs/gemini_talk.md`. MoVE reference: arXiv:2601.22887.

## Layout

```
jevelike/
  common.py         device/dtype init, dirs, logging
  configs.py        ModelConfig / MoveConfig / LoraConfig / JevConfig / TrainConfig (YAML)
  move.py           MoVE value bank + gate, LaVE per-layer values, resolve_move
  gpt.py            GPT (nanochat lineage): BTHD, rotary, sliding windows, softcap
  flash_attention.py
  tokenizer.py      RustBPETokenizer (rustbpe + tiktoken), 4 special tokens
  dataset.py        parquet shard reading, row-group buffering
  dataloader.py     distributed best-fit dataloader with BOS alignment
  optim.py          Muon + AdamW parameter groups
  loss_eval.py      bpb / ppl evaluation
  checkpoint.py     base checkpoints + adapter save/load
  lora.py           LoRALinear, apply_lora, freeze_base
  jev.py            JevAdapter, JevRenderer, answer_probs, calibration metrics
  data/
    prep.py         ruwiki -> parquet shards (train/val)
    contrastive.py  NOUL contrastive generation via OpenAI-compatible API
  train/
    base.py         base-train / base-eval
    jev_lora.py     jev-train / jev-eval
    jev_ask.py      jev-ask (System One inference: text + question -> answer + p)
    tokenizer.py    tok-train
configs/            base_d12_off / base_d12_move / base_d12_lave / base_d20_move / jev_lora_d12
tests/              pytest suite (CPU, tiny models)
```

## Setup

Python >= 3.10, [uv](https://docs.astral.sh/uv/).

```bash
# CPU
uv sync --extra cpu --group dev
# GPU (CUDA 12.8)
uv sync --extra gpu --group dev

# CPU torch (if a fresh venv): uv picks the index via [tool.uv.sources]
uv pip install --python .venv/bin/python --reinstall-package torch \
    --index-url https://download.pytorch.org/whl/cpu "torch==2.14.1+cpu"
```

> **Torch preload note.** If the machine's `LD_LIBRARY_PATH` contains a stale
> system libtorch (e.g. `/opt/libtorch/lib`), torch may load the wrong shared
> libraries. This repo ships a site-level preload shim:
> `.venv/lib/python3.10/site-packages/jevelike_torch_preload.pth` (+ `.py`)
> which `ctypes.CDLL`s the venv's `libgomp -> libc10 -> libtorch_cpu -> libtorch
> -> libshm -> libtorch_python` (RTLD_GLOBAL) before torch does. Do not delete
> these files on this machine; reinstalling torch via uv keeps them working.

The tools are exposed as console scripts (`base-train`, `base-eval`,
`jev-train`, `jev-eval`, `jev-ask`, `tok-train`, `prepare-ruwiki`,
`make-contrastive`) via `[project.scripts]`. They require the project to be installed once:

```bash
uv pip install -e .        # from the repo root, inside the venv
```

The package can also be used as a plain import from the project root without
installing (`python -m jevelike.train.base ...`), but the `*-eval` entry points
are only reachable through the installed console scripts, so installing is the
simplest path.

## Pipeline

1. **Prepare the corpus** (downloads `atBuba/ruwiki-dataset`, writes parquet
   shards `<data>/ruwiki_train_0000.parquet`, ... + `ruwiki_val_*.parquet`;
   the last shard is the val split):

   ```bash
   prepare-ruwiki --num-shards 100 --seed 42
   # --input ruwiki_full.txt to use a local dump (skips download)
   # --no-download: fail instead of downloading when no --input given
   ```

2. **Train the tokenizer** (rustbpe BPE, 4 special tokens
   `<|bos|> <|ctx|> <|q|> <|a|>`), saved to `<base>/tokenizer`:

   ```bash
   tok-train --vocab-size 65536 --max-chars 2000000000
   ```

3. **Pretrain the base model**:

   ```bash
   base-train --config configs/base_d12_off.yaml    # vanilla transformer
   base-train --config configs/base_d12_move.yaml   # MoVE value embeddings
   base-train --config configs/base_d12_lave.yaml   # LaVE (per-layer values)
   base-train --config configs/base_d20_move.yaml   # bigger D20
   # --set key=value ... for overrides, --device-type cuda|cpu,
   # --data-dir DIR, --run-name NAME
   ```

   Checkpoints land in `checkpoints/<model_tag>/model_XXXXXX.pt` (+ meta,
   optimizer). `base-eval --model-tag d12_move --step 100000` reports val bpb.

4. **Build the Jev dataset** (LLM-as-Teacher contrastive pairs, resumable;
   needs an OpenAI-compatible API, env `JEV_API_BASE` / `JEV_API_KEY` /
   `JEV_MODEL`). Input: JSONL `{"text": ...}` rows or a plain `.txt` file
   (one text per line). For each text the LLM produces a `true_statement` and
   a `false_statement` that differs in exactly one fact, yielding two rows
   (answer `да` / `нет`) that share a `pair_id`. Output: `<output>/raw.jsonl`
   (resumable) plus a deterministic split into `train.jsonl` / `val.jsonl`
   where both rows of a pair always land in the same split:

   ```bash
   make-contrastive --input texts.jsonl --output data/jev --val-frac 0.1
   # re-split from an existing raw.jsonl without API calls:
   make-contrastive --input texts.jsonl --output data/jev --split-only
   ```

5. **Train the Jev-Like-LoRA adapter** (base frozen):

   ```bash
   jev-train --config configs/jev_lora_d12.yaml
   ```

   Adapters land in `jev_checkpoints/<model_tag>/adapter_XXXXXX.pt`. After
   training, a per-task post-hoc temperature is fit on the val split
   (minimizing log-loss) and written to
   `jev_checkpoints/<model_tag>/calibration.json`.

6. **Evaluate the adapter** (per-task accuracy / ECE / brier / logloss, with
   before/after-calibration comparison when `calibration.json` exists):

   ```bash
   jev-eval --config configs/jev_lora_d12.yaml [--adapter PATH]
   ```

7. **Ask the adapter** (System One inference: one forward pass, answer word +
   probability over the task candidates; `--question` is repeatable and all
   questions share one pass; the calibrated temperature is applied
   automatically):

   ```bash
   jev-ask --config configs/jev_lora_d12.yaml --task noul \
       --text "Иван Иванов оформил возврат товара." \
       --question "Иван Иванов оформил возврат товара?"
   # context from stdin; override the temperature with --temperature
   ```

## Config reference

`configs/*.yaml` (YAML 1.1 — quote `"off"`, it would otherwise parse as bool).

| Key | Meaning |
|---|---|
| `model.n_layer/n_head/n_kv_head/n_embd` | D12 = 12/6/6/768, D20 = 20/10/10/1280 |
| `model.window_pattern` | `S` = local (half window), `L` = full; pattern tiled, last layer always full |
| `model.softcap` | logit softcap (15.0) |
| `move.mode` | `"off"`, `"move"`, `"lave"` |
| `move.num_slots` | MoVE slots per layer (auto = `n_layer // 2`) |
| `move.gated_standard` | gate the standard V alongside MoVE slots (MoVE only) |
| `move.lave_layers` | `"alt"` (every other layer) or `"all"` (LaVE) |
| `lora.rank/alpha/target_modules` | `q,k,v,proj,fc,mlp_proj` |
| `jev.letter_tokens` | 28 letters `АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЭЮЯ` (no Й/Ё/Ъ/Ы/Ь) |
| `total_batch_size: -1` | auto = `target_param_data_ratio` x params (12x) |

Answer candidate layout (40 total): `[0:2]` да/нет, `[2:30]` letters,
`[30:40]` digits.

## Tests

CPU-only, tiny models, no data downloads:

```bash
.venv/bin/python -m pytest tests/ -q        # 67 tests
.venv/bin/python -m pytest tests/ -m "not slow"
```

`tests/conftest.py` provides a deterministic `FakeTokenizer` (Jev candidate
ids are exact) and a real small rustbpe tokenizer for roundtrip tests.

## Design notes

- **MoVE**: one shared bank `E: vocab -> M x kv_dim`; per layer one lookup and
  `V = g0 ⊙ V + Σ_m g_m ⊙ M_m` with sigmoid gates (scale 2.0, init ~0.5).
- **LaVE**: per-layer single extra value, ungated `V = V + g ⊙ M_1` on selected
  layers (alt/all), cheaper than MoVE.
- **JevAdapter**: base forward is untouched except (a) LoRA deltas on q/k/v,
  (b) `extra_in` rows swapped in for `<|ctx|>/<|q|>/<|a|>`, (c) `extra_out`
  head producing logits for the 40 candidates. At init the adapter is
  (numerically) identity: `extra_out` copies `lm_head` rows and LoRA `B = 0`.
- Attention is BTHD; logits are softcapped; `lm_head` init std 0.001 (not tied
  to the token embedding).
- **Per-task probabilities**: `adapter.probs` is a softmax over all 40
  candidates; per-task probabilities (metrics, `jev-ask`) are the restricted
  softmax over that task's slice, followed by the post-hoc temperature from
  `calibration.json` (`calibrate_probs` / `fit_temperature` in `jev.py`).
- rustbpe merge order is not frequency-controllable on small corpora, so tests
  never rely on a specific word being a single token.
