# План исследования: VE/MoVE-базы + Jev-like адаптер

Целевой ПК: **RTX 3090 24GB**, Linux, CUDA 12.8, Python ≥ 3.10, [uv](https://docs.astral.sh/uv/).
Все команды — из корня репозитория `~/jev` (клон этого репо), venv `.venv`.

## 1. Гипотезы

- **H1** (VE/MoVE как память фактов): при равном токеном бюджете модель с банком value-эмбеддингов
  даёт более низкий val-bpb и заметно более низкий **fact-bpb / выше cloze-accuracy**, чем контроль без VE.
  Ориентир из MoVE-статьи (arXiv:2601.22887, D12, FineWeb-Edu): standard 0.838 → LaVE 0.822 → MoVE×1 0.819 → ×4 0.806 bpb
  (полный прогон ~15.7B токенов; у нас 2.5B — дельты будут меньше, важён порядок и matched-budget).
- **H2** (размещение): VE в глубоких слоях (для d12 — слои 6–11) лучше чередования `alt`
  (nanochat discussion #463).
- **H3** (фактная база → классификатор): Jev-like LoRA поверх MoVE-базы показывает выше accuracy
  и лучше калибровку (ECE/Brier), чем тот же рецепт поверх базы без VE — при неизменном рецепте адаптера.

## 2. Матрица экспериментов (фаза 1)

d12: `n_layer=12, n_embd=768, n_head=n_kv_head=6, T=2048, vocab≈65536`.
Backbone ≈ **186M** (wte 50.3 + lm_head 50.3 + матрицы 84.9). Банк в scaling-параметры НЕ входит
(`num_scaling_params` = матрицы + lm_head, как в nanochat) — авто-ratio ломает matched-budget только
через `total_batch_size=-1`, поэтому во всех рукавах он задан явно.

| Рукав | Конфиг | Банк | Итого params | Токены | Шаги (batch 524 288) | device_bs |
|---|---|---|---|---|---|---|
| A контроль | `base_d12_off_en` | — | ~186M | 2.5B | 4768 | 16 |
| B LaVE alt | `base_d12_lave_en` | 6 слоёв × 50.3M = +302M | ~488M | 2.5B | 4768 | 16 |
| C MoVE ×1 | `base_d12_move_x1_en` | M=6: +302M | ~488M | 2.5B | 4768 | 16 |
| D MoVE ×4 | `base_d12_move_x4_en` | M=24: +1208M | ~1.39B | 2.5B | 4768 | 8 |
| E deep (опц.) | `base_d12_lave_deep_en` | слои 6–11: +302M | ~488M | 2.5B | 4768 | 16 |

Оценка времени: d12 на 3090 ≈ 30–45k tok/s → **~18–24 ч на рукав**; сетка A→C→B→D ≈ 3–4 суток.
VRAM D (проверить smoke-прогоном): веса bf16 ~2.8G + AdamW-state банка fp32 ~9.7G + backbone ~1.5G
+ активации → ожидаемо 15–18 GB, влезает; при OOM — `--set device_batch_size=4`.

Порядок запуска: **A и C первыми** (главное сравнение H1), затем B, D, E.

## 3. Подготовка окружения на целевом ПК

```bash
git clone <url-репозитория> ~/jev && cd ~/jev
uv sync --extra gpu --group dev
uv pip install --python .venv/bin/python -e .
# проверка GPU (ожидается 2.x+cu128 и True):
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# если в LD_LIBRARY_PATH есть системный libtorch — см. примечание о preload-shim в README
```

Данные держим вне репо: `export JEVELIKE_DATA_DIR=/home/jev/data` (все команды ниже предполагают эту переменную;
для `prepare-*` можно и явный `--data-dir`). Требуется ~60 ГБ свободно.

## 4. Фаза 0 — данные и токенизатор

Все команды ниже реализованы и протестированы (см. §8). Для скачивания gated-датасетов
экспортируйте `HF_TOKEN`; `prepare-hf` поддерживает resume (`--skip-download` — перешардировать
уже скачанный кэш без сети).

**4.1 Английский корпус** (FineWeb-Edu, ~4B токенов скачиваем: 2.5B обучение + val + запас):

```bash
.venv/bin/prepare-hf \
    --dataset HuggingFaceFW/fineweb-edu --subsample sample-10BT \
    --max-tokens 4000000000 --num-shards 200 --seed 42
# пишет <data>/train_*.parquet + val-шарды (схема {"text": ...}); val = последние шарды, НЕ перезаписывать
```

**4.2 Мелкий корпус для двуязычного токенизатора** (ruwiki — только для покрытия кириллицы,
база на нём не тренируется; скрипт уже существует):

```bash
.venv/bin/prepare-ruwiki --data-dir /home/jev/data_tok_ru --num-shards 2
```

**4.3 Токенизатор** (BPE 65536, англ. + ~10% русской доли; обязательна single-token проверка
кандидатов Jev: `yes/no`, A–Z, 0–9 и на будущее `да/нет`, А–Я):

```bash
.venv/bin/tok-train --vocab-size 65536 --max-chars 8000000000 \
    --lang both --extra-data-dir /home/jev/data_tok_ru
# итог: <repo>/tokenizer (+ token_bytes.bin)
```

Если кандидат оказывается multi-token — `JevAdapter` упадёт на старте с ValueError; это ловится здесь,
на этапе sanity-чека, а не через сутки тренировки.

## 5. Фаза 1 — обучение баз

Все рукава: одинаковые seed=42, данные, batch, шаги. Логи: `runs/<model_tag>/<timestamp>/log.jsonl`
(`train_loss`, `val_bpb`, шаг/время), конфиг прогона копируется туда же.

**5.0 Smoke-прогон каждого рукава перед длинным запуском** (~10 мин, проверка VRAM/скорости):

```bash
.venv/bin/base-train --config configs/base_d12_move_x4_en.yaml \
    --set max_steps=20 save_every=0 eval_every=10 device_batch_size=8
nvidia-smi   # во время smoke: оценить VRAM; при OOM уменьшать device_batch_size (grad accum подстроится)
```

**5.1 Полные прогоны** (последовательно, в tmux; пример для рукава C):

```bash
tmux new -s train_C -d \
  ".venv/bin/base-train --config configs/base_d12_move_x1_en.yaml 2>&1 | tee /home/jev/logs/C.log"
# контроль прогресса:
tail -f "$(ls -td runs/d12_move_x1_en/* | head -1)/log.jsonl"
```

Аналогично A (`base_d12_off_en`), B (`base_d12_lave_en`), D (`base_d12_move_x4_en`, `device_batch_size=8`),
E (`base_d12_lave_deep_en`).

**Резюм после обрыва:**

```bash
.venv/bin/base-train --config configs/base_d12_move_x1_en.yaml --set resume_from_step=<N>
```

**Мониторинг VE-гейтов (урок gefen-эксперимента: насыщение гейтов / мёртвые строки банка = плато):**
в `log.jsonl` каждые `eval_every` пишутся статистики норм гейтов и весов банков; плато loss при живом LR —
первым делом смотреть туда.

**5.2 Оценка баз** (по matched-token точкам, не только финал):

```bash
# однократно собрать факт-набор из val-шардов (никогда из train — урок hygiene из gefen):
.venv/bin/eval-facts --build --data-dir /home/jev/data --data /home/jev/data/facts_eval.jsonl
# обычный val-bpb:
.venv/bin/base-eval --model-tag d12_move_x1_en --step 4767
# факт-память (fact-bpb + cloze top-1):
.venv/bin/eval-facts --model-tag d12_off_en     --steps 1000,2000,3000,4767 --data /home/jev/data/facts_eval.jsonl
.venv/bin/eval-facts --model-tag d12_move_x1_en --steps 1000,2000,3000,4767 --data /home/jev/data/facts_eval.jsonl
```

Cloze-ответы, не являющиеся single-token у текущего токенизатора, автоматически пропускаются.

## 6. Фаза 2 — Jev-like адаптер (H3)

**6.1 Учитель** (Qwen через llama.cpp, OpenAI-совместимый API; машина-хост учится не мешать тренировке
или использовать паузы между рукавами):

```bash
# на хосте учителя:
llama-server -m /models/Qwen3-14B-Q6_K.gguf --port 8000 --host 0.0.0.0
```

**6.2 Исходные тексты для пар** — сэмпл из held-out val (5k документов, не пересекаются с обучением базы):

```bash
.venv/bin/python - <<'PY'
import glob, json, random, pyarrow.parquet as pq
docs = []
for p in sorted(glob.glob("/home/jev/data/*val*.parquet")):
    docs += pq.read_table(p).column("text").to_pylist()
random.Random(42).shuffle(docs)
with open("/home/jev/data/facts_src.jsonl", "w", encoding="utf-8") as f:
    for d in docs[:5000]:
        f.write(json.dumps({"text": d[:3000]}, ensure_ascii=False) + "\n")
PY
```

**6.3 Генерация контрастивных пар** (EN — `--lang en`; по рецепту Nimble достаточно 2.5–4k пар):

```bash
export JEV_API_BASE=http://<teacher-host>:8000/v1 JEV_MODEL=qwen3-14b JEV_API_KEY=""
.venv/bin/make-contrastive --input /home/jev/data/facts_src.jsonl \
    --output /home/jev/data/jev_en --lang en --workers 8 --val-frac 0.1
# resume: повторный запуск добирает только недостающие тексты из raw.jsonl
```

**6.4 Обучение адаптера на каждой базе** (один рецепт, разная база — это и есть тест H3):

```bash
# на MoVE-базе:  configs/jev_lora_d12_move_en.yaml  (jev_base_checkpoint: checkpoints/d12_move_x1_en)
# на контроле:   configs/jev_lora_d12_off_en.yaml   (jev_base_checkpoint: checkpoints/d12_off_en)
.venv/bin/jev-train --config configs/jev_lora_d12_move_en.yaml
```

После обучения в `jev_checkpoints/<model_tag>/` появятся `adapter_*.pt` и `calibration.json`
(per-task температуры, подогнанные на val).

**6.5 Оценка и инференс:**

```bash
.venv/bin/jev-eval --config configs/jev_lora_d12_move_en.yaml   # acc/ECE/Brier/logloss до/после калибровки
.venv/bin/jev-ask  --config configs/jev_lora_d12_move_en.yaml --task noul \
    --text "The Eiffel Tower is located in Paris." \
    --question "The Eiffel Tower is located in Paris?" \
    --question "The Eiffel Tower is located in Berlin?"
```

Критерий H3: при равном числе шагов адаптер на move-базе ≥ адаптера на off-базе по noul-accuracy
и имеет ECE не выше после калибровки.

## 7. Критерии успеха и анализ

1. **H1**: Δbpb(C−A) < −0.005 на matched-token точках (полный эффект статьи — минус 0.019…0.041,
   но на 6× большем бюджете); ожидаемо более сильный сигнал на fact-bpb/cloze, чем на общем bpb.
2. **H2**: E лучше B; если нет — фиксируем «alt достаточно».
3. **H3**: см. 6.5. Побочный артефакт: кривые loss-vs-tokens всех рукавов → проверка «8–10 tok/param»
   (VE-рукавы должны достигать уровня контроля A на меньшем числе токенов).
4. Отчёт — таблица по точкам {1000, 2000, 3000, 4767} шагов: val_bpb, fact_bpb, cloze_acc (+ gate-статы для VE-рукавов).

## 8. Статус кодовой базы

Уже готово и протестировано (67 тестов): модель d12 off/lave/move, гейты, base-train/base-eval с grad accum
и резюмом, JevAdapter/LoRA, `jev-train`/`jev-eval`/`jev-ask`, калибровка, `make-contrastive` (RU), `prepare-ruwiki`.

Для запуска плана все доработки A1–A7 реализованы в этом репо (тесты: 90 зелёных):

| # | Доработка | Статус |
|---|---|---|
| A1 | `prepare-hf` — HF parquet → шарды (`--dataset/--subsample/--max-tokens`, resume, `VAL_SHA256.txt`) | ✅ |
| A2 | `tok-train --lang en\|ru\|both`, `--extra-data-dir`, жёсткий sanity-чек кандидатов (exit при multi-token; `--allow-multi-token` — только предупреждение) | ✅ |
| A3 | `lave_layers: "alt"\|"all"\|"deep"` или явный список слоёв | ✅ |
| A4 | `make-contrastive --lang ru\|en` (EN true/false statement; ответы yes/no; язык пишется в raw.jsonl) | ✅ |
| A5 | `eval-facts` — fact-bpb + entity-cloze (`--build` из val-шардов, `--model-tag/--steps`) | ✅ |
| A6 | Конфиги `base_d12_{off,lave,move_x1,move_x4,lave_deep}_en.yaml` (batch 524288, 4768 шагов) и `jev_lora_d12_{move,off}_en.yaml` | ✅ |
| A7 | `ve/bank_norm_mean`, `ve/bank_dead_frac`, `ve/gate_norm_{mean,max}` в `log.jsonl` на каждом eval | ✅ |

## 9. Риски

- **MixtureVitae**: структура датасета не инспектирована; если streaming-доступ неудобен — фаза 0 идёт
  только на FineWeb-Edu, а cloze-eval строится из held-out val-документов (план это допускает).
- **VRAM рукава D** (банк 1.2B + AdamW-state ~10G): smoke 5.0 обязателен; запасной путь `device_batch_size=4`.
- **Мульти-токен кандидаты** в новом токенизаторе — ловится sanity-чеком 4.3 до обучения.
- **Гигиена данных**: val-шарды фиксируются после записи (контрольная сумма в `data/VAL_SHA256`),
  смешение языков в корпус логируется (уроки gefen: перезаписанный val-шард и EN-грязь в RU-корпусе).
