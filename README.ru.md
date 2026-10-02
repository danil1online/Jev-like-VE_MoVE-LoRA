# Jev-like-VE / MoVE-LoRA

Обучение базовой модели с value-эмбеддингами (VE) / MoVE с нуля на
`atBuba/ruwiki-dataset` (pipeline в стиле nanochat), плюс **Jev-Like-LoRA
адаптер** поверх замороженной базы для типизированных решений:

- **NOUL** — да/нет (`да` / `нет`) по контексту + вопросу
- **choice** — ответы одной буквой (28 букв кириллицы)
- **score** — 0..9

Адаптер — это low-rank LoRA (по умолчанию q/k/v) плюс обучаемые строки для
специальных токенов `<|ctx|>`, `<|q|>`, `<|a|>` и заголовок, ограниченный
кандидатами (`extra_in` / `extra_out`), поэтому инференс — это один forward
pass с `softmax` по 40 кандидатам ответа.

Спецификация: `docs/gemini_talk.md`. Ссылка по MoVE: arXiv:2601.22887.

## Структура

```
jevelike/
  common.py         init устройства/dtype, каталоги, логирование
  configs.py        ModelConfig / MoveConfig / LoraConfig / JevConfig / TrainConfig (YAML)
  move.py           банк значений MoVE + гейты, per-layer значения LaVE, resolve_move
  gpt.py            GPT (родословная nanochat): BTHD, rotary, скользящие окна, softcap
  flash_attention.py
  tokenizer.py      RustBPETokenizer (rustbpe + tiktoken), 4 специальных токена
  dataset.py        чтение parquet-шардов, буферизация row-group'ов
  dataloader.py     распределённый best-fit dataloader с выравниванием по BOS
  optim.py          группы параметров Muon + AdamW
  loss_eval.py      оценка bpb / ppl
  checkpoint.py     чекпоинты базы + сохранение/загрузка адаптера
  lora.py           LoRALinear, apply_lora, freeze_base
  jev.py            JevAdapter, JevRenderer, answer_probs, метрики калибровки
  data/
    prep.py         ruwiki -> parquet-шарды (train/val)
    contrastive.py  генерация NOUL-контрастива через OpenAI-совместимый API
  train/
    base.py         base-train / base-eval
    jev_lora.py     jev-train / jev-eval
    tokenizer.py    tok-train
configs/            base_d12_off / base_d12_move / base_d12_lave / base_d20_move / jev_lora_d12
tests/              pytest-сьют (CPU, крошечные модели)
```

## Установка

Python >= 3.10, [uv](https://docs.astral.sh/uv/).

```bash
# CPU
uv sync --extra cpu --group dev
# GPU (CUDA 12.8)
uv sync --extra gpu --group dev

# CPU torch (для свежего venv): uv выбирает индекс через [tool.uv.sources]
uv pip install --python .venv/bin/python --reinstall-package torch \
    --index-url https://download.pytorch.org/whl/cpu "torch==2.14.1+cpu"
```

> **Заметка о preload torch.** Если `LD_LIBRARY_PATH` машины содержит устаревший
> системный libtorch (например `/opt/libtorch/lib`), torch может загрузить не те
> разделяемые библиотеки. В этом репо есть site-level preload-шим:
> `.venv/lib/python3.10/site-packages/jevelike_torch_preload.pth` (+ `.py`),
> который через `ctypes.CDLL` загружает библиотеки venv'а
> `libgomp -> libc10 -> libtorch_cpu -> libtorch -> libshm -> libtorch_python`
> (RTLD_GLOBAL) раньше, чем torch. Не удаляйте эти файлы на этой машине;
> переустановка torch через uv сохраняет их рабочими.

Инструменты вынесены в консольные скрипты (`base-train`, `base-eval`,
`jev-train`, `jev-eval`, `tok-train`, `prepare-ruwiki`, `make-contrastive`)
через `[project.scripts]`. Для них проект нужно один раз установить:

```bash
uv pip install -e .        # из корня репо, внутри venv
```

Пакет также можно использовать как обычный импорт из корня проекта без
установки (`python -m jevelike.train.base ...`), но entry points `*-eval`
достигаются только через установленные консольные скрипты, поэтому установка —
самый простой путь.

## Pipeline

1. **Подготовка корпуса** (скачивает `atBuba/ruwiki-dataset`, пишет
   parquet-шарды `<data>/ruwiki_train_0000.parquet`, ... +
   `ruwiki_val_*.parquet`; последний шард — val-сплит):

   ```bash
   prepare-ruwiki --num-shards 100 --seed 42
   # --input ruwiki_full.txt — использовать локальный дамп (без скачивания)
   # --no-download: ошибка вместо скачивания, если --input не задан
   ```

2. **Обучение токенизатора** (rustbpe BPE, 4 специальных токена
   `<|bos|> <|ctx|> <|q|> <|a|>`), сохраняется в `<base>/tokenizer`:

   ```bash
   tok-train --vocab-size 65536 --max-chars 2000000000
   ```

3. **Предобучение базовой модели**:

   ```bash
   base-train --config configs/base_d12_off.yaml    # vanilla transformer
   base-train --config configs/base_d12_move.yaml   # MoVE value-эмбеддинги
   base-train --config configs/base_d12_lave.yaml   # LaVE (per-layer значения)
   base-train --config configs/base_d20_move.yaml   # большая D20
   # --set key=value ... для переопределений, --device-type cuda|cpu,
   # --data-dir DIR, --run-name NAME
   ```

   Чекпоинты складываются в `checkpoints/<model_tag>/model_XXXXXX.pt`
   (+ meta, optimizer). `base-eval --model-tag d12_move --step 100000`
   выдаёт val bpb.

4. **Сборка Jev-датасета** (NOUL-контрастив, возобновляемый; нужен
   OpenAI-совместимый API, env `JEV_API_BASE` / `JEV_API_KEY` / `JEV_MODEL`).
   Строки входного JSONL: `{"text": ..., "question": ...}`. Скрипт добавляет
   по одной строке JSONL на каждую отвеченную пару; затем разбейте её на два
   файла, которые ожидает тренер (`<jev_data_dir>/train.jsonl` и `val.jsonl`,
   например 90/10):

   ```bash
   make-contrastive --input my_pairs.jsonl --output data/jev/answers.jsonl
   python - <<'EOF'
   import json
   rows = [json.loads(l) for l in open("data/jev/answers.jsonl") if l.strip()]
   cut = int(len(rows) * 0.9)
   for name, part in (("train", rows[:cut]), ("val", rows[cut:])):
       with open(f"data/jev/{name}.jsonl", "w") as f:
           for r in part:
               f.write(json.dumps(r, ensure_ascii=False) + "\n")
   EOF
   ```

5. **Обучение Jev-Like-LoRA адаптера** (база заморожена):

   ```bash
   jev-train --config configs/jev_lora_d12.yaml
   ```

   Адаптеры складываются в `jev_checkpoints/<model_tag>/adapter_XXXXXX.pt`.

6. **Оценка адаптера** (accuracy / ECE / brier / logloss по каждой задаче):

   ```bash
   jev-eval --config configs/jev_lora_d12.yaml [--adapter PATH]
   ```

## Справочник по конфигам

`configs/*.yaml` (YAML 1.1 — `"off"` нужно в кавычках, иначе спарсится как bool).

| Ключ | Значение |
|---|---|
| `model.n_layer/n_head/n_kv_head/n_embd` | D12 = 12/6/6/768, D20 = 20/10/10/1280 |
| `model.window_pattern` | `S` = локальный (пол-окна), `L` = полный; паттерн тайлитится, последний слой всегда полный |
| `model.softcap` | softcap логитов (15.0) |
| `move.mode` | `"off"`, `"move"`, `"lave"` |
| `move.num_slots` | слоты MoVE на слой (auto = `n_layer // 2`) |
| `move.gated_standard` | гейтить стандартный V рядом со слотами MoVE (только MoVE) |
| `move.lave_layers` | `"alt"` (через слой) или `"all"` (LaVE) |
| `lora.rank/alpha/target_modules` | `q,k,v,proj,fc,mlp_proj` |
| `jev.letter_tokens` | 28 букв `АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЭЮЯ` (без Й/Ё/Ъ/Ы/Ь) |
| `total_batch_size: -1` | auto = `target_param_data_ratio` x параметры (12x) |

Раскладка кандидатов ответа (всего 40): `[0:2]` да/нет, `[2:30]` буквы,
`[30:40]` цифры.

## Тесты

Только CPU, крошечные модели, без скачивания данных:

```bash
.venv/bin/python -m pytest tests/ -q        # 44 теста
.venv/bin/python -m pytest tests/ -m "not slow"
```

`tests/conftest.py` предоставляет детерминированный `FakeTokenizer` (id
кандидатов Jev точные) и настоящий маленький rustbpe-токенизатор для
roundtrip-тестов.

## Заметки по дизайну

- **MoVE**: один общий банк `E: vocab -> M x kv_dim`; на слой — один lookup и
  `V = g0 ⊙ V + Σ_m g_m ⊙ M_m` с сигмоид-гейтами (scale 2.0, init ~0.5).
- **LaVE**: одно дополнительное значение на слой, гейта нет
  `V = V + g ⊙ M_1` на выбранных слоях (alt/all), дешевле MoVE.
- **JevAdapter**: базовый forward не трогают, кроме (a) LoRA-дельт на q/k/v,
  (b) подмены строк `extra_in` для `<|ctx|>/<|q|>/<|a|>`, (c) заголовка
  `extra_out`, выдающего логиты для 40 кандидатов. При инициализации адаптер
  (численно) тождественен: `extra_out` копирует строки `lm_head`, LoRA `B = 0`.
- Внимания в BTHD; логиты softcapped; init std `lm_head` = 0.001 (не связан
  с токеновским эмбеддингом).
- Порядок слияний rustbpe на маленьких корпусах нельзя контролировать по
  частоте, поэтому тесты никогда не полагаются на то, что конкретное слово —
  один токен.
