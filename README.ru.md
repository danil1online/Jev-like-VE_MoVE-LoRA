# Jev-like-VE / MoVE-LoRA

Обучение базовой модели с value-эмбеддингами (VE) / MoVE с нуля на корпусах с
высокой плотностью фактов — `atBuba/ruwiki-dataset` или HF-parquet датасеты
вроде FineWeb-Edu (`prepare-hf`) — по pipeline в стиле nanochat, плюс
**Jev-Like-LoRA адаптер** поверх замороженной базы для типизированных решений:

- **NOUL** — да/нет по контексту + вопросу (слова ответов настраиваются:
  `да`/`нет` или `yes`/`no` через `jev.yes_token` / `jev.no_token`)
- **choice** — ответы одной буквой (кириллица или латиница, `jev.letter_tokens`)
- **score** — 0..9

Адаптер — это low-rank LoRA (по умолчанию q/k/v) плюс обучаемые строки для
специальных токенов `<|ctx|>`, `<|q|>`, `<|a|>` и заголовок, ограниченный
кандидатами (`extra_in` / `extra_out`), поэтому инференс — это один forward
pass с `softmax` по кандидатам ответа.

Спецификация: `docs/gemini_talk.md`. План экспериментов (сетка VE/MoVE с
равными бюджетами, команды для RTX 3090): `docs/experiments.md`. Ссылка по
MoVE: arXiv:2601.22887.

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
    prep_hf.py      prepare-hf: HF-parquet датасеты (напр. FineWeb-Edu) -> шарды
    contrastive.py  генерация NOUL-контрастива через OpenAI-совместимый API (--lang ru|en)
  train/
    base.py         base-train / base-eval
    eval_facts.py   eval-facts (fact-bpb + entity-cloze проба для базовых чекпоинтов)
    jev_lora.py     jev-train / jev-eval
    jev_ask.py      jev-ask (инференс System One: текст + вопрос -> ответ + p)
    tokenizer.py    tok-train (--lang ru|en|both, проверка single-token кандидатов)
configs/            base_d12_* (+ англ. рукава *_en), jev_lora_d12*; docs/experiments.md (план)
tests/              pytest-сьют (CPU, крошечные модели)
```

## Установка

Python >= 3.10, [uv](https://docs.astral.sh/uv/).

```bash
# CPU
uv sync --extra cpu --group dev
# GPU (CUDA 12.8)
uv sync --extra gpu --group dev
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
`jev-train`, `jev-eval`, `jev-ask`, `eval-facts`, `tok-train`, `prepare-ruwiki`,
`prepare-hf`, `make-contrastive`) через `[project.scripts]`. Для них проект нужно один раз установить:

```bash
uv pip install -e .        # из корня репо, внутри venv
```

Пакет также можно использовать как обычный импорт из корня проекта без
установки (`python -m jevelike.train.base ...`), но entry points `*-eval`
достигаются только через установленные консольные скрипты, поэтому установка —
самый простой путь.

## Pipeline

1. **Подготовка корпуса** — parquet-шарды с одной колонкой `text`, в sorted
   порядке хвостовые шарды — val-сплит. Либо ruwiki (маленький, хорош для
   покрытия кириллицы в токенизаторе):

   ```bash
   prepare-ruwiki --num-shards 100 --seed 42
   # --input ruwiki_full.txt — использовать локальный дамп (без скачивания)
   # --no-download: ошибка вместо скачивания, если --input не задан
   ```

   либо любой HF-parquet датасет (FineWeb-Edu и т.п.) с возобновляемым кэшем
   скачивания в `<data>/hf_cache` и манифестом `VAL_SHA256.txt`:

   ```bash
   prepare-hf --dataset HuggingFaceFW/fineweb-edu --subsample sample-10BT \
       --max-tokens 4000000000 --num-shards 200
   # HF_TOKEN для gated-датасетов; --skip-download перешардирует готовый кэш
   ```

2. **Обучение токенизатора** (rustbpe BPE, 4 специальных токена
   `<|bos|> <|ctx|> <|q|> <|a|>`), сохраняется в `<base>/tokenizer`. `--lang`
   задаёт, какие кандидаты Jev обязаны быть single-token (жёсткая проверка;
   `--allow-multi-token` ослабляет до предупреждения). Для англ. базы, к которой
   позже захотят задавать русские вопросы, учим двуязычно:

   ```bash
   tok-train --vocab-size 65536 --max-chars 2000000000
   # англ. корпус + покрытие кириллицы (extra-каталоги читаются первыми):
   tok-train --lang both --extra-data-dir /path/to/ruwiki_shards
   ```

3. **Предобучение базовой модели**:

   ```bash
   base-train --config configs/base_d12_off.yaml    # vanilla transformer
   base-train --config configs/base_d12_move.yaml   # MoVE value-эмбеддинги
   base-train --config configs/base_d12_lave.yaml   # LaVE (per-layer значения)
   base-train --config configs/base_d20_move.yaml   # большая D20
   # англ. рукава с равным бюджетом (docs/experiments.md):
   #   base_d12_off_en / _lave_en / _move_x1_en / _move_x4_en / _lave_deep_en
   # --set key=value ... для переопределений, --device-type cuda|cpu,
   # --data-dir DIR, --run-name NAME
   ```

    Чекпоинты складываются в `checkpoints/<model_tag>/model_XXXXXX.pt`
    (+ meta, optimizer). `base-eval --model-tag d12_move --step 100000`
    выдаёт val bpb. Каждый eval также пишет health VE (`ve/bank_norm_mean`,
    `ve/bank_dead_frac`, `ve/gate_norm_*`) в `runs/<tag>/<run>/log.jsonl`;
    бюджет val на один eval задаётся ключом `eval_tokens` (по умолчанию ~4.2M).

3b. **Проба фактной памяти** (H1 в docs/experiments.md): один раз собрать
    пробный набор из held-out val-шардов, затем оценивать чекпоинты по fact-bpb
    и entity-cloze top-1:

   ```bash
   eval-facts --build --data data/facts_eval.jsonl
   eval-facts --model-tag d12_move_x1_en --steps 1000,4767 --data data/facts_eval.jsonl
   ```

4. **Сборка Jev-датасета** (LLM-as-Teacher, контрастивные пары, возобновляемо;
   нужен OpenAI-совместимый API, env `JEV_API_BASE` / `JEV_API_KEY` /
   `JEV_MODEL`). Вход: JSONL-строки `{"text": ...}` или обычный `.txt` файл
   (один текст на строку). Для каждого текста LLM выдаёт `true_statement` и
   `false_statement`, отличающееся ровно одним фактом; получается две строки
   (ответ `да`/`нет`, или `yes`/`no` с `--lang en`) с общим `pair_id`. Выход:
   `<output>/raw.jsonl` (возобновляемый) плюс детерминированное разбиение на
   `train.jsonl` / `val.jsonl`, где обе строки пары всегда попадают в один сплит:

   ```bash
   make-contrastive --input texts.jsonl --output data/jev --val-frac 0.1
   # английские пары (yes/no):
   make-contrastive --input texts.jsonl --output data/jev_en --lang en
   # пере-разбить из готового raw.jsonl без вызовов API:
   make-contrastive --input texts.jsonl --output data/jev --split-only
   ```

5. **Обучение Jev-Like-LoRA адаптера** (база заморожена):

   ```bash
   jev-train --config configs/jev_lora_d12.yaml
   ```

   Адаптеры складываются в `jev_checkpoints/<model_tag>/adapter_XXXXXX.pt`.
   После обучения на val-сплите подбирается post-hoc температура по каждой
   задаче (минимизация log-loss) и записывается в
   `jev_checkpoints/<model_tag>/calibration.json`.

6. **Оценка адаптера** (accuracy / ECE / brier / logloss по каждой задаче,
   с сравнением до/после калибровки, если есть `calibration.json`):

   ```bash
   jev-eval --config configs/jev_lora_d12.yaml [--adapter PATH]
   ```

7. **Вопрос адаптеру** (инференс System One: один forward pass, слово-ответ +
   вероятность по кандидатам задачи; `--question` повторяемый, все вопросы
   идут одним forward pass'ом; калиброванная температура применяется
   автоматически):

   ```bash
   jev-ask --config configs/jev_lora_d12.yaml --task noul \
       --text "Иван Иванов оформил возврат товара." \
       --question "Иван Иванов оформил возврат товара?"
   # контекст из stdin; переопределить температуру: --temperature
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
| `move.lave_layers` | `"alt"` (через слой), `"all"`, `"deep"` (глубокая половина, напр. 6..11 для d12) или явный список `[6,7,8]` (LaVE) |
| `lora.rank/alpha/target_modules` | `q,k,v,proj,fc,mlp_proj` |
| `jev.letter_tokens` | 28 букв `АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЭЮЯ` (без Й/Ё/Ъ/Ы/Ь) |
| `total_batch_size: -1` | auto = `target_param_data_ratio` x scaling-параметры (12x); в рукавах с равным бюджетом `*_en` задан явно, чтобы размер банка не сдвигал batch |
| `eval_tokens` | val-токены на один eval во время base-train (по умолчанию ~4.2M) |

Раскладка кандидатов ответа (дефолтный RU-конфиг, всего 40): `[0:2]` да/нет,
`[2:30]` буквы, `[30:40]` цифры. В англ. конфигах (`*_en`) — `yes`/`no` + A–Z
(всего 38); раскладка всегда `[yes,no] + буквы + цифры`.

## Тесты

Только CPU, крошечные модели, без скачивания данных:

```bash
.venv/bin/python -m pytest tests/ -q        # 90 тестов
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
  `extra_out`, выдающего логиты для кандидатов ответа (40 при дефолтном
  RU-конфиге). При инициализации адаптер
  (численно) тождественен: `extra_out` копирует строки `lm_head`, LoRA `B = 0`.
- Внимания в BTHD; логиты softcapped; init std `lm_head` = 0.001 (не связан
  с токеновским эмбеддингом).
- **Вероятности по задачам**: `adapter.probs` — softmax по всем кандидатам
  ответа; per-task вероятности (метрики, `jev-ask`) — restricted softmax
  по слайсу задачи, затем post-hoc температура из `calibration.json`
  (`calibrate_probs` / `fit_temperature` в `jev.py`).
- Порядок слияний rustbpe на маленьких корпусах нельзя контролировать по
  частоте, поэтому тесты никогда не полагаются на то, что конкретное слово —
  один токен.
