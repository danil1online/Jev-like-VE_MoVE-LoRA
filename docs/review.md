# Ревизия кода (2026-10-02)

Полная ревизия `jevelike/*`, `train/*`, `data/*`, `configs/*`, `tests/*` с сопоставлением
с требованиями `docs/gemini_talk.md`. Все пути относ. от корня репозитория.

## A. Реальные баги (P0)

### A1. Адаптер-чекпоинт содержит всю базовую модель
- Где: `jevelike/train/jev_lora.py:263`, `jevelike/checkpoint.py:141-152`.
- Комментарий в `checkpoint.py` обещает «trainable adapter only, no base weights», но
  `save_adapter(..., adapter.state_dict(), ...)` сохраняет весь `JevAdapter`, а `self.model`
  — зарегистрированный сабмодуль: в файл уходят все `model.*` (wte, lm_head, move_bank,
  трансформер + инлайновые LoRA A/B). Для d12_move это ~1.2 ГБ на чекпоинт
  (move_bank ~600 МБ bf16, wte+lm_head ~200 МБ, трансформер ~360 МБ), ×3 при
  `jev_save_every=1000` / 3000 шагов.
- Хуже: при `jev-eval`/resume `load_adapter` → `adapter.load_state_dict(adapter_state)`
  перезаписывает только что собранные веса базы сохранёнными — `cfg.jev_base_checkpoint`
  в конфиге фактически игнорируется (тихий silent override).
- Фикс: компактный state (только LoRA A/B по `TARGET_MAP` + `extra_in` + `extra_out.weight`),
  загрузка в уже собранный адаптер (после `apply_lora`), warning при несовпадении
  `base_checkpoint` в meta с конфигом, backward-compat со старыми полными файлами.

### A2. `pin_memory` / `non_blocking` никогда не включаются
- Где: `jevelike/dataloader.py:92` — `use_cuda = device == "cuda"`, но `device` —
  `torch.device`; `torch.device('cuda') == 'cuda'` → `False` (проверено).
- Фикс: нормализовать `device` и проверять `device.type == "cuda"`.

### A3. `softcap` из конфига не используется
- Где: `jevelike/gpt.py:372` — `softcap = 15` захардкожен; `ModelConfig.softcap`
  (и `softcap: 15.0` во всех 5 YAML) — мёртвый ключ.
- Фикс: `softcap = self.config.softcap`.

### A4. LoRA dropout никогда не активен
- Где: `jevelike/train/jev_lora.py` — база строится с `phase="eval"` и `adapter.train()`
  нигде не вызывается; после `evaluate_adapter` (который ставит `adapter.eval()`) train-режим
  не восстанавливается.
- Фикс: `adapter.train()` после сборки адаптера + восстановление после eval.

### A5. Мёртвые/неподключённые ключи конфига
- Где: `jevelike/configs.py:223-224`, 4 YAML баз.
- `grad_clip: 1.0` объявлен, но в `train/base.py` клиппинга нет вовсе → подключить
  `torch.nn.utils.clip_grad_norm_` после `scaler.unscale_` (и в non-scaler ветке перед `step`).
- `muon_momentum: 0.95` — в референсе (nanochat `base_train.py:372`) momentum шедулится
  хардкодом 0.85→0.97→0.90, config-ключа нет; наш порт совпадает с референсом →
  удалить ключ из `TrainConfig` и YAML (чтобы конфиг не давал ложной иллюзии управления).

## B. Разрывы с требованиями `docs/gemini_talk.md` (P1)

### B1. Контрастивный генератор не генерирует пары
- Где: `jevelike/data/contrastive.py`.
- Рецепт дока (LLM-as-Teacher, шаги 1–2, по arXiv:2507.22729): на каждый текст LLM выдаёт
  `true_statement` + `false_statement`, отличающиеся ровно одним фактом → 2 строки (да/нет).
  Текущий `make-contrastive` лишь отвечает да/нет на уже готовый {text, question} — это
  лейблер, а не пар-генератор; суть contrastive-обучения отсутствует.
- Дополнительно: нет JSON-mode (релимся на первое слово ответа), только noul (нет choice/score),
  один выходной JSONL без split (в README ручной сникпет 90/10).
- Фикс: пар-генерация (true/false statement, 1 факт), JSON-mode, флаг `--val-frac`.

### B2. Нет инференс-интерфейса System One
- Doc: (state, вопросы) → вероятности за один forward pass, пример кода с `num_predict: 1`.
- В репо есть `JevAdapter.probs()` / `decide()`, но нет CLI, чтобы задать вопрос обученной
  модели (`jev-eval` печатает только метрики).
- Фикс: `jev-ask` — текст + список вопросов → ответ + калиброванная вероятность
  (контекст кодируется один раз, все вопросы — в одном проходе).

### B3. Калибровка измеряется, но не оптимизируется
- Doc подчёркивает RLCD-калибровку; сейчас CE + температура, ECE/Brier только в логах.
- Фикс: подгонка температуры на val (1 параметр) после обучения, применение в `probs`,
  отчёт в `jev-eval` (до/после).

## C. Улучшения (P2)

1. **Per-task маска в лоссе** (`jevelike/jev.py:68-96`): `jev_ce_loss` всегда ограничивает
   по всем 40 кандидатам — noul-пример размывает массу на 38 нерелевантных букв/цифр.
   `JevRenderer` уже знает слайсы; локальная правка, должна улучшить калибровку.
2. **Render-cache** (`jev_lora.py:86-100`): кэш падает в `./cache/` от CWD (докстринг —
   «next to the data»); ключ кэша (md5 текста) не включает идентичность токенизатора —
   перетренировал токенизатор → stale кэш с чужими id молча используется.
3. **`eval_tokens = 4.2M` захардкожен** (`base.py:268`) — на CPU smoke-раны
   `eval_every=250` мучительно медленные; вынести в конфиг/аргумент.
   — Исправлено (2026-10-07): стал ключом `TrainConfig.eval_tokens` (по умолчанию 4×2^20).
4. **Best-адаптер**: `jev-eval` берёт max step, а не лучший val-метрику; сохранять `best.pt`.
5. **`prep.py`**: при `len(docs) < total_shards` → `shard_size=0` → много пустых parquet;
   клампнуть `num_shards` к числу доков.
6. **DDP наполовину**: dataloader/optimizer/common имеют rank-пути, но `base.py` не
   инициализирует dist и не оборачивает модель — `torchrun` небезопасен. Для одной 3090 ок,
   но задокументировать «single-process only».
7. Мелочи: `cand_tensor` строится дважды (`jev.py:88,91`) + O(K) цикл `torch.where`;
   ECE по 15 равным бинам (можно квантильные); `base.py:69` — мёртвая ветка bool-конверсии
   после `json.loads`; `psutil`/`filelock` в pyproject не используются.

## Что хорошо

- Чистое разделение слоёв (gpt/move/lora/jev/loss_eval/optim/dataloader/checkpoint).
- MoVE/LaVE верны статье (покрыто тестами): общий банк `E(vocab×M×kv_dim)`,
  lookup один раз на forward, gate `scale·σ`, shared slot per layer pair.
- Честные чекпоинты/резюм базы (model+optim+dataloader+meta), scaling-law
  скейлинг batch/LR/WD, fused Muon/AdamW с chunking, ясные ошибки токенизатора
  (ValueError на мульти-токен кандидатах), 67 зелёных тестов.

## План

- **P0 (правильность):** A1–A5 — выполнено 2026-10-02 (см. git log).
- **P1 (требования дока):** B1, B2, B3 — выполнено 2026-10-03 (см. git log).
  - B1: `data/contrastive.py` переписан как пар-генератор (true/false statement,
    JSON-mode, `raw.jsonl` + детерминированный split с `--val-frac`/`--split-only`,
    обе строки пары в одном сплите).
  - B2: `train/jev_ask.py` + entry point `jev-ask` (текст + N вопросов → один
    forward pass → ответ + вероятность; `--temperature`, stdin-контекст).
  - B3: `jev.py:calibrate_probs`/`fit_temperature`; после `jev-train` — подбор
    per-task температуры на val → `jev_checkpoints/<tag>/calibration.json`;
    `jev-eval` и `jev-ask` применяют её (отчёт до/после).
  - Найденный при P1 баг: per-task метрики/ответы строились на срезе глобального
    softmax по 40 кандидатам (нормировка < 1) — заменено на restricted softmax
    по слайсу задачи (`collect_probs`, `jev_ask.ask`).
- **P2 (качество):** C1–C7.
