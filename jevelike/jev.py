"""
Jev-Like adapter: typed-decision tasks on top of a frozen base model.

Prompt format:  <|bos|> <|ctx|> {context} <|q|> {question} <|a|>
The answer is a single token from a small candidate set (да/нет, letters, digits),
predicted at the last position of the prompt. Loss is restricted cross-entropy
over the candidate tokens.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from jevelike.configs import LoraConfig, JevConfig
from jevelike.gpt import Linear
from jevelike.lora import apply_lora, freeze_base, LoRALinear, TARGET_MAP

SPECIAL_TOKENS = ("<|bos|>", "<|ctx|>", "<|q|>", "<|a|>")
IGNORE_INDEX = -100


# -----------------------------------------------------------------------------
# Rendering / batching

class JevRenderer:
    """Renders Jev items to (ids, target_token) and pads them into batches."""

    def __init__(self, tokenizer, jev_cfg: JevConfig):
        self.tokenizer = tokenizer
        self.cfg = jev_cfg
        self.bos_id = tokenizer.encode_single_token("<|bos|>")
        self.ctx_id = tokenizer.encode_single_token("<|ctx|>")
        self.q_id = tokenizer.encode_single_token("<|q|>")
        self.a_id = tokenizer.encode_single_token("<|a|>")
        self.words_by_task = jev_cfg.candidate_words()
        self.cand_ids = [tokenizer.encode_single_token(w) for w in self.words_by_task["noul"]] \
            + [tokenizer.encode_single_token(w) for w in self.words_by_task["choice"]] \
            + [tokenizer.encode_single_token(w) for w in self.words_by_task["score"]]
        self.word_to_cand = {w: i for i, w in enumerate(
            self.words_by_task["noul"] + self.words_by_task["choice"] + self.words_by_task["score"])}

    def candidates_for(self, task: str):
        assert task in self.words_by_task, f"Unknown task {task!r}; valid: {list(self.words_by_task)}"
        words = self.words_by_task[task]
        return words, [self.word_to_cand[w] for w in words]

    def render_one(self, text: str, question: str, answer: str):
        """Returns (ids, target_token_id). The answer is predicted at position len(ids)-1."""
        assert answer in self.word_to_cand, f"Answer {answer!r} is not a candidate word"
        text_ids = self.tokenizer.encode(text)
        q_ids = self.tokenizer.encode(question)
        ids = [self.bos_id, self.ctx_id] + text_ids + [self.q_id] + q_ids + [self.a_id]
        target = self.tokenizer.encode_single_token(answer)
        return ids, target

    def make_batch(self, items, max_len: int = None):
        """
        items: list of dicts {text, question, answer, task}.
        Returns x (B,T) right-padded, labels (B,T) with IGNORE_INDEX except at the
        answer position (len(ids)-1), and meta (per-item info).
        """
        rendered = [self.render_one(it["text"], it["question"], it["answer"]) for it in items]
        L = max(len(ids) for ids, _ in rendered)
        if max_len is not None:
            assert L <= max_len, f"Batch length {L} exceeds max_len {max_len}"
        B = len(rendered)
        x = torch.zeros(B, L, dtype=torch.long)
        labels = torch.full((B, L), IGNORE_INDEX, dtype=torch.long)
        meta = []
        for i, (it, (ids, target)) in enumerate(zip(items, rendered)):
            x[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            labels[i, len(ids) - 1] = target
            meta.append({"task": it.get("task", "noul"), "answer": it["answer"],
                         "pos": len(ids) - 1})
        return x, labels, meta


# -----------------------------------------------------------------------------
# Loss / probabilities / metrics

def jev_ce_loss(logits, labels, cand_ids, temperature: float = 1.0):
    """
    Restricted cross-entropy over candidate words.
    logits: (B,T,V) full vocab or (B,T,K) already restricted to cand_ids.
    labels: (B,T) token ids, IGNORE_INDEX (-100) marks masked positions.
    """
    K = len(cand_ids)
    if logits.size(-1) != K:
        cand = torch.as_tensor(cand_ids, device=logits.device, dtype=torch.long)
        logits = logits.index_select(-1, cand)
    logit_dtype = logits.dtype
    cand_tensor = torch.as_tensor(cand_ids, device=logits.device, dtype=torch.long)
    target_idx = torch.full_like(labels, -1, dtype=torch.long)
    for ci in range(K):
        target_idx = torch.where(labels == cand_tensor[ci], torch.tensor(ci, device=labels.device), target_idx)
    valid = labels != IGNORE_INDEX
    if not valid.any():
        return logits.new_zeros(())
    logp = F.log_softmax(logits.to(torch.float32) / temperature, dim=-1)
    nll = -logp[valid, target_idx[valid]]
    return nll.mean().to(logit_dtype)


def answer_probs(logits, cand_ids, temperature: float = 1.0):
    """Softmax probabilities over candidate words. logits (B,T,V) or (B,T,K) -> (B,T,K)."""
    K = len(cand_ids)
    if logits.size(-1) != K:
        cand = torch.as_tensor(cand_ids, device=logits.device, dtype=torch.long)
        logits = logits.index_select(-1, cand)
    return torch.softmax(logits.to(torch.float32) / temperature, dim=-1)


def decide(probs, cand_words):
    """argmax over candidates: probs (B,T,K) -> (B,T) word indices."""
    return torch.argmax(probs, dim=-1)


def calibration_metrics(probs, target_idx, bins: int = 15):
    """
    probs: (N, K) class probabilities; target_idx: (N,) ground-truth class index.
    Returns accuracy / ECE / Brier / log-loss.
    """
    probs = probs.to(torch.float32).clamp(min=1e-12)
    target_idx = target_idx.to(torch.long)
    pred = torch.argmax(probs, dim=-1)
    correct = (pred == target_idx).to(torch.float32)
    conf = probs.gather(1, target_idx.unsqueeze(1)).squeeze(1)
    onehot = F.one_hot(target_idx, probs.size(-1)).to(torch.float32)
    with torch.no_grad():
        acc = correct.mean().item()
        brier = ((probs - onehot) ** 2).sum(dim=-1).mean().item()
        logloss = (-conf.log()).mean().item()
        bin_ids = (conf * bins).floor().long().clamp(max=bins - 1)
        ece = 0.0
        for b in range(bins):
            mask = bin_ids == b
            if mask.any():
                ece += mask.float().mean().item() * abs(conf[mask].mean().item() - correct[mask].mean().item())
    return {"accuracy": acc, "ece": ece, "brier": brier, "logloss": logloss, "n": int(target_idx.numel())}


def calibrate_probs(probs, temperature: float):
    """Post-hoc temperature scaling: softmax(log(p) / T). T=1 is a no-op."""
    if temperature == 1.0:
        return probs
    logp = torch.log(probs.to(torch.float32).clamp(min=1e-12))
    return torch.softmax(logp / temperature, dim=-1)


def fit_temperature(probs, target_idx, lo: float = 0.05, hi: float = 10.0, steps: int = 200):
    """Fit a post-hoc temperature on validation probs by minimizing log-loss.
    Returns (temperature, best_logloss)."""
    probs = probs.to(torch.float32).clamp(min=1e-12)
    target_idx = target_idx.to(torch.long)
    logp = torch.log(probs)
    grid = torch.linspace(math.log(lo), math.log(hi), steps).exp().tolist()
    best_t, best_ll = 1.0, float("inf")
    for t in grid:
        p = torch.softmax(logp / t, dim=-1)
        ll = F.nll_loss(torch.log(p.clamp(min=1e-12)), target_idx).item()
        if ll < best_ll:
            best_t, best_ll = t, ll
    return best_t, best_ll


# -----------------------------------------------------------------------------
# Adapter

class JevAdapter(nn.Module):
    """
    Frozen base model +:
      - LoRA branches on the chosen attention projections
      - extra_in:  learnable embedding rows for <|ctx|>/<|q|>/<|a|> (init from base wte)
      - extra_out: (n_embd -> K) candidate logit head (init from base lm_head rows)
    The base lm_head is not used in the Jev forward pass (return_hidden=True).
    At init (LoRA B=0, extra_in == wte rows, extra_out == lm_head rows) the adapter
    reproduces the base model's restricted answer logits (up to the base softcap,
    which is ~identity at init-scale logits).
    """

    def __init__(self, model, lora_cfg: LoraConfig, jev_cfg: JevConfig, tokenizer):
        super().__init__()
        self.model = model
        self.jev_cfg = jev_cfg
        self.renderer = JevRenderer(tokenizer, jev_cfg)
        self.lora_adapters = apply_lora(model, lora_cfg)
        freeze_base(model)

        self.special_tokens = ("<|ctx|>", "<|q|>", "<|a|>")
        self.special_ids = torch.tensor(
            [tokenizer.encode_single_token(t) for t in self.special_tokens], dtype=torch.long
        )
        n_embd = model.config.n_embd
        wte_dtype = model.transformer.wte.weight.dtype
        with torch.no_grad():
            init_rows = model.transformer.wte.weight[self.special_ids].clone().to(wte_dtype)
        self.extra_in = nn.Parameter(init_rows)  # (3, n_embd)

        K = len(self.renderer.cand_ids)
        self.cand_ids = torch.tensor(self.renderer.cand_ids, dtype=torch.long)
        self.extra_out = Linear(n_embd, K, bias=False)
        with torch.no_grad():
            base_rows = model.lm_head.weight[self.cand_ids]  # (K, n_embd)
            self.extra_out.weight.copy_(base_rows)

    def embed_fn(self, idx):
        """wte with learnable rows for the ctx/q/a special tokens."""
        x = self.model.transformer.wte(idx)
        x = x.to(dtype=self.extra_in.dtype)
        for i, sid in enumerate(self.special_ids.tolist()):
            x = torch.where((idx == sid).unsqueeze(-1), self.extra_in[i].view(1, 1, -1), x)
        return x

    def hidden(self, idx):
        return self.model(idx, embed_fn=self.embed_fn, return_hidden=True)

    def forward(self, idx, labels):
        h = self.hidden(idx)
        cand_logits = self.extra_out(h)
        loss = jev_ce_loss(cand_logits, labels, self.cand_ids.tolist(), self.jev_cfg.temperature)
        return loss

    @torch.no_grad()
    def probs(self, idx):
        h = self.hidden(idx)
        return answer_probs(self.extra_out(h), self.cand_ids.tolist(), self.jev_cfg.temperature)

    def trainable_params(self):
        params = []
        for adapter in self.lora_adapters:
            params.extend([adapter.lora_A, adapter.lora_B])
        params.append(self.extra_in)
        params.append(self.extra_out.weight)
        return params

    # ------------------------------------------------------------------
    # Checkpointing: compact adapter state (trainable weights only, no base)
    # ------------------------------------------------------------------

    def _lora_param_map(self):
        target = {}
        for i, block in enumerate(self.model.transformer.h):
            for block_attr, module_attr in TARGET_MAP.values():
                m = getattr(getattr(block, block_attr), module_attr)
                if isinstance(m, LoRALinear):
                    prefix = f"h.{i}.{block_attr}.{module_attr}"
                    target[f"{prefix}.lora_A"] = m.lora_A
                    target[f"{prefix}.lora_B"] = m.lora_B
        return target

    def adapter_state_dict(self):
        """Compact state of the trainable adapter only (LoRA A/B + extra_in/out).
        The frozen base model is NOT included; it must be loaded separately from
        the base checkpoint (see meta['base_checkpoint'])."""
        state = {k: p for k, p in self._lora_param_map().items()}
        state["extra_in"] = self.extra_in
        state["extra_out.weight"] = self.extra_out.weight
        return state

    def load_adapter_state(self, state):
        """Load a compact adapter state into an already-built adapter (base weights
        come from the base checkpoint). Legacy full checkpoints (keys prefixed with
        'model.') are still accepted."""
        state = dict(state)
        if any(k.startswith("model.") for k in state):
            self.load_state_dict(state, strict=True)
            return
        target = self._lora_param_map()
        target["extra_in"] = self.extra_in
        target["extra_out.weight"] = self.extra_out.weight
        unknown = sorted(set(state) - set(target))
        if unknown:
            raise ValueError(f"Unknown adapter state keys: {unknown}")
        missing = sorted(set(target) - set(state))
        if missing:
            raise ValueError(f"Missing adapter state keys (LoRA target mismatch?): {missing}")
        with torch.no_grad():
            for k, v in state.items():
                target[k].copy_(v.to(target[k].dtype))
