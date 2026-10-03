"""
GPT model with optional MoVE / LaVE value embeddings.
Architecture and training recipe follow nanochat (karpathy/nanochat):
  - rotary embeddings (base 1e4), QK norm, untied embeddings, ReLU^2 MLP,
    norm after token embedding, no bias, GQA, FA3/SDPA
  - residual Lambdas (x0 + x1 residual mixing), attention "smear", backout
  - logit softcap, CE with ignore_index=-1
Value embeddings (this project's core addition, see jevelike/move.py):
  - mode "move": one shared (vocab x M x kv_dim) bank, single lookup per forward,
    per-layer router gate, gated standard + extra paths (MoVE paper)
  - mode "lave": per-layer banks on alternating layers, standard path ungated
    by default (LaVE)
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from jevelike.common import COMPUTE_DTYPE, print0
from jevelike.flash_attention import flash_attn
from jevelike.move import ResolvedMove, MoveBank, mix_value


# -----------------------------------------------------------------------------
# Basic modules
# -----------------------------------------------------------------------------

def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Master weights stay fp32 for optimizer precision; matmuls run in activation dtype."""

    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype), self.bias)


def apply_rotary_emb(x, cos, sin):
    # rotates by -theta (transpose of textbook convention), kept for compatibility
    assert x.ndim == 4  # (B, T, H, Dh)
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


# -----------------------------------------------------------------------------
# Attention / MLP / Block
# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx, move: ResolvedMove):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.move = move
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)

        # value-embedding router gate
        self.ve_gate = None
        if move.has_bank(layer_idx):
            self.ve_gate = Linear(move.gate_in_dim, move.gate_out_dim(self.n_kv_head), bias=False)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()
        cos, sin = cos_sin

        # (B, T, H, D) - FA3's native layout, no transpose needed
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value embeddings
        if ve is not None:
            assert self.ve_gate is not None
            gate_in = x[..., :self.move.gate_in_dim]
            gate_logits = self.ve_gate(gate_in)  # (B, T, gate_out)
            H, D = self.n_kv_head, self.head_dim
            if self.move.is_move:
                M = self.move.num_slots
                ve = ve.view(B, T, M, H, D).permute(0, 1, 3, 2, 4)  # (B,T,H,M,D)
                if self.move.gated_standard:
                    gate_logits = gate_logits.view(B, T, H, M + 1)
                else:
                    gate_logits = gate_logits.view(B, T, H, M)
            else:  # lave: single slot per layer
                ve = ve.view(B, T, H, D)
                if self.move.gated_standard:
                    gate_logits = gate_logits.view(B, T, H, 2)
            v = mix_value(v, ve, gate_logits, self.move.gate_scale, self.move.gated_standard)

        # Rotary embeddings + QK norm
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q = q * 1.2  # sharper attention (split scale between Q and K)
        k = k * 1.2

        # Flash Attention (FA3 or SDPA fallback)
        if kv_cache is None:
            y = flash_attn(q, k, v, window_size, is_causal=True)
        else:
            k_cache, v_cache = kv_cache.k_cache[self.layer_idx], kv_cache.v_cache[self.layer_idx]
            pos = kv_cache.pos
            k_cache[:, pos:pos+T] = k
            v_cache[:, pos:pos+T] = v
            y = flash_attn(q, k_cache[:, :pos+T], v_cache[:, :pos+T], window_size, is_causal=True)
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx, move: ResolvedMove):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx, move)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


# -----------------------------------------------------------------------------
# KV cache (minimal, for the naive generate path)
# -----------------------------------------------------------------------------

class KVCache:
    def __init__(self, n_layers, max_seq_len, n_kv_head, head_dim, dtype, device):
        self.n_layers = n_layers
        self.k_cache = [torch.zeros(1, max_seq_len, n_kv_head, head_dim, dtype=dtype, device=device)
                        for _ in range(n_layers)]
        self.v_cache = [torch.zeros(1, max_seq_len, n_kv_head, head_dim, dtype=dtype, device=device)
                        for _ in range(n_layers)]
        self.pos = 0
        self.prev_embedding = None

    def get_pos(self):
        return self.pos

    def advance(self, n):
        self.pos += n


# -----------------------------------------------------------------------------
# GPT
# -----------------------------------------------------------------------------

class GPT(nn.Module):

    def __init__(self, config, move: ResolvedMove = None, pad_vocab_size_to=64):
        """
        NOTE: this __init__ may run in a meta device context (shapes/dtypes only);
        all data is initialized in init_weights().
        """
        super().__init__()
        self.config = config
        if move is None:
            move = ResolvedMove(mode="off", num_slots=0, gate_scale=1.0,
                                gate_in_dim=0, gated_standard=False, lave_layer_indices=())
        self.move = move

        # per-layer window sizes: (left, right) tuples; pattern tiled; final layer always L
        self.window_sizes = self._compute_window_sizes(config)

        # pad vocab for efficiency
        self.vocab_padded = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if self.vocab_padded != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {self.vocab_padded} for efficiency")

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(self.vocab_padded, config.n_embd),
            "h": nn.ModuleList([Block(config, i, move) for i in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, self.vocab_padded, bias=False)

        # per-layer learnable scalars
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # smear: mix previous token's normalized embedding into current position
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # backout: subtract cached mid-layer residual before final norm
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))

        # value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        if move.is_move:
            self.move_bank = MoveBank(self.vocab_padded, move.num_slots, kv_dim)
        if move.is_lave:
            self.value_embeds = nn.ModuleDict({
                str(i): MoveBank(self.vocab_padded, move.num_slots, kv_dim)
                for i in move.lave_layer_indices
            })

        # rotary embeddings (10X over-computed; small and cheap)
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @staticmethod
    def _compute_window_sizes(config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = [char_to_window[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        window_sizes[-1] = (long_window, 0)  # final layer always full context
        return window_sizes

    @staticmethod
    def _precompute_rotary_embeddings(seq_len, head_dim, base=100000, device=None):
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    # ------------------------------------------------------------------
    # Weight initialization
    # ------------------------------------------------------------------

    @torch.no_grad()
    def init_weights(self):
        """
        wte:        normal, std=0.8
        lm_head:    normal, std=0.001
        c_q/c_k/c_v: uniform, std=1/sqrt(n_embd); c_proj: zeros
        mlp.c_fc:   uniform, std=0.4/sqrt(n_embd); mlp.c_proj: zeros
        value banks: uniform, std=1/sqrt(n_embd); gates: uniform(0, 0.02)
        """
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5  # uniform bound achieving the same std as normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

        if self.move.is_move:
            self.move_bank.init_weights(s)
        for bank in getattr(self, "value_embeds", {}).values():
            bank.init_weights(s)

        # rotary embeddings
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, self.config.head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE (fp32 master weights are fine for the optimizer)
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            if self.move.is_move:
                self.move_bank.to(dtype=COMPUTE_DTYPE)
            for bank in getattr(self, "value_embeds", {}).values():
                bank.to(dtype=COMPUTE_DTYPE)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean',
                embed_fn=None, return_hidden=False):
        B, T = idx.size()
        assert T <= self.cos.size(1), \
            f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        # embed
        if embed_fn is not None:
            x = embed_fn(idx)
        else:
            x = self.transformer.wte(idx)
        x = x.to(dtype=COMPUTE_DTYPE)
        x = norm(x)

        # Smear: mix previous token's embedding into current position (cheap bigram info)
        if kv_cache is None:
            assert T > 1, "Training forward pass should have T > 1"
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        # trunk
        x0 = x  # initial normalized embedding for x0 residual
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2
        x_backout = None
        # MoVE: one shared bank lookup per forward, shared by all layers
        move_ve = self.move_bank(idx) if self.move.is_move else None
        value_embeds = getattr(self, "value_embeds", {})
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            if self.move.is_move:
                ve = move_ve
            elif str(i) in value_embeds:
                ve = value_embeds[str(i)](idx).to(x.dtype)
            else:
                ve = None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
            if i == backout_layer:
                x_backout = x
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        if return_hidden:
            return x

        softcap = self.config.softcap
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=loss_reduction)
            return loss
        return logits

    # ------------------------------------------------------------------
    # Parameter bookkeeping
    # ------------------------------------------------------------------

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        h, q, t = self.config.n_head, self.config.head_dim, self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * self.num_matmul_params() + attn_flops

    def num_matmul_params(self):
        """Params that participate in matmuls with the token stream (all Linear modules)."""
        return sum(m.weight.numel() for m in self.modules() if isinstance(m, Linear))

    def num_scaling_params(self):
        """Parameter counts for scaling-law analysis (banks reported separately)."""
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = self.num_value_params()
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = (self.resid_lambdas.numel() + self.x0_lambdas.numel()
                   + self.smear_gate.weight.numel() + self.smear_lambda.numel()
                   + self.backout_lambda.numel())
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def num_value_params(self):
        n = 0
        if self.move.is_move:
            n += self.move_bank.weight.numel()
        for bank in getattr(self, "value_embeds", {}).values():
            n += bank.weight.numel()
        return n

    def get_device(self):
        return self.transformer.wte.weight.device

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, scalar_lr=0.5, move_gate_lr=0.005):
        from jevelike.optim import MuonAdamW
        model_dim = self.config.n_embd

        # split transformer params: muon matrices vs value-embedding gates
        gate_ids = set()
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                gate_ids.add(id(block.attn.ve_gate.weight))
        matrix_params = [p for p in self.transformer.h.parameters() if id(p) not in gate_ids]
        gate_params = [p for p in self.transformer.h.parameters() if id(p) in gate_ids]

        value_embeds_params = []
        if self.move.is_move:
            value_embeds_params += list(self.move_bank.parameters())
        for bank in getattr(self, "value_embeds", {}).values():
            value_embeds_params += list(bank.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        assert len(list(self.parameters())) == (
            len(matrix_params) + len(gate_params) + len(embedding_params) + len(lm_head_params)
            + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(smear_params)
        ), "optimizer param bookkeeping mismatch"

        # Scale the LR for the AdamW parameters by 1/sqrt(dmodel) (tuned for 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")

        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale,
                 betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale,
                 betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5,
                 betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=gate_params, lr=move_gate_lr,
                 betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01,
                 betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr,
                 betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=smear_params, lr=0.2,
                 betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(kind='muon', params=group_params, lr=matrix_lr,
                                     momentum=0.95, ns_steps=5, beta2=0.9,
                                     weight_decay=weight_decay))

        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42,
                 kv_cache=None, embed_fn=None):
        """
        Naive autoregressive streaming inference.
        tokens: list[int]; yields generated token ids one by one.
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            idx_cond = ids if kv_cache is None else (ids if kv_cache.get_pos() == 0 else ids[:, -1:])
            logits = self.forward(idx_cond, kv_cache=kv_cache, embed_fn=embed_fn)
            logits = logits[:, -1, :]
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            yield next_ids.item()
