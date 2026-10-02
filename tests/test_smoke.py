"""
End-to-end smoke tests on CPU with tiny models: forward/backward, optimizer,
checkpoint roundtrip, model rebuild, dataloader, and bpb evaluation.
"""
import math
import os
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from jevelike.checkpoint import (
    save_checkpoint, load_checkpoint, build_model, find_last_step,
    save_adapter, load_adapter, find_last_adapter,
)
from jevelike.configs import MoveConfig
from jevelike.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit
from jevelike.loss_eval import evaluate_bpb
from jevelike.move import resolve_move
from jevelike.gpt import GPT
from conftest import build_tiny, tiny_model_config, SMALL_CORPUS


def test_forward_backward_generate(model_config):
    model = build_tiny(model_config, mode="move", num_slots=2)
    B, T = 2, 16
    idx = torch.randint(0, model_config.vocab_size, (B, T))
    targets = torch.randint(0, model_config.vocab_size, (B, T))
    loss = model(idx, targets)
    assert loss.ndim == 0 and torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    n_grads = sum(1 for p in model.parameters() if p.grad is not None)
    assert n_grads > 0
    logits = model(idx)
    assert logits.shape == (B, T, model_config.vocab_size)
    # loss_reduction='none' returns per-position loss (flat, nanochat-style)
    loss2d = model(idx, targets, loss_reduction="none")
    assert loss2d.shape == (B * T,)
    # consistent with the mean reduction
    torch.testing.assert_close(loss2d.mean(), loss)
    # generate: list[int] in, yields token ids one by one
    prompt = idx[0, :T // 2].tolist()
    out = list(model.generate(prompt, max_tokens=4, seed=7))
    assert len(out) == 4
    assert all(0 <= t < model_config.vocab_size for t in out)
    # greedy (temperature=0) is deterministic
    g1 = list(model.generate(prompt, max_tokens=3, temperature=0.0))
    g2 = list(model.generate(prompt, max_tokens=3, temperature=0.0))
    assert g1 == g2


def test_lave_forward(model_config):
    model = build_tiny(model_config, mode="lave", lave_layers="alt")
    idx = torch.randint(0, model_config.vocab_size, (2, 16))
    loss = model(idx, torch.randint(0, model_config.vocab_size, (2, 16)))
    assert torch.isfinite(loss)


def test_optimizer_steps(model_config):
    model = build_tiny(model_config, mode="move", num_slots=2)
    optimizer = model.setup_optimizer()
    idx = torch.randint(0, model_config.vocab_size, (2, 16))
    targets = torch.randint(0, model_config.vocab_size, (2, 16))
    x0_before = model.x0_lambdas.detach().clone()
    for _ in range(3):
        optimizer.zero_grad()
        loss = model(idx, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    # scalar lr=0.5 -> x0_lambdas must have moved
    assert not torch.equal(model.x0_lambdas.detach(), x0_before)
    # optimizer state bookkeeping survived
    assert len(optimizer.state) >= 1


def test_checkpoint_roundtrip(model_config, tmp_path):
    model = build_tiny(model_config, mode="move", num_slots=2)
    optimizer = model.setup_optimizer()
    ckpt_dir = str(tmp_path / "d1" / "smoke")
    step = 42
    save_checkpoint(ckpt_dir, step, model.state_dict(), optimizer.state_dict(),
                    {"model_config": model_config.as_dict(),
                     "move_config": MoveConfig(mode="move", num_slots=2).as_dict()})
    assert find_last_step(ckpt_dir) == step
    model_data, optim_data, meta = load_checkpoint(ckpt_dir, step, "cpu", load_optimizer=True)
    assert set(model_data.keys()) == set(model.state_dict().keys())
    for k, v in model.state_dict().items():
        torch.testing.assert_close(model_data[k], v)
    assert meta["model_config"]["n_layer"] == model_config.n_layer
    assert len(optim_data) == len(optimizer.state_dict())


def test_build_model_from_checkpoint(model_config, small_tok, tmp_path):
    vocab = small_tok.get_vocab_size()
    cfg = tiny_model_config(vocab_size=vocab)
    move_cfg = MoveConfig(mode="move", num_slots=1)
    move = resolve_move(move_cfg, cfg)
    torch.manual_seed(0)
    model = GPT(cfg, move=move)
    model.init_weights()
    ckpt_dir = str(tmp_path / "d1" / "built")
    save_checkpoint(ckpt_dir, 7, model.state_dict(), None,
                    {"model_config": cfg.as_dict(), "move_config": move_cfg.as_dict()})
    tok_dir = str(tmp_path / "tokenizer")
    small_tok.save(tok_dir)
    model2, tok, meta = build_model(ckpt_dir, 7, torch.device("cpu"), "eval", tokenizer_dir=tok_dir)
    assert tok.get_vocab_size() == vocab
    for k, v in model.state_dict().items():
        torch.testing.assert_close(model2.state_dict()[k], v.float(), atol=1e-6, rtol=1e-5)
    assert model2.training is False
    idx = torch.randint(0, vocab, (1, 8))
    with torch.no_grad():
        assert model2(idx).shape == (1, 8, vocab)


def test_adapter_checkpoint_roundtrip(tmp_path):
    state = {"lora_A": torch.randn(2, 8), "extra": torch.randn(3)}
    meta = {"base_checkpoint": "checkpoints/d12_move/model_000250.pt", "lora_config": {"rank": 2}}
    path = str(tmp_path / "jev_checkpoints" / "d12_move" / "adapter_000123.pt")
    save_adapter(path, 123, state, meta)
    loaded, step, lmeta = load_adapter(path, "cpu")
    assert step == 123
    torch.testing.assert_close(loaded["lora_A"], state["lora_A"])
    assert lmeta["lora_config"]["rank"] == 2
    assert find_last_adapter(os.path.dirname(path)) == 123


def _make_data_dir(data_dir, n_docs=60, row_group_size=10):
    os.makedirs(data_dir, exist_ok=True)
    for shard in range(2):  # train shard + val shard (last file is val)
        texts = [
            (f"это документ номер {i} с несколькими словами и цифрами {i % 10} " * 4)
            for i in range(shard * 30, shard * 30 + n_docs // 2)
        ]
        pq.write_table(pa.table({"text": texts}),
                       os.path.join(data_dir, f"part-{shard:03d}.parquet"),
                       row_group_size=row_group_size)


def test_dataloader_bos_bestfit(small_tok, tmp_path):
    data_dir = str(tmp_path / "data")
    _make_data_dir(data_dir)
    B, T = 2, 16
    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        small_tok, B, T, "train", data_dir,
        tokenizer_threads=1, tokenizer_batch_size=32, device="cpu", buffer_size=16,
    )
    x, y, state = next(iter(loader))
    assert x.shape == (B, T) and y.shape == (B, T)
    bos = small_tok.get_bos_token_id()
    assert (x[:, 0] == bos).all()
    # targets are inputs shifted by one (last target column comes from the T+1-th buffer cell)
    torch.testing.assert_close(y[:, :-1], x[:, 1:])
    assert set(x.flatten().tolist()) <= set(range(small_tok.get_vocab_size()))
    assert state["epoch"] >= 1
    # val split uses only the last parquet file
    loader_val = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        small_tok, B, T, "val", data_dir,
        tokenizer_threads=1, tokenizer_batch_size=32, device="cpu", buffer_size=16,
    )
    xv, yv, _ = next(iter(loader_val))
    assert xv.shape == (B, T)


def test_evaluate_bpb(model_config, small_tok):
    model = build_tiny(tiny_model_config(vocab_size=small_tok.get_vocab_size()), mode="off")
    torch.manual_seed(0)
    batches = iter([
        (torch.randint(0, small_tok.get_vocab_size(), (2, 16)),
         torch.randint(0, small_tok.get_vocab_size(), (2, 16)))
        for _ in range(4)
    ])
    token_bytes = torch.randint(1, 7, (small_tok.get_vocab_size(),), dtype=torch.int32)
    bpb = evaluate_bpb(model, batches, steps=3, token_bytes=token_bytes)
    assert math.isfinite(bpb) and bpb > 0
    # all-zero token_bytes -> no bytes counted -> inf
    batches2 = iter([
        (torch.randint(0, small_tok.get_vocab_size(), (2, 16)),
         torch.randint(0, small_tok.get_vocab_size(), (2, 16)))
        for _ in range(2)
    ])
    assert evaluate_bpb(model, batches2, steps=1,
                        token_bytes=torch.zeros_like(token_bytes)) == float("inf")
