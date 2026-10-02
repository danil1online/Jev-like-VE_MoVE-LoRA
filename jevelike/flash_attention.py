"""
Flash Attention wrapper with graceful fallback.
Tries `kernels` (flash-attn-3) first, then PyTorch SDPA.
Ported from nanochat (karpathy/nanochat).

Layout (FA3-native, no transposes): q (B, Tq, H, Dh), k/v (B, Tk, Hkv, Dh).
window_size: (left, right) tuple, right is always 0 (causal). left = -1 for full
context, else the window in tokens.
Causality is defined relative to the END of k: query i (of Tq) sits at absolute
position (Tk - Tq) + i and may attend to keys <= that position (and within window).
"""
import torch


def _left(window_size):
    if isinstance(window_size, (tuple, list)):
        return window_size[0]
    return window_size


def flash_attn(q, k, v, window_size, is_causal=True, scale=None):
    B, Tq, H, Dh = q.shape
    Hkv = k.shape[2]
    Tk = k.shape[1]
    left = _left(window_size)

    try:
        from kernels import get_kernel
        flash_attn_3 = get_kernel("flash_attn_3")
        return flash_attn_3.flash_attn_func(q, k, v, causal=is_causal,
                                            window_size=window_size, softmax_scale=scale)
    except Exception:
        pass

    # slow path: PyTorch SDPA
    q_s = q.transpose(1, 2)
    k_s = k.transpose(1, 2)
    v_s = v.transpose(1, 2)
    enable_gqa = (H != Hkv)

    full_window = left < 0 or left >= Tk
    if is_causal and full_window and Tq == Tk:
        out = torch.nn.functional.scaled_dot_product_attention(
            q_s, k_s, v_s, is_causal=True, scale=scale, enable_gqa=enable_gqa)
        return out.transpose(1, 2)

    row = (Tk - Tq) + torch.arange(Tq, device=q.device)
    col = torch.arange(Tk, device=q.device)
    mask = col[None, :] <= row[:, None]
    if not full_window:
        mask = mask & (row[:, None] - col[None, :]) <= left
    out = torch.nn.functional.scaled_dot_product_attention(
        q_s, k_s, v_s, attn_mask=mask, scale=scale, enable_gqa=enable_gqa)
    return out.transpose(1, 2)
