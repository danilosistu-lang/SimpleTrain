"""Fused Triton kernels: SiTUGLU, RMSNorm+Residual, fused Cross-Entropy.

Each function exposes a `*_triton` entry point and a `*_torch` fallback.
The public dispatchers pick Triton when the kernel + arch are supported,
otherwise fall back to PyTorch (and log the reason once).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:                                       # pragma: no cover
    _HAS_TRITON = False

from utils.logger import get_logger, log_fallback_warning

log = get_logger()


def _supported_arch() -> bool:
    if not _HAS_TRITON or not torch.cuda.is_available():
        return False
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        return False
    return (major, minor) in {(8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0)}


# =========================================================================== #
# 1) Fused SiTUGLU / SwiGLU
# =========================================================================== #
# Computes:
#     gate   = x @ W_gate.T
#     up     = x @ W_up.T
#     act    = silu(gate)            # SiTUGLU uses SiLU; SwiGLU uses Swish (same)
#     pre    = act * up
#     out    = pre @ W_down.T
#
# We fuse the *elementwise* part (silu * up) into a single kernel — the matmuls
# themselves are handled by cuBLAS/torch (they already saturate the tensor
# cores). This removes one global-memory round-trip and the SiLU / elementwise
# mul dispatch overhead.
# =========================================================================== #

if _HAS_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_N": 128}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_N": 128}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_N": 256}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_N": 256}, num_warps=16, num_stages=4),
        ],
        key=["N", "H"],
    )
    @triton.jit
    def _silu_mul_kernel(
        A_ptr, B_ptr, OUT_ptr,
        N, H: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        # one program per (row-block, hidden-col-block)
        pid_n = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        a = tl.load(A_ptr + offs_n * H + pid_h, mask=n_mask, other=0.0).to(tl.float32)
        b = tl.load(B_ptr + offs_n * H + pid_h, mask=n_mask, other=0.0).to(tl.float32)
        # silu(a) * b = a * sigmoid(a) * b
        out = a * tl.sigmoid(a) * b
        tl.store(OUT_ptr + offs_n * H + pid_h, out.to(tl.bfloat16), mask=n_mask)


def silu_mul_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Fused SiLU(a) * b. Inputs/outputs are [N, H] bf16/fp16."""
    assert a.shape == b.shape
    N, H = a.shape
    out = torch.empty_like(a)
    BLOCK_N = 128
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), H)
    _silu_mul_kernel[grid](a, b, out, N, H, BLOCK_N=BLOCK_N)
    return out


def silu_mul_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (F.silu(a.float()) * b.float()).to(a.dtype)


def fused_silu_mul(a: torch.Tensor, b: torch.Tensor, *, use_triton: bool = True) -> torch.Tensor:
    if use_triton and _supported_arch():
        try:
            return silu_mul_triton(a, b)
        except Exception as e:                           # pragma: no cover
            log_fallback_warning(f"silu_mul runtime: {e}")
    elif use_triton:
        log_fallback_warning("silu_mul: triton unavailable or unsupported arch")
    return silu_mul_torch(a, b)


# =========================================================================== #
# 2) Fused RMSNorm + Residual Add
# =========================================================================== #
# Computes, for each row x of [N, D]:
#     rms = sqrt(mean(x^2) + eps)
#     y   = x / rms * weight + residual
# =========================================================================== #

if _HAS_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_D": 512}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_D": 1024}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_D": 2048}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_D": 2048}, num_warps=16, num_stages=4),
            triton.Config({"BLOCK_D": 4096}, num_warps=16, num_stages=4),
        ],
        key=["D"],
    )
    @triton.jit
    def _rmsnorm_residual_kernel(
        X_ptr, R_ptr, W_ptr, OUT_ptr,
        D, EPS,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_D)
        d_mask = offs < D
        x = tl.load(X_ptr + row * D + offs, mask=d_mask, other=0.0).to(tl.float32)
        r = tl.load(R_ptr + row * D + offs, mask=d_mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + offs, mask=d_mask, other=0.0).to(tl.float32)
        # mean(x^2) = sum(x^2) / D
        ss = tl.sum(x * x, axis=0) / D
        rms = 1.0 / tl.sqrt(ss + EPS)
        y = x * rms * w + r
        tl.store(OUT_ptr + row * D + offs, y.to(tl.bfloat16), mask=d_mask)


def rmsnorm_residual_triton(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """x: [N, D] bf16. residual: [N, D]. weight: [D]. Returns [N, D] bf16."""
    N, D = x.shape
    out = torch.empty_like(x)
    grid = (N,)
    _rmsnorm_residual_kernel[grid](x, residual, weight, out, D, eps)
    return out


def rmsnorm_residual_torch(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    xf = x.float()
    ms = xf.pow(2).mean(dim=-1, keepdim=True)
    rms_inv = torch.rsqrt(ms + eps)
    y = xf * rms_inv * weight.float() + residual.float()
    return y.to(x.dtype)


def fused_rmsnorm_residual(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float,
    *, use_triton: bool = True,
) -> torch.Tensor:
    if use_triton and _supported_arch():
        try:
            return rmsnorm_residual_triton(x, residual, weight, eps)
        except Exception as e:                           # pragma: no cover
            log_fallback_warning(f"rmsnorm_residual runtime: {e}")
    elif use_triton:
        log_fallback_warning("rmsnorm_residual: triton unavailable or unsupported arch")
    return rmsnorm_residual_torch(x, residual, weight, eps)


# =========================================================================== #
# 3) Fused Cross-Entropy with online softmax
# =========================================================================== #
# Computes:
#     logits    : [N, V]
#     labels    : [N]
#     loss      = -log_softmax(logits)[range(N), labels].mean()
#     softmax_out : [N, V]   (for backward / KL distillation)
#
# The fused kernel uses the numerically-stable **online softmax** trick:
# iterate over the V dimension in tiles, keep a running max + running sum,
# then normalise in a second pass. Avoids materialising a full FP32 softmax
# in HBM.
# =========================================================================== #

if _HAS_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_V": 1024}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_V": 2048}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_V": 4096}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_V": 8192}, num_warps=16, num_stages=4),
        ],
        key=["V"],
    )
    @triton.jit
    def _ce_forward_kernel(
        LOGITS_ptr, LABELS_ptr,
        LOSS_ptr, LSE_ptr,                # LSE = log-sum-exp, [N]
        N, V,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= N:
            return
        # pass 1: find max + sum(exp(x - max))
        m_val = -float("inf")
        s_val = 0.0
        for v0 in range(0, V, BLOCK_V):
            offs = v0 + tl.arange(0, BLOCK_V)
            mask = offs < V
            x = tl.load(LOGITS_ptr + row * V + offs, mask=mask, other=-float("inf")).to(tl.float32)
            m_block = tl.max(x, axis=0)
            m_new = tl.maximum(m_val, m_block)
            # re-scale running sum
            alpha = tl.exp(m_val - m_new)
            beta = tl.exp(m_block - m_new)
            s_val = s_val * alpha + tl.sum(beta * tl.where(mask, 1.0, 0.0), axis=0)
            m_val = m_new

        lse = m_val + tl.log(s_val)

        # pass 2: compute log_softmax at the label index
        label = tl.load(LABELS_ptr + row).to(tl.int32)
        # we need logits[label]; load just that one
        x_label = tl.load(LOGITS_ptr + row * V + label).to(tl.float32)
        loss_val = lse - x_label

        tl.store(LOSS_ptr + row, loss_val)
        tl.store(LSE_ptr + row, lse)


    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_V": 1024}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_V": 2048}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_V": 4096}, num_warps=8, num_stages=3),
        ],
        key=["V"],
    )
    @triton.jit
    def _ce_softmax_kernel(
        LOGITS_ptr, LSE_ptr, OUT_ptr,
        N, V,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= N:
            return
        lse = tl.load(LSE_ptr + row).to(tl.float32)
        for v0 in range(0, V, BLOCK_V):
            offs = v0 + tl.arange(0, BLOCK_V)
            mask = offs < V
            x = tl.load(LOGITS_ptr + row * V + offs, mask=mask, other=-float("inf")).to(tl.float32)
            p = tl.exp(x - lse)
            tl.store(OUT_ptr + row * V + offs, p.to(tl.bfloat16), mask=mask)


def cross_entropy_triton(
    logits: torch.Tensor,     # [N, V]
    labels: torch.Tensor,     # [N]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (mean_loss, softmax_probs[N, V])."""
    N, V = logits.shape
    loss = torch.empty(N, device=logits.device, dtype=torch.float32)
    lse = torch.empty(N, device=logits.device, dtype=torch.float32)
    softmax = torch.empty_like(logits)

    grid_loss = (N,)
    _ce_forward_kernel[grid_loss](logits, labels, loss, lse, N, V)
    _ce_softmax_kernel[grid_loss](logits, lse, softmax, N, V)
    return loss.mean(), softmax


def cross_entropy_torch(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    lp = F.log_softmax(logits.float(), dim=-1)
    nll = -lp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    softmax = lp.exp().to(logits.dtype)
    return nll.mean(), softmax


def fused_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, *, use_triton: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if use_triton and _supported_arch():
        try:
            return cross_entropy_triton(logits, labels)
        except Exception as e:                           # pragma: no cover
            log_fallback_warning(f"cross_entropy runtime: {e}")
    elif use_triton:
        log_fallback_warning("cross_entropy: triton unavailable or unsupported arch")
    return cross_entropy_torch(logits, labels)
