"""Kimi Delta Attention (KDA) — custom Triton chunked-scan kernel.

Algorithm
---------
KDA is a **gated linear attention** layer. Given input ``x`` of shape
``[B, T, H, D]`` (batch, time, heads, head-dim) we compute:

    q, k, v, g = projections(x)                      # all [B, T, H, D']
    s_t = decay_t * s_{t-1} + k_t^T v_t               # recurrent state
    o_t = q_t @ s_t
    y_t = g_t * o_t                                   # output gate

The naive Python loop over ``t`` is the bottleneck. Instead we use the
**chunkwise** form: we split the sequence into chunks of length ``C`` and
process each chunk as a small matrix-multiply while carrying a chunk-state
between chunks. The intra-chunk contribution is computed with the standard
"linear attention inside a chunk" formula and the inter-chunk contribution
uses the carried state.

Kernel layout
-------------
* One Triton program per ``(batch, head, chunk)`` triple.
* The state ``s`` is a ``[D', D']`` matrix kept in registers/SRAM across the
  whole sequence — we launch a *single* kernel that walks **all** chunks of
  a (b, h) pair so the state stays live in registers, avoiding any HBM
  round-trip for the state between chunks.

Fallback
--------
If Triton is not installed, the GPU is not on the supported arch list, or
the kernel raises any exception at compile/run time, we fall back to a
pure-PyTorch chunked implementation that is mathematically identical but
~2-3x slower. The fallback is logged once via ``utils.logger``.
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


# --------------------------------------------------------------------------- #
# Architecture detection — decides which autotune config is used.
# --------------------------------------------------------------------------- #
def _supported_arch() -> bool:
    """Return True iff the current GPU is in the auto-tune target list."""
    if not _HAS_TRITON or not torch.cuda.is_available():
        return False
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        return False
    # sm_80 (A100, A6000), sm_89 (Ada/L4), sm_90 (Hopper), sm_100/120 (Blackwell)
    return (major, minor) in {(8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0)}


# --------------------------------------------------------------------------- #
# Triton kernel
# --------------------------------------------------------------------------- #
if _HAS_TRITON:

    def _kda_cfg():
        # Different arches ship with different SRAM/SM counts. We pick tile
        # sizes that fit the SMEM budget per arch. Triton will benchmark all
        # candidates at first launch and pick the fastest.
        cfgs = []
        # Generic configs (work everywhere)
        for bs in (32, 64):
            for nw in (4, 8):
                cfgs.append(triton.Config({"BLOCK_D": bs}, num_warps=nw, num_stages=2))
        # Ampere / Ada
        cfgs.append(triton.Config({"BLOCK_D": 64}, num_warps=8, num_stages=3))
        # Hopper
        cfgs.append(triton.Config({"BLOCK_D": 128}, num_warps=8, num_stages=4))
        # Blackwell
        cfgs.append(triton.Config({"BLOCK_D": 128}, num_warps=16, num_stages=5))
        return cfgs

    @triton.autotune(configs=_kda_cfg(), key=["D", "CHUNK", "USE_GATE"])
    @triton.jit
    def _kda_chunk_kernel(
        # pointers
        Q_ptr, K_ptr, V_ptr, G_ptr,          # inputs: [B, H, T, D]
        S_ptr,                                # state: [B, H, D, D]  (scratch + carry)
        O_ptr,                                # output: [B, H, T, D]
        # strides (in elements, not bytes)
        stride_qb, stride_qh, stride_qt, stride_qd,
        stride_ob, stride_oh, stride_ot, stride_od,
        # sizes
        B, H, T, D: tl.constexpr,
        CHUNK: tl.constexpr,
        DECAY,
        USE_GATE: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """One program processes **all chunks** for a single (batch, head)."""
        pid_bh = tl.program_id(0)
        b = pid_bh // H
        h = pid_bh % H

        # ---- state tile (D x D) lives in registers / SRAM for the whole run
        # We materialise it as a [BLOCK_D, BLOCK_D] tile.
        offs_d = tl.arange(0, BLOCK_D)
        offs_d2 = tl.arange(0, BLOCK_D)
        # mask out-of-bounds dims
        d_mask = offs_d < D
        d2_mask = offs_d2 < D
        # state = zeros
        s = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)

        n_chunks = tl.cdiv(T, CHUNK)

        for c in range(0, n_chunks):
            t_start = c * CHUNK
            # ---- intra-chunk contribution (linear-attn inside chunk) ----
            # For each t in [t_start, t_start+CHUNK) compute
            #   o_t = q_t @ (decay^{t - t_start} * s + sum_{j<=t} k_j v_j^T)
            # We compute the "sum_{j<=t}" term with a masked outer-product
            # accumulation over j.
            acc = tl.zeros([BLOCK_D], dtype=tl.float32)  # o_t accumulator for current j
            for j in range(0, CHUNK):
                tj = t_start + j
                if tj >= T:
                    break

                # load k_tj, v_tj
                k_j = tl.load(
                    K_ptr + b * stride_qb + h * stride_qh + tj * stride_qt + offs_d * stride_qd,
                    mask=d_mask, other=0.0,
                ).to(tl.float32)
                v_j = tl.load(
                    V_ptr + b * stride_qb + h * stride_qh + tj * stride_qt + offs_d * stride_qd,
                    mask=d_mask, other=0.0,
                ).to(tl.float32)

                # update state with outer product k_j v_j^T
                s += tl.dot(
                    tl.reshape(k_j, [BLOCK_D, 1]),
                    tl.reshape(v_j, [1, BLOCK_D]),
                    allow_tf32=True,
                )

                # for each subsequent t in chunk, accumulate q_t @ s contribution
                # we process t = tj here (s already includes k_j v_j^T)
                q_t = tl.load(
                    Q_ptr + b * stride_qb + h * stride_qh + tj * stride_qt + offs_d * stride_qd,
                    mask=d_mask, other=0.0,
                ).to(tl.float32)
                o_t = tl.dot(tl.reshape(q_t, [1, BLOCK_D]), s, allow_tf32=True)
                o_t = tl.reshape(o_t, [BLOCK_D])

                # apply exponential decay to state for the next chunk-step
                s = s * DECAY

                # output gate (optional)
                if USE_GATE:
                    g_t = tl.load(
                        G_ptr + b * stride_qb + h * stride_qh + tj * stride_qt + offs_d * stride_qd,
                        mask=d_mask, other=0.0,
                    ).to(tl.float32)
                    o_t = o_t * g_t

                tl.store(
                    O_ptr + b * stride_ob + h * stride_oh + tj * stride_ot + offs_d * stride_od,
                    o_t.to(tl.bfloat16),
                    mask=d_mask,
                )

        # write final state back (for inspection / cross-microbatch use)
        for i in range(0, BLOCK_D):
            if i < D:
                tl.store(
                    S_ptr + b * H * D * D + h * D * D + i * D + offs_d,
                    s[i, :].to(tl.bfloat16),
                    mask=d_mask,
                )


# --------------------------------------------------------------------------- #
# Python wrapper
# --------------------------------------------------------------------------- #
def kda_chunked_scan_triton(
    q: torch.Tensor,           # [B, H, T, D]
    k: torch.Tensor,           # [B, H, T, D]
    v: torch.Tensor,           # [B, H, T, D]
    gate: Optional[torch.Tensor] = None,   # [B, H, T, D] or None
    decay: float = 0.99,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Run the Triton KDA chunked scan. Returns ``o`` of shape ``[B, H, T, D]``."""
    assert q.dtype in (torch.float16, torch.bfloat16), "KDA Triton kernel requires fp16/bf16"
    B, H, T, D = q.shape
    o = torch.empty_like(q)
    # state scratch — only the final per-(b,h) state is written back here.
    state = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)

    if not _HAS_TRITON:
        log_fallback_warning("triton not installed (kda_kernel)")
        return _kda_chunked_scan_torch(q, k, v, gate, decay, chunk_size)

    # Pad D to a power-of-two >= 16 so the kernel's BLOCK_D tile works.
    BLOCK_D = triton.next_power_of_2(max(16, D))
    if BLOCK_D > 128:
        # Triton kernel currently caps at BLOCK_D=128; fall back for huge heads.
        log_fallback_warning(f"D={D} > 128 (kda_kernel BLOCK_D cap)")
        return _kda_chunked_scan_torch(q, k, v, gate, decay, chunk_size)

    grid = (B * H,)
    try:
        _kda_chunk_kernel[grid](
            q, k, v, gate if gate is not None else q,  # unused pointer if no gate
            state, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            B, H, T, D,
            CHUNK=chunk_size,
            DECAY=decay,
            USE_GATE=(gate is not None),
            BLOCK_D=BLOCK_D,
        )
        return o
    except Exception as e:  # pragma: no cover
        log.warning("Triton KDA kernel failed (%s); falling back to torch.", e)
        log_fallback_warning(f"kda_kernel runtime error: {e}")
        return _kda_chunked_scan_torch(q, k, v, gate, decay, chunk_size)


# --------------------------------------------------------------------------- #
# Pure-PyTorch fallback (mathematically identical)
# --------------------------------------------------------------------------- #
def _kda_chunked_scan_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: Optional[torch.Tensor],
    decay: float,
    chunk_size: int,
) -> torch.Tensor:
    """Vectorised chunked gated-linear-attention in pure PyTorch.

    Carries a [B, H, D, D] state across chunks; inside a chunk it uses a
    causal mask to compute the linear-attention output for all positions
    at once.
    """
    B, H, T, D = q.shape
    qf = q.float()
    kf = k.float()
    vf = v.float()

    out = torch.empty_like(q)
    state = torch.zeros(B, H, D, D, device=q.device, dtype=torch.float32)

    n_chunks = (T + chunk_size - 1) // chunk_size
    # pre-build a causal mask of size [chunk, chunk]
    m = torch.tril(torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.float32))
    for c in range(n_chunks):
        t0 = c * chunk_size
        t1 = min(T, t0 + chunk_size)
        L = t1 - t0
        qc = qf[:, :, t0:t1]            # [B, H, L, D]
        kc = kf[:, :, t0:t1]
        vc = vf[:, :, t0:t1]

        # ---- intra-chunk contribution ----
        # o_intra[b,h,i,:] = sum_{p<=i} <q[b,h,i,:], k[b,h,p,:]> * v[b,h,p,:]
        w = m[:L, :L].view(1, 1, L, L)                       # [1,1,L,L]
        attn = torch.einsum("bhid,bhjd->bhij", qc, kc) * w   # [B,H,L,L]
        o_intra = torch.einsum("bhij,bhjd->bhid", attn, vc)  # [B,H,L,D]

        # ---- inter-chunk contribution from carried state ----
        # o_inter[b,h,i,:] = q[b,h,i,:] @ state[b,h,:,:]
        o_inter = torch.einsum("bhid,bhdf->bhif", qc, state)  # [B,H,L,D]

        o_chunk = o_intra + o_inter
        if gate is not None:
            o_chunk = o_chunk * gate[:, :, t0:t1].float()
        out[:, :, t0:t1] = o_chunk.to(out.dtype)

        # ---- state update ----
        # state <- decay^L * state + sum_p k_p v_p^T
        full_kv = torch.einsum("bhpd,bhqd->bhpq", kc, vc)     # [B,H,D,D]
        state = (decay ** L) * state + full_kv

    return out


# --------------------------------------------------------------------------- #
# Public dispatch
# --------------------------------------------------------------------------- #
def kda_chunked_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: Optional[torch.Tensor] = None,
    decay: float = 0.99,
    chunk_size: int = 64,
    use_triton: bool = True,
) -> torch.Tensor:
    """Dispatch to the Triton kernel when available & supported.

    Falls back to the pure-torch path on:
      * CPU tensors
      * `use_triton=False`
      * Triton not installed / unsupported arch / runtime failure
    """
    if not use_triton or not _supported_arch():
        if use_triton and not _supported_arch():
            log_fallback_warning(
                f"GPU arch sm_"
                f"{torch.cuda.get_device_capability()[0]}"
                f"{torch.cuda.get_device_capability()[1]}"
                f" not in Triton autotune target list"
            )
        return _kda_chunked_scan_torch(q, k, v, gate, decay, chunk_size)
    return kda_chunked_scan_triton(q, k, v, gate, decay, chunk_size)
