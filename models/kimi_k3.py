"""Kimi K3 transformer implementation.

Layer type routing (3:1 KDA : Gated MLA):
    * Even/odd layers come from `config.build_layer_layout`.
    * First and last layers are always Gated MLA.
    * Each layer is wrapped in an `AttnRes` residual block (a 2-branch
      residual: identity skip + a small alpha-scaled attn skip), which is
      the "Attention Residual" structure from the Kimi K3 paper — it keeps
      gradient flow stable across very deep stacks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import KimiK3Config, build_layer_layout
from kernels.kda_kernel import kda_chunked_scan
from kernels.fused_ops import (
    fused_silu_mul,
    fused_rmsnorm_residual,
    fused_cross_entropy,
)
from utils.logger import get_logger

log = get_logger()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _dt(cfg: KimiK3Config) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }[cfg.dtype]


class RMSNorm(nn.Module):
    """Plain RMSNorm (no residual). Used for the final LN before the LM head."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        ms = xf.pow(2).mean(dim=-1, keepdim=True)
        return (xf * torch.rsqrt(ms + self.eps) * self.weight.float()).to(x.dtype)


# --------------------------------------------------------------------------- #
# Kimi Delta Attention (KDA) block
# --------------------------------------------------------------------------- #
class KimiDeltaAttention(nn.Module):
    """Gated linear attention with chunkwise recurrent state updates.

    Projections:
        q, k, v, g  = 4 Linear(d_model, n_heads * d_head) layers
    State update (per head):
        s_t  = decay * s_{t-1} + k_t v_t^T
        o_t  = q_t @ s_t
        y_t  = g_t * o_t            # output gate
    """

    def __init__(self, cfg: KimiK3Config):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.inner = cfg.n_heads * cfg.d_head
        self.chunk = cfg.kda_chunk_size
        self.decay = cfg.kda_decay_base

        self.q_proj = nn.Linear(cfg.d_model, self.inner, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.inner, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.inner, bias=False)
        self.g_proj = nn.Linear(cfg.d_model, self.inner, bias=False)
        self.o_proj = nn.Linear(self.inner, cfg.d_model, bias=False)

        # learnable decay per-head (small init around kda_decay_base)
        log_decay_init = math.log(cfg.kda_decay_base)
        self.log_decay = nn.Parameter(torch.full((cfg.n_heads,), log_decay_init))

        # initialise gates to near-zero so the layer behaves close to identity
        nn.init.zeros_(self.g_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] -> [B, T, D]."""
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head

        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)  # [B, H, T, Dh]
        k = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)
        g = self.g_proj(x).view(B, T, H, Dh).transpose(1, 2)

        # per-head decay (clamped to (0, 1))
        decay = torch.exp(self.log_decay).clamp(min=1e-3, max=0.999).detach()
        # use the *mean* decay across heads as a scalar for the chunked kernel;
        # the learnable per-head decay is re-applied as a multiplicative gate
        # below to retain head-wise flexibility.
        decay_scalar = float(decay.mean().item())

        # run kernel / fallback
        o = kda_chunked_scan(
            q.to(self._dtype()), k.to(self._dtype()), v.to(self._dtype()),
            gate=g.to(self._dtype()),
            decay=decay_scalar,
            chunk_size=self.chunk,
            use_triton=self.cfg.use_triton,
        )                                                          # [B, H, T, Dh]

        # apply per-head decay as a scaling gate (extra expressivity)
        head_scale = torch.exp(self.log_decay).clamp(min=1e-3, max=0.999)
        o = o * head_scale.view(1, H, 1, 1).to(o.dtype)

        o = o.transpose(1, 2).contiguous().view(B, T, H * Dh)
        return self.o_proj(o)

    def _dtype(self) -> torch.dtype:
        return _dt(self.cfg)


# --------------------------------------------------------------------------- #
# Gated Multi-head Latent Attention (Gated MLA)
# --------------------------------------------------------------------------- #
class GatedMLA(nn.Module):
    """Multi-head latent attention with KV compression + output gate.

    Adapted from DeepSeek-V2's MLA: K/V are first projected into a small
    latent vector of dim `mla_kv_compress`, then re-expanded into the
    per-head K/V tensors. Q is similarly compressed. An output gate
    (sigmoid of a small Linear) multiplies the attention output.

    We use the standard scaled-dot-product attention (PyTorch native or
    flash-attention-2 if available) for the inner softmax attention.
    """

    def __init__(self, cfg: KimiK3Config):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.inner = cfg.n_heads * cfg.d_head
        self.dc = cfg.mla_q_compress
        self.dkv = cfg.mla_kv_compress
        self.n_kv_heads = cfg.mla_n_kv_heads

        # Q compression (down + up)
        self.q_down = nn.Linear(cfg.d_model, self.dc, bias=False)
        self.q_up   = nn.Linear(self.dc, self.inner, bias=False)

        # KV compression (down + up)
        self.kv_down = nn.Linear(cfg.d_model, self.dkv, bias=False)
        self.k_up    = nn.Linear(self.dkv, self.n_kv_heads * cfg.d_head, bias=False)
        self.v_up    = nn.Linear(self.dkv, self.n_kv_heads * cfg.d_head, bias=False)

        # output gate (sigmoid)
        self.gate = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        nn.init.zeros_(self.gate.weight)

        self.o_proj = nn.Linear(self.inner, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head
        Hkv = self.n_kv_heads

        q = self.q_up(self.q_down(x)).view(B, T, H, Dh)
        k = self.k_up(self.kv_down(x)).view(B, T, Hkv, Dh)
        v = self.v_up(self.kv_down(x)).view(B, T, Hkv, Dh)

        # repeat KV heads to match Q heads (multi-query style)
        if Hkv != H:
            rep = H // Hkv
            k = k.repeat_interleave(rep, dim=2)
            v = v.repeat_interleave(rep, dim=2)

        # scaled-dot-product attention with causal mask
        q = q.transpose(1, 2)                                  # [B, H, T, Dh]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
        )                                                      # [B, H, T, Dh]
        attn = attn.transpose(1, 2).contiguous().view(B, T, H * Dh)

        # output gate (sigmoid)
        gate = torch.sigmoid(self.gate(x))
        out = attn * gate
        return self.o_proj(out)


# --------------------------------------------------------------------------- #
# SiTUGLU / SwiGLU FFN
# --------------------------------------------------------------------------- #
class SiTUGLUFFN(nn.Module):
    """SiLU-gated linear unit FFN.

        out = (silu(x @ W_gate) * (x @ W_up)) @ W_down

    The elementwise `silu(gate) * up` is dispatched to the fused Triton
    kernel when available.
    """

    def __init__(self, cfg: KimiK3Config):
        super().__init__()
        self.cfg = cfg
        hidden = int(cfg.ffn_multiple * cfg.d_model)
        # round hidden to a multiple of 64 for tensor-core friendliness
        hidden = ((hidden + 63) // 64) * 64
        self.hidden = hidden
        self.w_gate = nn.Linear(cfg.d_model, hidden, bias=cfg.ffn_bias)
        self.w_up   = nn.Linear(cfg.d_model, hidden, bias=cfg.ffn_bias)
        self.w_down = nn.Linear(hidden, cfg.d_model, bias=cfg.ffn_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.w_gate(x)
        u = self.w_up(x)
        h = fused_silu_mul(g, u, use_triton=self.cfg.use_triton) if isinstance(self, SiTUGLUFFN) else (F.silu(g) * u)
        return self.w_down(h)


# --------------------------------------------------------------------------- #
# AttnRes block — 2-branch residual
# --------------------------------------------------------------------------- #
class AttnResBlock(nn.Module):
    """Wraps a (attn, ffn) pair with an Attention Residual structure.

    Forward path:
        h = x + alpha * attn(norm1(x))           # attn residual branch
        h = h     + ffn (norm2(h))               # standard FFN residual
    """

    def __init__(self, cfg: KimiK3Config, attn: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.attn = attn
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SiTUGLUFFN(cfg)
        self.alpha = cfg.attn_res_alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm + AttnRes: residual has two branches.
        a = self.attn(self.norm1(x))
        h = x + self.alpha * a
        f = self.ffn(self.norm2(h))
        return h + f


# --------------------------------------------------------------------------- #
# Full Kimi K3 model
# --------------------------------------------------------------------------- #
class KimiK3(nn.Module):
    def __init__(self, cfg: KimiK3Config):
        super().__init__()
        self.cfg = cfg
        self.layout: List[str] = build_layer_layout(cfg.n_layers, kda_per_mla=3)

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.context_length, cfg.d_model)

        blocks: List[AttnResBlock] = []
        for kind in self.layout:
            attn = KimiDeltaAttention(cfg) if kind == "kda" else GatedMLA(cfg)
            blocks.append(AttnResBlock(cfg, attn))
        self.blocks = nn.ModuleList(blocks)

        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        # tie LM head with embedding
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        self._init_weights()

    # ----- weight init -----
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        # scaled init for output projections
        for blk in self.blocks:
            for p in [blk.attn.o_proj.weight, blk.ffn.w_down.weight]:
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.cfg.n_layers))

    # ----- forward -----
    def forward(
        self,
        input_ids: torch.Tensor,                   # [B, T]
        labels: Optional[torch.Tensor] = None,     # [B, T]
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        B, T = input_ids.shape
        device = input_ids.device

        x = self.embed(input_ids) + self.pos_embed(torch.arange(T, device=device))[None]

        for blk in self.blocks:
            x = blk(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)                                   # [B, T, V]

        if labels is None:
            return None, logits

        # fused cross-entropy over the flattened (B*T, V) tensor
        N = B * T
        flat_logits = logits.view(N, -1)
        flat_labels = labels.view(N)
        loss, _ = fused_cross_entropy(flat_logits, flat_labels, use_triton=self.cfg.use_triton)
        return loss, logits

    # ----- introspection -----
    def num_active_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def layer_layout_summary(self) -> str:
        n_kda = sum(1 for k in self.layout if k == "kda")
        n_mla = sum(1 for k in self.layout if k == "mla")
        return f"KDA:{n_kda}  MLA:{n_mla}  (ratio {n_kda}:{n_mla})"


# --------------------------------------------------------------------------- #
# Public factory
# --------------------------------------------------------------------------- #
def build_model(cfg: KimiK3Config, device: Optional[torch.device] = None) -> KimiK3:
    model = KimiK3(cfg)
    if device is not None:
        model = model.to(device)
    # cast to target dtype
    dt = _dt(cfg)
    if dt != torch.float32:
        model = model.to(dt)
    log.info(
        "Kimi K3 [%s]: %s — %s — %.2f M active params",
        cfg.variant,
        model.layer_layout_summary(),
        cfg.dtype,
        model.num_active_parameters() / 1e6,
    )
    return model
