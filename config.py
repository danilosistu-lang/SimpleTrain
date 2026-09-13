"""Variant configurations for the Kimi K3 transformer.

The Kimi K3 hybrid stack alternates layers in a 3:1 ratio:
    3x Kimi Delta Attention (KDA)  ->  1x Gated Multi-head Latent Attention (MLA)

For an `n_layers` block count that is a multiple of 4 we get exactly
`3/4 * n_layers` KDA blocks and `1/4 * n_layers` Gated MLA blocks. We also
reserve the very first block as a Gated MLA layer (so the network opens with
global attention) and the very last block as a Gated MLA layer (so the
final hidden state mixes globally before the LM head). This is a common
arrangement in hybrid linear-attention LLMs (e.g. Jamba, Kimi-K2).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Tuple, List


# --------------------------------------------------------------------------- #
# Hyper-parameter container
# --------------------------------------------------------------------------- #
@dataclass
class KimiK3Config:
    """Full configuration for a single Kimi K3 variant."""

    variant: str
    vocab_size: int = 50304            # rounded-up GPT-2 vocab (power of 2)
    context_length: int = 2048

    # --- core dimensions ---
    d_model: int = 2048
    n_layers: int = 24
    n_heads: int = 16
    d_head: int = 128

    # --- KDA-specific ---
    kda_chunk_size: int = 64           # chunk length for chunkwise recurrence
    kda_state_dim: int = 128           # recurrent state dim per head
    kda_decay_base: float = 0.99       # exponential decay base for the gate

    # --- Gated MLA-specific ---
    mla_kv_compress: int = 512         # latent KV dim
    mla_q_compress: int = 512          # latent Q dim (<= d_model)
    mla_n_kv_heads: int = 4            # number of KV heads (multi-query)

    # --- FFN (SiTUGLU / SwiGLU) ---
    ffn_multiple: float = 4.0          # hidden = multiple * d_model (rounded)
    ffn_bias: bool = False

    # --- residual / norm ---
    attn_res_alpha: float = 0.1        # AttnRes scaling for the attn residual
    norm_eps: float = 1e-6

    # --- optimisation defaults (overridable from CLI) ---
    batch_size: int = 16
    grad_accum: int = 1
    num_steps: int = 10000
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 200
    grad_clip: float = 1.0

    # --- kernel / runtime ---
    use_triton: bool = True
    dtype: str = "bfloat16"            # "bfloat16" | "float16" | "float32"

    # --- dropout (kept tiny; mostly for ablations) ---
    embed_dropout: float = 0.0
    attn_dropout: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def estimated_active_params(self) -> int:
        """Rough active-parameter count (excluding embeddings/head sharing).

        Counts: embed (tied with LM head) + n_layers * (attn + ffn).
        KDA: q/k/v/g/o projections ≈ 5 * d^2.
        Gated MLA: q_down+up + kv_down + k_up + v_up + o_proj + gate.
        FFN (SwiGLU/SiTUGLU): 3 * d * hidden.
        """
        d, L = self.d_model, self.n_layers
        V = self.vocab_size
        # KDA: q, k, v, g, o  ->  5 * d^2
        kda_params = 5 * d * d
        # Gated MLA (latent compression)
        mla_params = (
            2 * d * self.mla_q_compress                 # q_down + q_up
            + d * self.mla_kv_compress                  # kv_down (shared)
            + self.mla_kv_compress * self.mla_n_kv_heads * self.d_head   # k_up
            + self.mla_kv_compress * self.mla_n_kv_heads * self.d_head   # v_up
            + self.n_heads * self.d_head * d            # o_proj
            + d * d                                     # gate
        )
        h = int(self.ffn_multiple * d)
        h = ((h + 63) // 64) * 64                       # match the rounding in SiTUGLUFFN
        ffn_params = 3 * d * h
        per_layer = kda_params + mla_params + ffn_params
        # Embed and LM head are tied -> count once.
        total = V * d + L * per_layer
        return int(total)


# --------------------------------------------------------------------------- #
# Variant table
# --------------------------------------------------------------------------- #
VARIANTS: Dict[str, KimiK3Config] = {
    "100m": KimiK3Config(
        variant="100m",
        d_model=512,
        n_layers=10,
        n_heads=8,
        d_head=64,
        kda_state_dim=64,
        mla_kv_compress=192,
        mla_q_compress=192,
        mla_n_kv_heads=2,
        ffn_multiple=4.0,
        context_length=2048,
    ),
    "1b": KimiK3Config(
        variant="1b",
        d_model=1280,
        n_layers=24,
        n_heads=10,
        d_head=128,
        kda_state_dim=128,
        mla_kv_compress=384,
        mla_q_compress=384,
        mla_n_kv_heads=4,
        ffn_multiple=4.0,
        context_length=2048,
    ),
    "3b": KimiK3Config(
        variant="3b",
        d_model=2048,
        n_layers=28,
        n_heads=16,
        d_head=128,
        kda_state_dim=128,
        mla_kv_compress=512,
        mla_q_compress=512,
        mla_n_kv_heads=8,
        ffn_multiple=4.0,
        context_length=4096,
    ),
    "7b": KimiK3Config(
        variant="7b",
        d_model=3072,
        n_layers=36,
        n_heads=24,
        d_head=128,
        kda_state_dim=192,
        mla_kv_compress=768,
        mla_q_compress=768,
        mla_n_kv_heads=12,
        ffn_multiple=4.0,
        context_length=4096,
    ),
    "10b": KimiK3Config(
        variant="10b",
        d_model=3584,
        n_layers=40,
        n_heads=28,
        d_head=128,
        kda_state_dim=192,
        mla_kv_compress=896,
        mla_q_compress=896,
        mla_n_kv_heads=16,
        ffn_multiple=4.0,
        context_length=4096,
    ),
}


# --------------------------------------------------------------------------- #
# Layer layout
# --------------------------------------------------------------------------- #
def build_layer_layout(n_layers: int, kda_per_mla: int = 3) -> List[str]:
    """Return per-layer types as a list of "kda" / "mla".

    Layout policy:
      * First and last layers are `mla` (global context at boundaries).
      * The interior layers follow the 3:1 KDA:MLA rhythm.
    """
    if n_layers < 2:
        return ["mla"] * n_layers
    layout: List[str] = []
    cycle = ["kda"] * kda_per_mla + ["mla"]
    idx = 0
    while len(layout) < n_layers:
        kind = cycle[idx % len(cycle)]
        layout.append(kind)
        idx += 1
    # Force the first and last block to be MLA so the network opens and
    # closes with global attention.
    layout[0] = "mla"
    layout[-1] = "mla"
    return layout


def get_config(variant: str, **overrides: Any) -> KimiK3Config:
    """Look up a variant config and apply CLI overrides."""
    if variant not in VARIANTS:
        raise ValueError(
            f"Unknown variant '{variant}'. "
            f"Valid choices: {list(VARIANTS.keys())}"
        )
    cfg = VARIANTS[variant]
    # Make a fresh copy so we never mutate the global table.
    cfg = KimiK3Config(**{**cfg.to_dict(), **overrides})
    return cfg


# --------------------------------------------------------------------------- #
# CLI -> config overrides
# --------------------------------------------------------------------------- #
def config_from_cli(
    variant: str,
    batch_size: int,
    grad_accum: int,
    num_steps: int,
    learning_rate: float,
    context_length: int,
    use_triton: bool,
    dtype: str = "bfloat16",
) -> KimiK3Config:
    cfg = get_config(variant)
    cfg.batch_size = batch_size
    cfg.grad_accum = grad_accum
    cfg.num_steps = num_steps
    cfg.learning_rate = learning_rate
    cfg.context_length = context_length
    cfg.use_triton = use_triton
    cfg.dtype = dtype
    return cfg
