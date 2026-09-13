SimpleTrain
Pretrain a Kimi K3 hybrid transformer (KDA + Gated MLA) on FineWeb
with custom Triton kernels and a ≥30% MFU target across Ampere / Ada /
Hopper / Blackwell.
```bash
$ git clone https://github.com/danilosistu-lang/SimpleTrain && cd SimpleTrain
$ pip install -r requirements.txt
$ python train.py --variant 1b --batch_size 16 --num_steps 10000 \
                  --wandb_project simpletrain-dev
```
---
Architecture — Kimi K3 (hybrid)
The model alternates two attention mechanisms in a 3 : 1 ratio:
Layer type	Role
Kimi Delta Attention (KDA)	Gated linear attention with chunkwise recurrent state updates. Cheap, no softmax, O(T·D) cost.
Gated Multi-head Latent Attn	Compressed-KV multi-query attention (DeepSeek-V2 style). Used for global context.
Layer layout policy (`config.build_layer_layout`):
Interior layers follow the 3:1 KDA:MLA rhythm.
First and last layers are Gated MLA so the network opens and closes with global attention.
On top of the hybrid attention stack:
AttnRes — 2-branch attention residual: `h = x + α·attn(norm(x))` then `h = h + ffn(norm(h))`. Stabilises gradient flow across very deep stacks.
SiTUGLU / SwiGLU FFN — `out = (silu(x@W_gate) ⊙ (x@W_up)) @ W_down` with the elementwise `silu·mul` fused into a Triton kernel.
RMSNorm everywhere (final LN + per-block pre-norm).
Tied input/output embeddings.
---
Variants
`--variant` selects one of five scales (estimated active parameter counts,
tied embeddings, bf16):
variant	d_model	n_layers	n_heads	d_head	est. params
`100m`	512	10	8	64	~80 M
`1b`	1280	24	10	128	~860 M
`3b`	2048	28	16	128	~2.45 B
`7b`	3072	36	24	128	~6.95 B
`10b`	3584	40	28	128	~10.5 B
The 1B / 3B / 7B / 10B variants use `context_length = 4096` by default;
100m and 1b use 2048. Override with `--context_length`.
---
CLI
```
python train.py --variant 1b --batch_size 16 --num_steps 10000 \
                --grad_accum 1 --learning_rate 3e-4 \
                --context_length 2048 \
                --wandb_project simpletrain-dev \
                --use_triton
```
Flag	Default	Notes
`--variant`	`1b`	One of `100m / 1b / 3b / 7b / 10b`.
`--batch_size`	`16`	Per-GPU micro-batch.
`--grad_accum`	`1`	Gradient accumulation steps.
`--num_steps`	`10000`	Total optimiser steps.
`--learning_rate`	`3e-4`	Peak LR; cosine warmup → decay to 10% of peak.
`--context_length`	`2048`	Sequence length.
`--warmup_steps`	`200`	Linear warmup length.
`--weight_decay`	`0.1`	AdamW weight decay (applied to 2-D params only).
`--grad_clip`	`1.0`	Max grad norm.
`--wandb` / `--no-wandb`	`--wandb`	Toggle WandB.
`--wandb_project`	`simpletrain-kimi-k3`	WandB project name.
`--wandb_run_name`	`kimi-k3-<variant>`	WandB run name.
`--use_triton` / `--no-triton`	`--use_triton`	Toggle custom Triton kernels.
`--dtype`	`bfloat16`	`bfloat16` / `float16` / `float32`.
`--compile`	off	Apply `torch.compile(mode="reduce-overhead")` to the model.
`--fineweb_subset`	`sample-10BT`	FineWeb subset (e.g. `sample-10BT`, `sample-100BT`, `2023-10`).
`--log_every`	`10`	Log every N steps.
`--eval_every`	`500`	Run validation every N steps.
`--eval_steps`	`20`	Number of validation micro-batches.
`--seed`	`1337`	RNG seed.
---
Triton kernels
All kernels live under `kernels/` and ship with a pure-PyTorch fallback
that is mathematically identical (and logged once via
`utils.logger.log_fallback_warning`). The dispatch function checks:
Triton is importable.
GPU compute-capability is in the autotune target list
`{sm_80, sm_86, sm_89, sm_90, sm_100, sm_120}`
→ Ampere, Ada, Hopper, Blackwell.
The kernel doesn't raise at compile/run time.
If any check fails, the function falls back to the torch path.
`kernels/kda_kernel.py` — chunked gated linear attention
One Triton program per `(batch, head)` pair. The recurrent state `S ∈ ℝ^{D×D}`
is held in registers across all chunks of the sequence, so the only HBM
traffic for the state is a single write-back at the end.
Mathematical formulation:
```
s_t  = decay · s_{t-1} + k_t v_t^T
o_t  = q_t @ s_t
y_t  = g_t · o_t                # output gate
```
Autotune configs cover `BLOCK_D ∈ {32, 64, 128}` ×
`num_warps ∈ {4, 8, 16}` × `num_stages ∈ {2, 3, 4, 5}` — the per-arch best
is picked at first launch. Caps `BLOCK_D = 128` (larger heads fall back
to torch).
`kernels/fused_ops.py`
Function	Fuses
`fused_silu_mul`	`silu(a) ⊙ b` — removes one global-memory round-trip per FFN.
`fused_rmsnorm_residual`	`x / rms(x) · weight + residual` — single kernel, no intermediates.
`fused_cross_entropy`	Stable online softmax (running max + running sum) + NLL → mean loss + softmax probs.
---
MFU accounting
`utils/metrics.estimate_flops_per_step`:
```
FLOPs / step  ≈  6 · N_params · (batch_size · context_length · grad_accum)
```
`utils/metrics.compute_mfu`:
```
MFU = (FLOPs / step_latency) / peak_tflops
```
Peak FLOP tables (`utils/metrics._GPU_PEAK_TFLOPS`) cover:
Family	SKUs
Ampere	A100 (40G/80G, SXM/PCIe), RTX A6000
Ada/Lovelace	RTX 5050, 5070, 5070 Ti, 5080, 5090, RTX 4090
Hopper	H100 SXM/PCIe, H800
Blackwell	B200, B300, GB200
If the GPU isn't in the table, the peak is estimated from
`SM_count × 0.25 TF/s` (very conservative) — patch the table if you have
more accurate numbers.
---
Data pipeline (`dataset.py`)
Streams `HuggingFaceFW/fineweb` (subset `sample-10BT` by default) via
`datasets.load_dataset(..., streaming=True)`.
Tokenises with tiktoken (`gpt2` BPE by default, swap to
`cl100k_base` by passing `encoding="cl100k_base"` to `Tokenizer`).
Falls back to HF's `GPT2TokenizerFast` if tiktoken isn't installed.
Sequence packing into `[B, T]` micro-batches with pinned-memory
prefetching on a background thread (`PackedTokenIterator`).
If FineWeb can't be streamed (e.g. offline dev), falls back to
`wikitext-2-raw-v1` so smoke runs still work.
---
File layout
```
SimpleTrain/
├── train.py                # CLI + training loop
├── config.py               # Variant configs + layer layout
├── dataset.py              # FineWeb streaming + tokenizer + packing
├── models/
│   ├── __init__.py
│   └── kimi_k3.py          # KDA + Gated MLA + AttnRes + SiTUGLU FFN
├── kernels/
│   ├── __init__.py
│   ├── kda_kernel.py       # Chunked gated-linear-attention Triton kernel
│   └── fused_ops.py        # SiTUGLU / RMSNorm+Residual / fused CE
├── utils/
│   ├── __init__.py
│   ├── metrics.py          # GPU peak FLOP table + MFU/FLOP accounting
│   └── logger.py           # WandB + stdout logging
├── requirements.txt
└── README.md
```
---
Onboarding summary
When training starts, an onboarding block is printed:
```
==============================================================================
SimpleTrain — Kimi K3 (KDA + Gated MLA)  Pretraining
==============================================================================
GPU           : NVIDIA A100-SXM4-80GB
Compute cap.  : sm_80  (A100)
Peak TFLOP/s  : 312.0  (FP16/BF16 dense tensor)
Device memory : 80.0 GiB
Active params : 860.0 M  (≈0.860 B)
Target MFU    : 30%  (≥30% required by spec)
==============================================================================
Layer layout : KDA:17  MLA:7  (ratio 17:7)
Variant      : 1b  (d_model=1280, n_layers=24)
Batch        : 16 × 1 (grad_accum)
Context      : 2048
Triton       : enabled
Total steps  : 10000
==============================================================================
```
---
Example performance (target)
On a single A100 80GB, `--variant 1b --batch_size 16 --context_length 2048`:
Metric	Target	Notes
Step latency	~0.4–0.6 s	With Triton kernels + bf16.
Tokens / sec	~55 K	`batch_size · context_length / step_latency`.
MFU	≥ 30%	196 TF / step ÷ 312 TF/s peak ≈ 63% at 1 s/step.
GPU memory	< 60 GiB	bf16 + AdamW state (8 bytes/param).
MFU scales linearly with batch × context; for H100/B200 use
`--batch_size 32` or `--grad_accum 4` to saturate the tensor cores.
---
Notes & limitations
The KDA Triton kernel currently caps `BLOCK_D` at 128. Heads larger
than 128 fall back to the (vectorised) torch implementation — fast
enough for development but won't hit peak MFU.
The `gate` projection in both KDA and Gated MLA is zero-initialised so
the layers behave close to identity at step 0 — this is intentional
and improves early-training stability.
Position embeddings use a simple learned `nn.Embedding(context_length, d)`
for portability. RoPE would be a drop-in replacement and is left as a
follow-up.
`flash-attn` is in `requirements.txt` as an optional Linux-only dep —
the Gated MLA path uses `F.scaled_dot_product_attention` which will
dispatch to flash-attention-2 when it's available.
---
License
MIT.
