"""Hardware metrics: GPU peak FLOP tables + MFU / FLOP accounting.

The peak FLOP tables cover all GPU SKUs called out in the project brief:

    * Ampere      : A100 40G/80G, RTX A6000
    * Ada/Lovelace: RTX 5050, RTX 5090 (also covers RTX 4090 as a courtesy)
    * Hopper      : H100 SXM / PCIe
    * Blackwell   : B200, B300

Numbers are *theoretical* FP16/BF16 dense peaks (no sparsity, no FP8).
Sources: NVIDIA datasheets + cuBLAS reference benchmarks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch


# --------------------------------------------------------------------------- #
# Peak FLOP database (TFLOP/s, FP16/BF16 dense, non-tensor-core baseline)
# --------------------------------------------------------------------------- #
#
# Notes:
# * Hopper / Blackwell tensor-core TFLOP/s are quoted for the **warp-level
#   matrix-multiply-assist (mma)** path that user Triton kernels typically
#   hit. We deliberately do not use the cuDNN "structured sparsity" 2x number
#   nor the FP8 numbers, so the MFU we compute is conservative.
# * B300 numbers are based on the publicly announced specs at the time of
#   writing; if the SKU ships with higher clocks the table can be patched
#   in-place.
#
_GPU_PEAK_TFLOPS: Dict[str, float] = {
    # ---- Ampere ----
    "A100-SXM-40GB":      312.0,
    "A100-SXM-80GB":      312.0,
    "A100-PCIe-40GB":     312.0,
    "A100-PCIe-80GB":     312.0,
    "A100":               312.0,
    "RTX A6000":          155.0,
    "Quadro RTX A6000":   155.0,
    # ---- Ada Lovelace / RTX 50 series (Blackwell-SK consumer class) ----
    "RTX 5050":            25.0,
    "RTX 5070":            46.0,
    "RTX 5070 Ti":         70.0,
    "RTX 5080":            90.0,
    "RTX 5090":           126.0,    # Blackwell GB202, FP16 dense tensor
    "RTX 4090":            82.6,
    "RTX 4080":            48.7,
    "RTX 3090":            35.6,
    # ---- Hopper ----
    "H100-SXM":           989.0,
    "H100-PCIe":          756.0,
    "H100":               989.0,
    "H800":               989.0,
    # ---- Blackwell datacenter ----
    "B200":              2250.0,    # FP16/BF16 dense tensor peak
    "B300":              2800.0,
    "GB200":             2250.0,
}


# --------------------------------------------------------------------------- #
# GPU info
# --------------------------------------------------------------------------- #
@dataclass
class GPUInfo:
    name: str               # e.g. "NVIDIA A100-SXM4-40GB"
    short: str              # canonical key into _GPU_PEAK_TFLOPS, e.g. "A100"
    peak_tflops: float      # FP16/BF16 dense tensor peak
    total_memory_gib: float
    cc_major: int           # compute capability major (8 = Ampere, 9 = Hopper/Ada, 10 = Blackwell)
    cc_minor: int

    def __str__(self) -> str:
        return (
            f"{self.name} (cc {self.cc_major}.{self.cc_minor}, "
            f"{self.peak_tflops:.1f} TF/s FP16/BF16, "
            f"{self.total_memory_gib:.1f} GiB)"
        )


def _canonical_short(name: str) -> str:
    """Map a raw `torch.cuda.get_device_name()` string to a peak-table key."""
    n = name.upper()
    if "A100" in n:
        return "A100"
    if "A6000" in n:
        return "RTX A6000"
    if "H100" in n:
        return "H100"
    if "H800" in n:
        return "H800"
    if "B200" in n or "GB200" in n:
        return "B200"
    if "B300" in n:
        return "B300"
    # RTX 50xx / 40xx / 30xx consumer cards
    for tag in ("5090", "5080", "5070 TI", "5070", "5050", "4090", "4080", "3090"):
        if tag in n:
            return f"RTX {tag}"
    return name.strip()


def detect_gpu(device: Optional[torch.device] = None) -> Optional[GPUInfo]:
    """Return a :class:`GPUInfo` for the requested device or ``None`` on CPU."""
    if not torch.cuda.is_available():
        return None
    dev = device or torch.cuda.current_device()
    raw_name = torch.cuda.get_device_name(dev)
    short = _canonical_short(raw_name)
    peak = _GPU_PEAK_TFLOPS.get(
        short,
        # Fallback: derive a rough peak from SM count × clock.
        torch.cuda.get_device_properties(dev).multi_processor_count * 0.25,
    )
    prop = torch.cuda.get_device_properties(dev)
    total_gib = prop.total_memory / (1024 ** 3)
    return GPUInfo(
        name=raw_name,
        short=short,
        peak_tflops=float(peak),
        total_memory_gib=total_gib,
        cc_major=int(prop.major),
        cc_minor=int(prop.minor),
    )


# --------------------------------------------------------------------------- #
# FLOP accounting
# --------------------------------------------------------------------------- #
def estimate_flops_per_step(
    n_params: int,
    batch_size: int,
    context_length: int,
    grad_accum: int = 1,
) -> int:
    """Approximate FLOPs executed in *one optimizer step*.

    Standard approximation for dense transformer pretraining (forward+backward):

        FLOPs ≈ 6 * N_params * tokens_processed

    where ``tokens_processed = batch_size * context_length * grad_accum``.
    See Chinchilla / PaLM compute-measurement conventions.
    """
    tokens = batch_size * context_length * max(1, grad_accum)
    return int(6 * n_params * tokens)


def compute_mfu(
    flops_per_step: int,
    step_latency_sec: float,
    peak_tflops: float,
) -> float:
    """Return MFU as a fraction in ``[0, 1]``."""
    if step_latency_sec <= 0 or peak_tflops <= 0:
        return 0.0
    actual_tflops = (flops_per_step / step_latency_sec) / 1e12
    return float(min(1.0, actual_tflops / peak_tflops))


def tokens_per_second(batch_size: int, context_length: int, step_latency_sec: float) -> float:
    if step_latency_sec <= 0:
        return 0.0
    return float(batch_size * context_length) / step_latency_sec


# --------------------------------------------------------------------------- #
# Cosine LR with warmup (used by the training loop)
# --------------------------------------------------------------------------- #
def cosine_warmup_lr(step: int, *, peak_lr: float, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup -> cosine decay to 10% of peak."""
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    if step >= total_steps:
        return 0.1 * peak_lr
    decay_steps = max(1, total_steps - warmup_steps)
    progress = (step - warmup_steps) / decay_steps
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    # Decay down to 10% of peak
    return peak_lr * (0.1 + 0.9 * cos)


# --------------------------------------------------------------------------- #
# Onboarding banner
# --------------------------------------------------------------------------- #
def onboarding_summary(gpu: Optional[GPUInfo], n_params: int, target_mfu: float = 0.30) -> str:
    """Pretty onboarding string printed before the training loop starts."""
    sep = "=" * 78
    lines = [
        sep,
        "SimpleTrain — Kimi K3 (KDA + Gated MLA)  Pretraining",
        sep,
    ]
    if gpu is None:
        lines += [
            "GPU           : <not detected>  (running on CPU — kernels will fall back)",
            "Peak TFLOP/s  : n/a",
        ]
    else:
        lines += [
            f"GPU           : {gpu.name}",
            f"Compute cap.  : sm_{gpu.cc_major}{gpu.cc_minor}  ({gpu.short})",
            f"Peak TFLOP/s  : {gpu.peak_tflops:,.1f}  (FP16/BF16 dense tensor)",
            f"Device memory : {gpu.total_memory_gib:.1f} GiB",
        ]
    lines += [
        f"Active params : {n_params/1e6:,.1f} M  (≈{n_params/1e9:.3f} B)",
        f"Target MFU    : {target_mfu*100:.0f}%  (≥30% required by spec)",
        sep,
    ]
    return "\n".join(lines)
