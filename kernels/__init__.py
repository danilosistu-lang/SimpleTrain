"""Custom Triton kernels for SimpleTrain.

All kernels gracefully fall back to native PyTorch (or torch.compile) when
Triton is unavailable or the detected GPU architecture is not in the
auto-tune target list. The fallback path is logged once per process via
:mod:`utils.logger`.
"""
from . import kda_kernel, fused_ops  # noqa: F401

__all__ = ["kda_kernel", "fused_ops"]
