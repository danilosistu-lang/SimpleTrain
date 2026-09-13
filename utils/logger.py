"""WandB integration + lightweight stdout logging.

This module is intentionally tolerant of `wandb` not being installed: when
WandB is unavailable or `--no-wandb` was passed, all calls become no-ops
and a small in-memory metrics buffer keeps the training loop functional.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# --------------------------------------------------------------------------- #
# stdout / stderr logger
# --------------------------------------------------------------------------- #
_LOG = logging.getLogger("simpletrain")
if not _LOG.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%H:%M:%S")
    )
    _LOG.addHandler(handler)
_LOG.setLevel(logging.INFO)


def get_logger() -> logging.Logger:
    return _LOG


# --------------------------------------------------------------------------- #
# WandB wrapper
# --------------------------------------------------------------------------- #
@dataclass
class WandBTracker:
    project: str = "simpletrain-kimi-k3"
    enabled: bool = False
    run_name: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    _wandb: Any = None
    _last_flush: float = 0.0

    def init(self) -> "WandBTracker":
        if not self.enabled:
            get_logger().info("WandB disabled — logging to stdout only.")
            return self
        try:
            import wandb  # type: ignore
            self._wandb = wandb
            wandb.init(
                project=self.project,
                name=self.run_name,
                config=self.config,
                reinit=True,
            )
            get_logger().info("WandB initialised: project=%s run=%s", self.project, wandb.run.name)
        except Exception as e:  # pragma: no cover
            get_logger().warning("WandB init failed (%s) — falling back to stdout.", e)
            self.enabled = False
            self._wandb = None
        return self

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        # Always echo to stdout (compact form) so the user sees progress.
        compact = {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in metrics.items()}
        get_logger().info("step=%s %s", step if step is not None else "?", compact)
        if not self.enabled or self._wandb is None:
            return
        try:
            self._wandb.log(metrics, step=step)
        except Exception as e:  # pragma: no cover
            get_logger().warning("wandb.log failed: %s", e)

    def finish(self) -> None:
        if self.enabled and self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# Convenience builders
# --------------------------------------------------------------------------- #
def make_tracker(
    *,
    enabled: bool,
    project: str,
    run_name: Optional[str],
    config: Dict[str, Any],
) -> WandBTracker:
    return WandBTracker(
        project=project,
        enabled=enabled and os.environ.get("WANDB_MODE", "online") != "disabled",
        run_name=run_name,
        config=config,
    ).init()


def log_fallback_warning(reason: str) -> None:
    """Single-shot warning used by kernel modules when they fall back."""
    key = f"_fallback_warned::{reason}"
    if getattr(log_fallback_warning, key, False):
        return
    setattr(log_fallback_warning, key, True)
    get_logger().warning(
        "Triton fallback engaged: %s — using native PyTorch path "
        "(consider torch.compile for partial recovery).", reason
    )


def now_ms() -> float:
    return time.perf_counter() * 1000.0
