#!/usr/bin/env python3
"""SimpleTrain — pretrain a Kimi K3 hybrid (KDA + Gated MLA) transformer.

Example:
    $ python train.py --variant 1b --batch_size 16 --num_steps 10000 \
                      --wandb_project simpletrain-dev
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

# Make sibling modules importable when running from inside SimpleTrain/
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F

from config import config_from_cli
from dataset import Tokenizer, make_train_loader
from models import build_model
from utils import metrics
from utils.logger import get_logger, make_tracker, now_ms
from utils.metrics import (
    cosine_warmup_lr,
    detect_gpu,
    estimate_flops_per_step,
    onboarding_summary,
    tokens_per_second,
)

log = get_logger()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train.py",
        description="Pretrain a Kimi K3 transformer on FineWeb (SimpleTrain).",
    )
    p.add_argument("--variant", choices=["100m", "1b", "3b", "7b", "10b"], default="1b")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--num_steps", type=int, default=10000)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--context_length", type=int, default=2048)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # WandB
    p.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    p.add_argument("--no-wandb", dest="wandb", action="store_false")
    p.add_argument("--wandb_project", default="simpletrain-kimi-k3")
    p.add_argument("--wandb_run_name", default=None)
    # Kernels
    p.add_argument("--use_triton", dest="use_triton", action="store_true", default=True)
    p.add_argument("--no-triton", dest="use_triton", action="store_false")
    # Data
    p.add_argument("--fineweb_subset", default="sample-10BT",
                   help="FineWeb subset to stream (e.g. sample-10BT, sample-100BT, 2023-10).")
    # Misc
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--compile", action="store_true", default=False,
                   help="Apply torch.compile to the model (experimental).")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=20)
    return p


# --------------------------------------------------------------------------- #
# Optimizer
# --------------------------------------------------------------------------- #
def build_optimizer(model: torch.nn.Module, cfg):
    """AdamW with weight decay applied only to 2D params (standard GPT recipe)."""
    decay, nodecay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay.append(p)
        else:
            nodecay.append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": nodecay, "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(
        groups, lr=cfg.learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=False
    )
    return opt


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def main():
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)

    # 1) Build config
    cfg = config_from_cli(
        variant=args.variant,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        context_length=args.context_length,
        use_triton=args.use_triton,
        dtype=args.dtype,
    )
    cfg.weight_decay = args.weight_decay
    cfg.warmup_steps = args.warmup_steps
    cfg.grad_clip    = args.grad_clip

    # 2) Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_device(0)
    else:
        device = torch.device("cpu")
        if cfg.use_triton:
            log.warning("No CUDA device — Triton kernels disabled automatically.")
            cfg.use_triton = False

    gpu = detect_gpu()

    # 3) Build model
    model = build_model(cfg, device=device)
    n_params = model.num_active_parameters()

    if args.compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            log.info("torch.compile enabled (mode=reduce-overhead).")
        except Exception as e:
            log.warning("torch.compile failed (%s) — running eager.", e)

    # 4) Optimizer
    opt = build_optimizer(model, cfg)

    # 5) WandB
    tracker = make_tracker(
        enabled=args.wandb,
        project=args.wandb_project,
        run_name=args.wandb_run_name or f"kimi-k3-{cfg.variant}",
        config={
            **cfg.to_dict(),
            "gpu": gpu.short if gpu else "cpu",
            "peak_tflops": gpu.peak_tflops if gpu else 0.0,
            "n_params": n_params,
            "layout": model.layout if hasattr(model, "layout") else "?",
        },
    )

    # 6) Onboarding summary
    print(onboarding_summary(gpu, n_params, target_mfu=0.30))
    print(f"Layer layout : {model.layer_layout_summary()}")
    print(f"Variant      : {cfg.variant}  (d_model={cfg.d_model}, n_layers={cfg.n_layers})")
    print(f"Batch        : {cfg.batch_size} × {cfg.grad_accum} (grad_accum)")
    print(f"Context      : {cfg.context_length}")
    print(f"Triton       : {'enabled' if cfg.use_triton else 'disabled'}")
    print(f"Total steps  : {cfg.num_steps}")
    print("=" * 78)

    # 7) Data
    tokenizer = Tokenizer()
    # patch vocab size into config if mismatch
    if tokenizer.vocab_size != cfg.vocab_size:
        log.info(
            "Adjusting vocab_size %d -> %d to match tokenizer.",
            cfg.vocab_size, tokenizer.vocab_size,
        )
        cfg.vocab_size = tokenizer.vocab_size
        # rebuild model with corrected vocab
        model = build_model(cfg, device=device)
        if args.compile:
            try: model = torch.compile(model, mode="reduce-overhead")
            except Exception: pass
        opt = build_optimizer(model, cfg)
        n_params = model.num_active_parameters()

    train_iter = make_train_loader(cfg, device, tokenizer=tokenizer,
                                   subset=args.fineweb_subset, split="train")
    val_iter   = make_train_loader(cfg, device, tokenizer=tokenizer,
                                   subset=args.fineweb_subset, split="validation")

    # 8) FLOPs accounting
    flops_per_step = estimate_flops_per_step(
        n_params=n_params,
        batch_size=cfg.batch_size,
        context_length=cfg.context_length,
        grad_accum=cfg.grad_accum,
    )
    peak_tflops = gpu.peak_tflops if gpu else 0.0

    # 9) Training loop
    model.train()
    train_iter_obj = iter(train_iter)
    step = 0
    accum_loss = 0.0
    log.info("Starting training ...")

    while step < cfg.num_steps:
        t0 = time.perf_counter()

        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(cfg.grad_accum):
            try:
                batch = next(train_iter_obj)
            except StopIteration:
                train_iter_obj = iter(train_iter)
                batch = next(train_iter_obj)

            loss, _logits = model(batch.input_ids, batch.labels)
            (loss / cfg.grad_accum).backward()
            total_loss += loss.item()

        # gradient clip
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        # LR schedule
        lr = cosine_warmup_lr(
            step, peak_lr=cfg.learning_rate,
            warmup_steps=cfg.warmup_steps, total_steps=cfg.num_steps,
        )
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.step()

        t1 = time.perf_counter()
        step_latency = t1 - t0
        mean_loss = total_loss / cfg.grad_accum
        accum_loss += mean_loss

        # ---- log every log_every ----
        if (step + 1) % args.log_every == 0:
            mfu = metrics.compute_mfu(flops_per_step, step_latency, peak_tflops) if peak_tflops > 0 else 0.0
            tps = tokens_per_second(cfg.batch_size, cfg.context_length, step_latency)
            mem = (torch.cuda.memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
            perplexity = math.exp(min(20.0, mean_loss))

            tracker.log({
                "train/loss":       mean_loss,
                "train/perplexity": perplexity,
                "train/lr":         lr,
                "train/step_latency_ms": step_latency * 1000,
                "train/tokens_per_sec":  tps,
                "train/mfu":        mfu,
                "train/gpu_mem_gib": mem,
            }, step=step + 1)

            print(
                f"step {step+1:>6}/{cfg.num_steps}  "
                f"loss={mean_loss:7.4f}  ppl={perplexity:8.2f}  "
                f"lr={lr:.2e}  "
                f"lat={step_latency*1000:7.1f}ms  "
                f"tok/s={tps:8.0f}  "
                f"mfu={mfu*100:5.1f}%  "
                f"mem={mem:5.2f}GiB"
            )

        # ---- periodic eval ----
        if (step + 1) % args.eval_every == 0 and args.eval_steps > 0:
            val_loss = evaluate(model, val_iter, args.eval_steps, cfg)
            tracker.log({
                "val/loss": val_loss,
                "val/perplexity": math.exp(min(20.0, val_loss)),
            }, step=step + 1)
            print(f"           -- val_loss={val_loss:.4f}  val_ppl={math.exp(min(20.0, val_loss)):.2f}")
            model.train()

        step += 1

    # 10) Done
    log.info("Training complete. Final avg loss = %.4f", accum_loss / max(1, cfg.num_steps))
    tracker.finish()
    train_iter.close()
    val_iter.close()


# --------------------------------------------------------------------------- #
# Eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, val_iter, steps: int, cfg) -> float:
    model.eval()
    total, n = 0.0, 0
    it = iter(val_iter)
    for _ in range(steps):
        try:
            b = next(it)
        except StopIteration:
            break
        loss, _ = model(b.input_ids, b.labels)
        total += loss.item()
        n += 1
    return total / max(1, n)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
