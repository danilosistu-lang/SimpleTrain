"""FineWeb dataset loader + tokenizer + sequence packing.

Features
--------
* Streams `HuggingFaceFW/fineweb` (or `fineweb-edu`) via `datasets`.
* Tokenises with `tiktoken` (cl100k_base / GPT-2 BPE) or HuggingFace's
  GPT-2 tokenizer as a fallback.
* **Sequence packing**: concatenates tokens up to `context_length` and emits
  fixed `[B, T]` micro-batches. Labels = inputs shifted by one (the pack
  boundary is treated as a real prediction target; for cleaner training the
  user can pass `--no_pack` to use simple per-doc truncation).
* Pre-fetching with a background thread + pinned memory.
"""
from __future__ import annotations

import os
import queue
import threading
from dataclasses import dataclass
from typing import Iterator, List, Optional

import torch

from utils.logger import get_logger

log = get_logger()


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #
class Tokenizer:
    """Thin wrapper that prefers tiktoken, falls back to HF GPT-2 BPE."""

    def __init__(self, encoding: str = "gpt2"):
        self.encoding_name = encoding
        self._enc = None
        try:
            import tiktoken
            self._enc = tiktoken.get_encoding(encoding if encoding != "llama" else "cl100k_base")
            self.vocab_size = self._enc.n_vocab
            log.info("Tokenizer: tiktoken %s (vocab=%d)", encoding, self.vocab_size)
        except Exception as e:
            log.warning("tiktoken unavailable (%s); falling back to HF GPT-2 tokenizer.", e)
            from transformers import GPT2TokenizerFast
            self._enc = GPT2TokenizerFast.from_pretrained("gpt2")
            self.vocab_size = self._enc.vocab_size

    def encode(self, text: str) -> List[int]:
        if hasattr(self._enc, "encode"):
            ids = self._enc.encode(text)
        else:                                            # HF tokenizer
            ids = self._enc.encode(text)["input_ids"]
        return list(ids)

    def decode(self, ids: List[int]) -> str:
        return self._enc.decode(ids)


# --------------------------------------------------------------------------- #
# Dataset streamer
# --------------------------------------------------------------------------- #
def stream_fineweb(
    split: str = "train",
    subset: str = "sample-10BT",
    dataset_name: str = "HuggingFaceFW/fineweb",
):
    """Lazy-stream FineWeb from HF Hub.

    The 10BT sample is the smallest publicly-streamable subset and is what
    we use by default. Pass `subset="sample-100BT"` or a named dump like
    `subset="2023-10"` for larger streams.
    """
    from datasets import load_dataset
    log.info("Streaming %s (%s/%s) ...", dataset_name, subset, split)
    try:
        ds = load_dataset(dataset_name, name=subset, split=split, streaming=True)
    except Exception as e:
        log.warning(
            "Could not stream %s[%s/%s] (%s). "
            "Falling back to tiny `wikitext-2-raw-v1` for smoke-testing.",
            dataset_name, subset, split, e,
        )
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split, streaming=True)
    return ds


# --------------------------------------------------------------------------- #
# Packed-token iterator
# --------------------------------------------------------------------------- #
@dataclass
class PackedBatch:
    input_ids: torch.Tensor      # [B, T]
    labels:    torch.Tensor      # [B, T]


class PackedTokenIterator:
    """Consumes the HF stream, tokenises, packs into [B, T] tensors.

    Runs tokenisation + packing on a background thread so the GPU never
    starves. Uses a bounded queue + pinned memory.
    """

    def __init__(
        self,
        stream,
        tokenizer: Tokenizer,
        batch_size: int,
        context_length: int,
        device: torch.device,
        prefetch: int = 4,
        pack: bool = True,
    ):
        self.stream = stream
        self.tok = tokenizer
        self.B = batch_size
        self.T = context_length
        self.device = device
        self.pack = pack
        self._q: queue.Queue = queue.Queue(maxsize=prefetch)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # ------------- background worker -------------
    def _worker(self):
        try:
            buf: List[int] = []
            target_len = self.B * self.T
            for ex in self.stream:
                if self._stop.is_set():
                    break
                text = ex.get("text", "")
                if not text:
                    continue
                ids = self.tok.encode(text)
                if not self.pack:
                    # truncate / pad per-doc; we still need target_len tokens.
                    ids = ids[: self.T]
                    buf.extend(ids)
                    buf.extend([0] * (self.T - len(ids)))
                else:
                    buf.extend(ids)
                while len(buf) >= target_len:
                    chunk = buf[:target_len]
                    buf = buf[target_len:]
                    self._push(chunk)
            # final partial flush
            if len(buf) >= self.T:
                # pad to one last batch
                buf.extend([0] * (target_len - len(buf)))
                self._push(buf[:target_len])
        except Exception as e:                          # pragma: no cover
            log.error("PackedTokenIterator worker crashed: %s", e)
        finally:
            self._q.put(None)                            # sentinel

    def _push(self, chunk: List[int]):
        arr = torch.tensor(chunk, dtype=torch.long).view(self.B, self.T)
        labels = torch.cat([arr[:, 1:], torch.zeros(self.B, 1, dtype=torch.long)], dim=1)
        if self.device.type == "cuda":
            arr = arr.pin_memory()
            labels = labels.pin_memory()
        self._q.put((arr, labels))

    # ------------- public API -------------
    def __iter__(self) -> Iterator[PackedBatch]:
        while True:
            item = self._q.get()
            if item is None:
                return
            arr, labels = item
            yield PackedBatch(
                input_ids=arr.to(self.device, non_blocking=True),
                labels=labels.to(self.device, non_blocking=True),
            )

    def close(self):
        self._stop.set()
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass


# --------------------------------------------------------------------------- #
# Convenience factory
# --------------------------------------------------------------------------- #
def make_train_loader(
    cfg,
    device: torch.device,
    tokenizer: Optional[Tokenizer] = None,
    split: str = "train",
    subset: str = "sample-10BT",
) -> PackedTokenIterator:
    tok = tokenizer or Tokenizer()
    stream = stream_fineweb(split=split, subset=subset)
    return PackedTokenIterator(
        stream=stream,
        tokenizer=tok,
        batch_size=cfg.batch_size,
        context_length=cfg.context_length,
        device=device,
        pack=True,
    )
