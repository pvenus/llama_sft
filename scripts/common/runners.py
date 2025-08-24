# runners.py — unified runner implementations for HF and MLX backends
from __future__ import annotations
import os
import sys as _sys

__all__ = ["get_runner", "init_runner", "release_runner", "clear_runners", "runner_count"]

# Runner pool (indexed). Each index maps to a runner instance and its cache key.
from typing import Dict, Tuple, Any
import threading
_POOL_LOCK = threading.Lock()
RUNNERS: Dict[int, Any] = {}
RUNNER_KEYS: Dict[int, Tuple[str, str, str]] = {}

# --- Compatibility shim: handle broken/old tqdm variants ---
def _patch_tqdm_if_broken():
    """Make tqdm safe/no-op if it's missing pieces to avoid crashes in hooks."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")
    try:
        import tqdm as _tq
        # Early exit if both attributes exist
        if hasattr(_tq, "tqdm") and hasattr(_tq, "_lock"):
            return
        # Patch missing pieces
        if not hasattr(_tq, "tqdm"):
            class _Dummy:
                _lock = None
                def __init__(self, *a, **k): pass
                def update(self, *a, **k): pass
                def close(self, *a, **k): pass
                def set_description(self, *a, **k): pass
            _tq.tqdm = _Dummy  # type: ignore
        if not hasattr(_tq, "_lock"):
            try:
                _tq._lock = None  # type: ignore
            except Exception:
                pass
    except ImportError:
        # Provide a minimal stub module if tqdm is missing entirely
        class _Dummy:
            _lock = None
            def __init__(self, *a, **k): pass
            def update(self, *a, **k): pass
            def close(self, *a, **k): pass
            def set_description(self, *a, **k): pass
        stub = type("_TQDMStub", (), {"tqdm": _Dummy, "_lock": None})()
        _sys.modules["tqdm"] = stub  # type: ignore

# --- HF (Transformers) runner ---
class _HFRunner:
    def __init__(self, model: str, adapters: str | None = None):
        try:
            import torch
            torch.manual_seed(0)
        except Exception:
            pass
        _patch_tqdm_if_broken()
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        # torch is already imported above; reuse it below
        from transformers import AutoTokenizer, AutoModelForCausalLM
        try:
            from peft import PeftModel  # optional
        except Exception:
            PeftModel = None

        # Prefer MPS on macOS, then CUDA, else CPU
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.tok = AutoTokenizer.from_pretrained(model)

        # Normalize pad/eos tokens so decoding/stopping is consistent
        if self.tok.pad_token_id is None and self.tok.eos_token_id is not None:
            try:
                self.tok.pad_token = self.tok.eos_token
            except Exception:
                pass

        dtype = (torch.float16 if self.device in ("cuda", "mps") else None)
        self.m = AutoModelForCausalLM.from_pretrained(model, torch_dtype=dtype)

        if adapters and PeftModel is not None:
            self.m = PeftModel.from_pretrained(self.m, adapters)

        if self.device != "cpu":
            self.m = self.m.to(self.device)

        try:
            self.m.eval()
        except Exception:
            pass

    def generate(self, prompt: str, max_tokens: int = 32) -> str:
        import torch
        inputs = self.tok(prompt, return_tensors="pt", add_special_tokens=False)
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        in_len = inputs["input_ids"].shape[1]
        seq = self.m.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            eos_token_id=self.tok.eos_token_id,
            pad_token_id=self.tok.pad_token_id,
        )
        gen_ids = seq[0][in_len:]
        return self.tok.decode(gen_ids, skip_special_tokens=True)

def init_runner(model: str, adapters: str | None = None, index: int = 0):
    """Initialize (or reuse) a runner at the given pool index.

    The runner is reused if the (platform, model, adapters) key matches.
    Otherwise, a new runner is created and stored at that index.
    """
    key = (_sys.platform, model, adapters or "")
    with _POOL_LOCK:
        cur = RUNNERS.get(index)
        cur_key = RUNNER_KEYS.get(index)
        if cur is not None and cur_key == key:
            return cur
        try:
            runner = _HFRunner(model, adapters)
        except Exception as e:
            raise RuntimeError(f"Model load failed: {e}")
        RUNNERS[index] = runner
        RUNNER_KEYS[index] = key
        return runner

def get_runner(model: str, adapters: str | None = None, index: int = 0):
    """Get (or lazily create) a runner at the given pool index."""
    with _POOL_LOCK:
        if index in RUNNERS:
            return RUNNERS[index]
    return init_runner(model, adapters, index=index)

# --- Pool management helpers ---
def release_runner(index: int = 0):
    """Delete the runner at the given index from the pool (if any)."""
    with _POOL_LOCK:
        RUNNERS.pop(index, None)
        RUNNER_KEYS.pop(index, None)

def clear_runners():
    """Clear all runners from the pool."""
    with _POOL_LOCK:
        RUNNERS.clear()
        RUNNER_KEYS.clear()

def runner_count() -> int:
    """Return the number of active runners in the pool."""
    with _POOL_LOCK:
        return len(RUNNERS)