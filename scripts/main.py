from __future__ import annotations

import platform
from pathlib import Path

# package-local imports (run as module: `python -m scripts.main`)
from .data.dataset_preparer import prepare_dataset
from .train.train_sft import train_sft
from .eval.eval_suite import run_eval_suite

# optional helper: detect backend (mlx vs others)
try:
    from .common.env import backend  # returns 'mlx' on Apple Silicon setups
except Exception:
    def backend() -> str:
        # conservative default
        return "cpu"


def pick_model() -> str:
    """
    Choose a default base model depending on OS / backend.
    - Windows: meta-llama/Llama-3.2-1B-Instruct  (note: gated on HF; requires access)
    - Apple Silicon (mlx backend): mlx-community/Bio-Medical-Llama-3-2-1B-CoT-012025
    - Other (Linux/CPU/CUDA): Qwen/Qwen2.5-1.5B-Instruct (open, CUDA-friendly)
    """
    if platform.system().lower() == "windows":
        return "meta-llama/Llama-3.2-1B-Instruct"

    be = (backend() or "").lower()
    if be == "mlx":
        return "mlx-community/Bio-Medical-Llama-3-2-1B-CoT-012025"

    # default for non-Apple Silicon (Linux/CPU/CUDA)
    return "Qwen/Qwen2.5-1.5B-Instruct"


if __name__ == "__main__":
    # 1) Prepare dataset (uses defaults in dataset_preparer)
    R_D = prepare_dataset()

    # 2) Pick model based on environment
    base_model = pick_model()
    print(f"[main] Selected base model: {base_model}")

    # 3) Train SFT (LoRA) — pass the chosen base model if supported
    #    train_sft returns the adapters path (e.g., outputs/.../adapters.npz or a folder)
    R_T = train_sft(
        train_path=R_D["train_path"],
        eval_path=R_D["eval_path"],
        base_model=base_model,  # if train_sft ignores it, it will use its own default
    )

    # 4) Evaluate (baseline vs finetuned suite). We pass model + adapters.
    #    run_eval_suite will read eval file from arguments or use defaults.
    adapters_path = str(R_T) if isinstance(R_T, (str, Path)) else str(R_T)
    run_eval_suite(model=base_model, adapters=adapters_path)
