from __future__ import annotations

from dataclasses import dataclass

# Support both "python -m scripts.train.train_sft" and direct "python scripts/train/train_sft.py"
try:
    from ..common.env import backend as env_backend  # when run as a package module
except Exception:  # ImportError or no parent package
    import os, sys
    _THIS = os.path.abspath(__file__)
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from scripts.common.env import backend as env_backend  # fall back to absolute-in-project import
import argparse
from typing import Optional
from pathlib import Path
from dataclasses import dataclass

__all__ = [
    "train_sft",
]

@dataclass
class SFTConfig:
    base_model: str = "meta-llama/Llama-3.2-1B-Instruct"
    train_path: str | Path = ""
    eval_path: str | Path = ""
    output_dir: str | Path = "outputs/train/adapters/sample/"
    batch_size: int = 2
    iters: int = 500
    eval_every_steps: int = 100
    save_every_steps: int = 200
    lr: float = 2e-4
    max_seq_len: int = 512
    val_batches: int = 50
    mask_prompt: bool = True
    grad_checkpoint: bool = True
    fine_tune_type: str = "lora"   # lora | dora | full
    steps_per_report: int = 50
    resume_adapter_file: Optional[str | Path] = None


def train_sft(cfg: SFTConfig) -> Path:
    """Public entrypoint: defaults live in SFTConfig; callers pass an instance here."""
    be = env_backend()
    if be == "mlx":
        from scripts.train.mlx_trainer import train_with_mlx
        return train_with_mlx(cfg)
    else:
        from scripts.train.hf_trainer import train_with_peft
        return train_with_peft(cfg)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run LoRA fine-tuning with mlx_lm without YAML; also importable as train_sft()",
    )

    # Required paths
    ap.add_argument(
        "--train-path",
        required=True,
        help="Path to train.jsonl source file",
    )
    ap.add_argument(
        "--eval-path",
        required=True,
        help="Path to eval.jsonl source file",
    )

    # Model and output
    ap.add_argument(
        "--base-model",
        help="HF repo id or local path of the base model",
    )
    ap.add_argument(
        "--output-dir",
        help=(
            "Directory to write LoRA adapters "
            "(checkpoints + adapter_config.json)."
        ),
    )

    # Training parameters
    ap.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for training and evaluation",
    )
    ap.add_argument(
        "--iters",
        type=int,
        help="Number of training iterations",
    )
    ap.add_argument(
        "--eval-every-steps",
        type=int,
        help="Evaluation frequency in steps",
    )
    ap.add_argument(
        "--save-every-steps",
        type=int,
        help="Checkpoint save frequency in steps",
    )
    ap.add_argument(
        "--lr",
        type=float,
        help="Learning rate",
    )
    ap.add_argument(
        "--max-seq-len",
        type=int,
        help="Maximum sequence length",
    )
    ap.add_argument(
        "--val-batches",
        type=int,
        help="Number of validation batches",
    )

    # Flags
    ap.add_argument(
        "--no-mask-prompt",
        action="store_true",
        help="Do not mask prompt tokens in loss",
    )
    ap.add_argument(
        "--no-grad-checkpoint",
        action="store_true",
        help="Disable gradient checkpointing",
    )

    # Fine-tuning options
    ap.add_argument(
        "--fine-tune-type",
        choices=["lora", "dora", "full"],
        help="Type of fine-tuning to perform",
    )
    ap.add_argument(
        "--steps-per-report",
        type=int,
        help="Number of steps between logging reports",
    )
    ap.add_argument(
        "--resume-adapter-file",
        help="Resume from this adapters file",
    )

    return ap.parse_args()


def main():
    args = _parse_args()
    cfg = SFTConfig()  # start from dataclass defaults

    # required paths
    cfg.train_path = args.train_path
    cfg.eval_path = args.eval_path

    # optional overrides (only if provided)
    if args.base_model is not None:
        cfg.base_model = args.base_model
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.iters is not None:
        cfg.iters = args.iters
    if args.eval_every_steps is not None:
        cfg.eval_every_steps = args.eval_every_steps
    if args.save_every_steps is not None:
        cfg.save_every_steps = args.save_every_steps
    if args.lr is not None:
        cfg.lr = args.lr
    if args.max_seq_len is not None:
        cfg.max_seq_len = args.max_seq_len
    if args.val_batches is not None:
        cfg.val_batches = args.val_batches
    if args.fine_tune_type is not None:
        cfg.fine_tune_type = args.fine_tune_type
    if args.steps_per_report is not None:
        cfg.steps_per_report = args.steps_per_report
    if args.resume_adapter_file is not None:
        cfg.resume_adapter_file = args.resume_adapter_file

    # boolean flags (invert when flags are present)
    if getattr(args, "no_mask_prompt", False):
        cfg.mask_prompt = False
    if getattr(args, "no_grad_checkpoint", False):
        cfg.grad_checkpoint = False

    train_sft(cfg)


if __name__ == "__main__":
    main()