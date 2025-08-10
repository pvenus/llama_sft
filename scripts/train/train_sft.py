from __future__ import annotations
import subprocess
from pathlib import Path
import shlex
import sys
import shutil
import argparse
from typing import Optional

__all__ = [
    "train_sft",
    "prepare_data_dir",
]


def prepare_data_dir(train_path: Path | str, eval_path: Path | str, work_dir: Path) -> Path:
    """Prepare directory structure that mlx_lm lora expects.
    It requires filenames to be exactly train.jsonl / valid.jsonl inside a folder.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(train_path), work_dir / "train.jsonl")
    shutil.copy2(Path(eval_path),  work_dir / "valid.jsonl")
    return work_dir


def train_sft(
    *,
    base_model: str = "mlx-community/Bio-Medical-Llama-3-2-1B-CoT-012025",
    train_path: str | Path,
    eval_path: str | Path,
    output_dir: str | Path = "outputs/train/adapters/sample/",
    batch_size: int = 2,
    iters: int = 500,
    eval_every_steps: int = 100,
    save_every_steps: int = 200,
    lr: float = 2e-4,
    max_seq_len: int = 512,
    val_batches: int = 50,
    mask_prompt: bool = True,
    grad_checkpoint: bool = True,
    fine_tune_type: str = "lora",  # lora | dora | full
    steps_per_report: int = 50,
    resume_adapter_file: Optional[str | Path] = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # mlx_lm lora expects a directory with train.jsonl / valid.jsonl
    data_dir = prepare_data_dir(train_path, eval_path, output_dir / "lora_data")

    adapter_path = output_dir

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", base_model,
        "--train",                  # enable training mode
        "--data", str(data_dir),    # directory containing train.jsonl / valid.jsonl
        "--adapter-path", str(adapter_path),
        "--batch-size", str(batch_size),
        "--iters", str(iters),
        "--learning-rate", str(lr),
        "--steps-per-eval", str(eval_every_steps),
        "--save-every", str(save_every_steps),
        "--max-seq-length", str(max_seq_len),
        "--val-batches", str(val_batches),
        "--fine-tune-type", fine_tune_type,
        "--steps-per-report", str(steps_per_report),
    ]

    if mask_prompt:
        cmd.append("--mask-prompt")
    if grad_checkpoint:
        cmd.append("--grad-checkpoint")
    if resume_adapter_file:
        cmd += ["--resume-adapter-file", str(resume_adapter_file)]

    print("Running:\n ", " \\\n  ".join(map(shlex.quote, cmd)))
    subprocess.run(cmd, check=True)

    print(f"\n[OK] Finished. LoRA adapters saved under directory: {adapter_path}")
    print(f"Data dir used: {data_dir}")
    return adapter_path


# -------- CLI (no YAML; import-friendly) --------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run LoRA fine-tuning with mlx_lm without YAML; also importable as train_sft()",
    )
    ap.add_argument("--base-model", default="mlx-community/Bio-Medical-Llama-3-2-1B-CoT-012025",
                    help="HF repo id or local path of the base model")
    ap.add_argument("--train-path", required=True, help="Path to train.jsonl source file")
    ap.add_argument("--eval-path", required=True, help="Path to eval.jsonl source file")
    ap.add_argument(
        "--output-dir",
        default="outputs/train/adapters",
        help="Directory to write LoRA adapters (checkpoints + adapter_config.json).",
    )

    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--eval-every-steps", type=int, default=100)
    ap.add_argument("--save-every-steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--val-batches", type=int, default=50)

    ap.add_argument("--no-mask-prompt", action="store_true", help="Do not mask prompt tokens in loss")
    ap.add_argument("--no-grad-checkpoint", action="store_true", help="Disable gradient checkpointing")
    ap.add_argument("--fine-tune-type", default="lora", choices=["lora", "dora", "full"])
    ap.add_argument("--steps-per-report", type=int, default=50)
    ap.add_argument("--resume-adapter-file", default=None, help="Resume from this adapters file")

    return ap.parse_args()


def main():
    args = _parse_args()
    train_sft(
        base_model=args.__dict__["base_model"],
        train_path=args.__dict__["train_path"],
        eval_path=args.__dict__["eval_path"],
        output_dir=args.__dict__["output_dir"],
        batch_size=args.__dict__["batch_size"],
        iters=args.__dict__["iters"],
        eval_every_steps=args.__dict__["eval_every_steps"],
        save_every_steps=args.__dict__["save_every_steps"],
        lr=args.__dict__["lr"],
        max_seq_len=args.__dict__["max_seq_len"],
        val_batches=args.__dict__["val_batches"],
        mask_prompt=not args.__dict__["no_mask_prompt"],
        grad_checkpoint=not args.__dict__["no_grad_checkpoint"],
        fine_tune_type=args.__dict__["fine_tune_type"],
        steps_per_report=args.__dict__["steps_per_report"],
        resume_adapter_file=args.__dict__["resume_adapter_file"],
    )


if __name__ == "__main__":
    main()