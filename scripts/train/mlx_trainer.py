from __future__ import annotations
from pathlib import Path
import sys
import subprocess
import shlex
from .trainter_helper import _ensure_dir, _prepare_train, _finished_msg


def train_with_mlx(cfg) -> Path:
    """Run LoRA fine-tuning via mlx_lm CLI using settings from cfg."""
    output_dir = _ensure_dir(cfg.output_dir)

    # Unified data layout for MLX CLI
    data_dir, _, _ = _prepare_train(cfg)

    adapter_path = output_dir

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", cfg.base_model,
        "--train",
        "--data", str(data_dir),
        "--adapter-path", str(adapter_path),
        "--batch-size", str(cfg.batch_size),
        "--iters", str(cfg.iters),
        "--learning-rate", str(cfg.lr),
        "--steps-per-eval", str(cfg.eval_every_steps),
        "--save-every", str(cfg.save_every_steps),
        "--max-seq-length", str(cfg.max_seq_len),
        "--val-batches", str(cfg.val_batches),
        "--fine-tune-type", cfg.fine_tune_type,
        "--steps-per-report", str(cfg.steps_per_report),
    ]

    if getattr(cfg, "mask_prompt", False):
        cmd.append("--mask-prompt")
    if getattr(cfg, "grad_checkpoint", False):
        cmd.append("--grad-checkpoint")
    if getattr(cfg, "resume_adapter_file", None):
        cmd += ["--resume-adapter-file", str(cfg.resume_adapter_file)]

    print("Running:\n ", " \\\n  ".join(map(shlex.quote, cmd)))
    out = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(
            f"[mlx_lm lora failed] code={out.returncode}\nSTDOUT:\n{out.stdout or '<empty>'}\nSTDERR:\n{out.stderr or '<empty>'}\n"
        )
    if out.stdout:
        print(out.stdout)

    _finished_msg("MLX LoRA training", output_dir, data_dir)
    return adapter_path
