from __future__ import annotations
import platform

def is_macos_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}


def has_cuda() -> bool:
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def backend() -> str:
    """
    macOS + Apple Silicon -> 'mlx'
    CUDA 가능 -> 'cuda'
    그 외 -> 'cpu'
    """
    if is_macos_apple_silicon():
        return "mlx"
    if has_cuda():
        return "cuda"
    return "cpu"
from pathlib import Path
import shlex
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

def _train_with_mlx_lm(
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

def _train_with_peft_transformers(
    *,
    base_model: str,
    train_path: str | Path,
    eval_path: str | Path,
    output_dir: str | Path,
    batch_size: int,
    iters: int,
    eval_every_steps: int,
    save_every_steps: int,
    lr: float,
    max_seq_len: int,
    val_batches: int,
    mask_prompt: bool,
    grad_checkpoint: bool,
    fine_tune_type: str,
    steps_per_report: int,
    resume_adapter_file: Optional[str | Path],
) -> Path:
    from pathlib import Path
    from dataclasses import dataclass
    from typing import List, Dict, Any
    import json
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
    )
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- load data (chat jsonl: messages + assistant) ---
    def load_jsonl(p: str | Path):
        rows = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    def to_text(ex: Dict[str, Any]) -> Dict[str, str]:
        msgs = ex.get("messages", [])
        sys_txt = ""
        user_txt = ""
        for m in msgs:
            if m.get("role") == "system":
                sys_txt = m.get("content", "")
            if m.get("role") == "user":
                user_txt = m.get("content", "")
        prompt = (sys_txt + "\n\n" if sys_txt else "") + f"User: {user_txt}\nAssistant: "
        target = ex.get("assistant", "")
        return {"prompt": prompt, "target": target}

    train_rows = [to_text(r) for r in load_jsonl(train_path)]
    eval_rows  = [to_text(r) for r in load_jsonl(eval_path)]

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tokenize(batch):
        # concatenate prompt+target; supervise on all tokens (simple SFT)
        texts = [p + t for p, t in zip(batch["prompt"], batch["target"])]
        enc = tok(texts, padding=True, truncation=True, max_length=max_seq_len)
        return enc

    train_ds = Dataset.from_list(train_rows).map(tokenize, batched=True, remove_columns=["prompt", "target"])  # type: ignore
    eval_ds  = Dataset.from_list(eval_rows ).map(tokenize, batched=True, remove_columns=["prompt", "target"])  # type: ignore

    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    if device == "cuda":
        model = model.to("cuda")
    if grad_checkpoint:
        model.gradient_checkpointing_enable()

    if fine_tune_type.lower() != "lora":
        raise ValueError("This path currently supports LoRA only. Use fine_tune_type='lora'.")

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"],
    )
    model = get_peft_model(model, lora_cfg)

    if resume_adapter_file:
        # PeftModel expects a directory; if a file is given, use its parent
        resume_dir = Path(resume_adapter_file)
        resume_dir = resume_dir if resume_dir.is_dir() else resume_dir.parent
        model = PeftModel.from_pretrained(model, resume_dir)

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        max_steps=iters,
        eval_strategy="steps",
        eval_steps=eval_every_steps,
        save_steps=save_every_steps,
        learning_rate=lr,
        logging_steps=steps_per_report,
        fp16=(device == "cuda"),
        report_to=[],
        save_total_limit=2,
    )

    # For plain next-token LM we can use default collator
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tok,
    )

    trainer.train()

    # save PEFT adapters in output_dir
    model.save_pretrained(str(output_dir))
    tok.save_pretrained(str(output_dir))

    print(f"\n[OK] Finished. PEFT LoRA adapters saved under directory: {output_dir}")
    return output_dir

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
    be = backend()
    if be == "mlx":
        return _train_with_mlx_lm(
            base_model=base_model,
            train_path=train_path,
            eval_path=eval_path,
            output_dir=output_dir,
            batch_size=batch_size,
            iters=iters,
            eval_every_steps=eval_every_steps,
            save_every_steps=save_every_steps,
            lr=lr,
            max_seq_len=max_seq_len,
            val_batches=val_batches,
            mask_prompt=mask_prompt,
            grad_checkpoint=grad_checkpoint,
            fine_tune_type=fine_tune_type,
            steps_per_report=steps_per_report,
            resume_adapter_file=resume_adapter_file,
        )
    else:
        return _train_with_peft_transformers(
            base_model=base_model,
            train_path=train_path,
            eval_path=eval_path,
            output_dir=output_dir,
            batch_size=batch_size,
            iters=iters,
            eval_every_steps=eval_every_steps,
            save_every_steps=save_every_steps,
            lr=lr,
            max_seq_len=max_seq_len,
            val_batches=val_batches,
            mask_prompt=mask_prompt,
            grad_checkpoint=grad_checkpoint,
            fine_tune_type=fine_tune_type,
            steps_per_report=steps_per_report,
            resume_adapter_file=resume_adapter_file,
        )


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
from scripts.common.env import backend
import subprocess
import sys

_HF_CACHE = {}

def _gen_hf(model: str, prompt: str, max_tokens=32, adapter_path=None) -> str:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    try:
        from peft import PeftModel
    except Exception:
        PeftModel = None  # optional

    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = (model, adapter_path, device)
    tok, m = _HF_CACHE.get(key, (None, None))
    if tok is None:
        tok = AutoTokenizer.from_pretrained(model)
        m = AutoModelForCausalLM.from_pretrained(model, torch_dtype=(torch.float16 if device=="cuda" else None))
        if adapter_path and PeftModel is not None:
            m = PeftModel.from_pretrained(m, adapter_path)
        if device == "cuda":
            m = m.to("cuda")
        _HF_CACHE[key] = (tok, m)
    inputs = tok(prompt, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to("cuda") for k,v in inputs.items()}
    out = m.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    return tok.decode(out[0], skip_special_tokens=True)

def gen_cli(model, prompt, max_tokens=32, adapter_path=None):
    if backend() == "mlx":
        # original MLX path (keep existing CLI call implementation here)
        cmd = [
            sys.executable, "-m", "mlx_lm", "generate",
            "--model", model,
            "--ignore-chat-template",
            "--temp", "0.0", "--top-k", "1",
            "--max-tokens", str(max_tokens),
            "--prompt", prompt,
        ]
        if adapter_path:
            cmd += ["--adapter-path", adapter_path]
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            # fallback to alternative entrypoint once, then fail
            cmd2 = [
                sys.executable, "-m", "mlx_lm.generate",
                "--model", model,
                "--ignore-chat-template",
                "--temp", "0.0", "--top-k", "1",
                "--max-tokens", str(max_tokens),
                "--prompt", prompt,
            ]
            if adapter_path:
                cmd2 += ["--adapter-path", adapter_path]
            out2 = subprocess.run(cmd2, capture_output=True, text=True)
            if out2.returncode != 0:
                err = (
                    f"[mlx_lm generate] failed twice.\n\n"
                    f"Command: {out.args}\nReturn code: {out.returncode}\nSTDERR:\n{out.stderr or '<empty>'}\n\n--- RETRY ---\n"
                    f"Command: {out2.args}\nReturn code: {out2.returncode}\nSTDERR:\n{out2.stderr or '<empty>'}\n\n"
                )
                raise RuntimeError(err)
            return out2.stdout.strip()
        return out.stdout.strip()
    else:
        return _gen_hf(model, prompt, max_tokens=max_tokens, adapter_path=adapter_path)