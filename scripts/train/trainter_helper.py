from __future__ import annotations
from pathlib import Path
from shutil import copy2


def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _prepare_data_dir(train_path: Path | str, eval_path: Path | str, work_dir: Path) -> Path:
    """Create unified layout: work_dir/{train.jsonl, valid.jsonl}."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    copy2(Path(train_path), work_dir / "train.jsonl")
    copy2(Path(eval_path),  work_dir / "valid.jsonl")
    return work_dir


def _prepare_train(cfg) -> tuple[Path, Path, Path]:
    data_dir = _prepare_data_dir(cfg.train_path, cfg.eval_path, Path(cfg.output_dir) / "lora_data")
    return data_dir, data_dir / "train.jsonl", data_dir / "valid.jsonl"


def _finished_msg(kind: str, output_dir: Path, data_dir: Path | None = None) -> None:
    print(f"\n[OK] Finished {kind}. Artifacts: {output_dir}")
    if data_dir is not None:
        print(f"Data dir used: {data_dir}")
