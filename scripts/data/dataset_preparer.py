#!/usr/bin/env python3
"""Dataset Preparer

Convert CSV/JSON sources into chat-style JSONL with train/eval(/test) splits.
- Supports JSON arrays (with configurable keys) and CSV with input/output columns
- Provides optional best-effort repair for malformed JSON strings in CSV
- Keeps the original CLI interface intact

Usage (examples):
  python -m scripts.dataset_preparer \
    --in-root assets/datasets/sample \
    --out-dir outputs/datasets/sample
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

# ==============================================================================
# Loaders
# ==============================================================================


def _repair_json_string(s: str) -> str:
    """Best-effort fixer for common malformed JSON-in-CSV cases.

    Applied only when `--repair` is provided. This heuristic tries to:
    1) Unescape common sequences (\" -> ", \n, \t)
    2) Quote bare keys, e.g., ``{calls: ...}`` -> ``{"calls": ...}``

    Parameters
    ----------
    s : str
        Raw JSON string from a CSV cell.

    Returns
    -------
    str
        A potentially repaired JSON string.
    """
    t = s.strip()
    # 1) collapse common escaped quotes/newlines/tabs once
    t = t.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
    # 2) add quotes around bare keys: {calls: ...} or , calls: ...
    #    This will NOT touch already-quoted keys
    t = re.sub(r'([\{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', t)
    return t


def load_json_array(
    path: Path,
    user_key: str = "user",
    assistant_key: str = "assistant",
) -> List[Dict[str, str]]:
    """Load a JSON array file into canonical {user, assistant} items.

    The assistant value may be a JSON string or a dict; both are normalized
    to a compact JSON string with UTF-8 preserved.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: top-level JSON must be an array")

    items: List[Dict[str, str]] = []
    for ex in data:
        if not isinstance(ex, dict):
            raise ValueError(f"{path}: array elements must be JSON objects, got: {type(ex)}")
        if user_key not in ex or assistant_key not in ex:
            missing = [k for k in (user_key, assistant_key) if k not in ex]
            raise KeyError(f"{path}: missing key(s) {missing} in JSON object: {ex}")

        user_val = ex[user_key]
        assistant_val = ex[assistant_key]

        if not isinstance(user_val, str):
            raise TypeError(f"{path}: '{user_key}' must be a string, got {type(user_val)}")

        # Normalize assistant to a valid JSON string
        if isinstance(assistant_val, dict):
            assistant_str = json.dumps(assistant_val, ensure_ascii=False)
        elif isinstance(assistant_val, str):
            try:
                obj = json.loads(assistant_val)
            except Exception as e:  # keep path context for debugging
                raise ValueError(
                    f"{path}: invalid assistant JSON string: {e}\nvalue={assistant_val!r}"
                ) from e
            assistant_str = json.dumps(obj, ensure_ascii=False)
        else:
            raise TypeError(f"{path}: '{assistant_key}' must be dict or JSON string")

        items.append({"user": user_val.strip(), "assistant": assistant_str})
    return items


def load_csv(
    path: Path,
    input_col: str = "input",
    output_col: str = "output_json",
    repair: bool = False,
) -> List[Dict[str, str]]:
    """Load a CSV into canonical {user, assistant} items.

    If `repair=True`, apply a best-effort fix for malformed JSON strings.
    Empty or incomplete rows are skipped.
    """
    items: List[Dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{path.name}: CSV has no header row")
        if input_col not in reader.fieldnames or output_col not in reader.fieldnames:
            raise KeyError(
                f"{path.name}: CSV must contain columns '{input_col}', '{output_col}'. Found: {reader.fieldnames}"
            )
        for row in reader:
            user = (row.get(input_col) or "").strip()
            raw_out = (row.get(output_col) or "").strip()
            if not user or not raw_out:
                # skip empty/incomplete rows
                continue
            try:
                out_obj = json.loads(raw_out)
            except Exception as e:
                if repair:
                    fixed = _repair_json_string(raw_out)
                    try:
                        out_obj = json.loads(fixed)
                    except Exception as e2:
                        raise ValueError(
                            f"{path.name}: invalid JSON in column '{output_col}'.\n"
                            f"  original error: {e}\n  value={raw_out!r}\n"
                            f"  after repair error: {e2}\n  repaired={fixed!r}"
                        ) from e2
                else:
                    raise ValueError(
                        f"{path.name}: invalid JSON in column '{output_col}': {e}\nvalue={raw_out!r}"
                    ) from e
            assistant_str = json.dumps(out_obj, ensure_ascii=False)
            items.append({"user": user, "assistant": assistant_str})
    return items


def to_messages_item(user_text: str, tool_json_str: str, system_prompt: str) -> Dict[str, Any]:
    """Wrap a QA pair into chat-style messages with a system prompt."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "assistant": tool_json_str,
    }


# ==============================================================================
# Utilities
# ==============================================================================


def parse_split(spec: str) -> List[float]:
    """Parse split like '90,10' or '80,10,10' and return normalized ratios.

    Accepts either percentages or already-normalized ratios; values are normalized
    to sum to 1.0 to avoid rounding drift.
    """
    try:
        parts = [float(x.strip()) for x in spec.split(",") if x.strip()]
    except Exception as e:
        raise ValueError(
            "--split must be a comma-separated list of numbers, e.g., '90,10' or '80,10,10'"
        ) from e
    if not parts:
        raise ValueError("--split is empty")

    s = sum(parts)
    ratios = parts if s <= 1.001 else [x / s for x in parts]
    if len(ratios) not in (2, 3):
        raise ValueError("--split must have 2 (train,eval) or 3 (train,eval,test) values")

    total = sum(ratios)
    return [x / total for x in ratios]


def gather_files(root: Path, pattern: str, recursive: bool = True) -> List[Path]:
    """Return files under *root* matching a (possibly comma-separated) glob *pattern*."""
    if not pattern:
        return []
    files: List[Path] = []
    for pat in (p.strip() for p in pattern.split(",") if p.strip()):
        files.extend(root.rglob(pat) if recursive else root.glob(pat))
    # unique & sorted
    return sorted({p for p in files if p.is_file()})


def dump_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Write rows as JSONL (UTF-8) to *path*."""
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ==============================================================================
# Core
# ==============================================================================


def prepare_dataset(
    *,
    in_root: str = "assets/datasets/sample",
    out_dir: str = "outputs/datasets/sample",
    json_pattern: str = "json/*.json",
    csv_pattern: str = "csv/*.csv",
    csv_input_col: str = "input",
    csv_output_col: str = "output_json",
    system_prompt: str = (
        'You are a function calling agent. Output ONLY a single JSON object of the form: '
        '{"calls":[{"name":"...", "arguments":{}}]}'
    ),
    split: str = "80,10,10",
    seed: int = 42,
    no_shuffle: bool = False,
    limit: int = 0,
    train_name: str = "train.jsonl",
    eval_name: str = "eval.jsonl",
    test_name: str = "test.jsonl",
    repair: bool = False,
) -> Dict[str, Any]:
    """Prepare train/eval(/test) JSONL files and return a small report dict."""
    in_root_path = Path(in_root)
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Collect inputs
    json_files = gather_files(in_root_path, json_pattern)
    csv_files = gather_files(in_root_path, csv_pattern)

    if not json_files and not csv_files:
        print(
            f"[ERR] No input files found under {in_root_path}. "
            f"json_pattern='{json_pattern}', csv_pattern='{csv_pattern}'",
            file=sys.stderr,
        )
        raise FileNotFoundError("No input files found")

    items: List[Dict[str, str]] = []
    for p in json_files:
        items.extend(load_json_array(p))
    for p in csv_files:
        items.extend(load_csv(p, input_col=csv_input_col, output_col=csv_output_col, repair=repair))

    # transform to messages format
    msg_items: List[Dict[str, Any]] = [
        to_messages_item(ex["user"], ex["assistant"], system_prompt) for ex in items
    ]

    # optional limit
    if limit and limit > 0:
        msg_items = msg_items[:limit]

    # shuffle
    if not no_shuffle:
        rng = random.Random(seed)
        rng.shuffle(msg_items)

    # split
    ratios = parse_split(split)
    n = len(msg_items)
    if len(ratios) == 2:
        n_train = int(n * ratios[0])
        train, eval_, test_ = msg_items[:n_train], msg_items[n_train:], []
    else:
        n_train = int(n * ratios[0])
        n_eval = int(n * ratios[1])
        train = msg_items[:n_train]
        eval_ = msg_items[n_train : n_train + n_eval]
        test_ = msg_items[n_train + n_eval :]

    train_path = out_dir_path / train_name
    eval_path = out_dir_path / eval_name
    test_path = out_dir_path / test_name if test_ else None

    dump_jsonl(train_path, train)
    dump_jsonl(eval_path, eval_)
    if test_ and test_path is not None:
        dump_jsonl(test_path, test_)

    # minimal reporting
    print(
        f"done: {len(train)} train, {len(eval_)} eval" + (f", {len(test_)} test" if test_ else "")
    )
    print(f"from {len(json_files)} JSON file(s), {len(csv_files)} CSV file(s)")
    print(f"in_root={in_root_path} out_dir={out_dir_path}")

    result: Dict[str, Any] = {
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "test_path": str(test_path) if test_path else None,
        "counts": {"train": len(train), "eval": len(eval_), "test": len(test_) if test_ else 0},
        "sources": {"json_files": len(json_files), "csv_files": len(csv_files)},
        "repair": repair,
    }
    print(result)
    return result


def resolve_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Normalize CLI args for `prepare_dataset`. Keeps the schema stable for tests."""
    return {
        "in_root": args.in_root,
        "out_dir": args.out_dir,
        "json_pattern": args.json_pattern,
        "csv_pattern": args.csv_pattern,
        "csv_input_col": args.csv_input_col,
        "csv_output_col": args.csv_output_col,
        "system_prompt": args.system_prompt,
        "split": args.split,
        "seed": args.seed,
        "no_shuffle": args.no_shuffle,
        "limit": args.limit,
        "train_name": args.train_name,
        "eval_name": args.eval_name,
        "test_name": args.test_name,
        "repair": args.repair,
    }


# ==============================================================================
# CLI
# ==============================================================================


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Prepare dataset (CSV/JSON -> chat JSONL) with train/eval(/test) splits.",
    )
    # Paths & Patterns
    ap.add_argument("--in-root", default="assets/datasets/sample", help="input root folder that contains json/ and/or csv/ subfolders")
    ap.add_argument("--out-dir", default="outputs/datasets/sample", help="output folder for train/eval(/test).jsonl")
    ap.add_argument("--json-pattern", default="json/*.json", help="glob under in-root for JSON arrays (comma-separated allowed)")
    ap.add_argument("--csv-pattern", default="csv/*.csv", help="glob under in-root for CSV files (comma-separated allowed)")

    # CSV Columns
    ap.add_argument("--csv-input-col", default="input", help="CSV column for user text")
    ap.add_argument("--csv-output-col", default="output_json", help="CSV column for assistant JSON string")

    # Prompt & Split
    ap.add_argument("--system-prompt", default='You are a function calling agent. Output ONLY a single JSON object of the form: {"calls":[{"name":"...", "arguments":{}}]}', help="system prompt inserted in messages[0]")
    ap.add_argument("--split", default="80,10,10", help="train,eval(,test) percentages. e.g., '90,10' or '80,10,10'")
    ap.add_argument("--seed", type=int, default=42, help="random seed for shuffling and split")
    ap.add_argument("--no-shuffle", action="store_true", help="do not shuffle before split")
    ap.add_argument("--limit", type=int, default=0, help="optional cap on total items before split (0=all)")

    # Output Filenames & Repair
    ap.add_argument("--train-name", default="train.jsonl")
    ap.add_argument("--eval-name", default="eval.jsonl")
    ap.add_argument("--test-name", default="test.jsonl")
    ap.add_argument("--repair", action="store_true", help="try to auto-repair malformed JSON strings in CSV (best-effort)")
    return ap


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    R = resolve_args(args)
    # brief config echo
    print(
        f"in_root={R['in_root']} out_dir={R['out_dir']} split={R['split']} "
        f"json={R['json_pattern']} csv={R['csv_pattern']} repair={R['repair']}"
    )
    prepare_dataset(**R)


if __name__ == "__main__":
    main()