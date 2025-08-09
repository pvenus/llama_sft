#!/usr/bin/env python3
# scripts/dataset_preparer.py (generic)
import json, csv, random, pathlib, argparse, sys, re
from pathlib import Path

# ---- loaders ----

# Best-effort fixer for common malformed JSON-in-CSV cases
#   - unescape sequences like \" -> "
#   - convert unquoted keys (e.g., {calls: ...}) to quoted ({"calls": ...})
#   - keep other characters intact
# NOTE: This is heuristic; enable only when --repair is set.

def _repair_json_string(s: str) -> str:
    t = s.strip()
    # 1) collapse common escaped quotes/newlines/backslashes once
    t = t.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
    # 2) add quotes around bare keys: {calls: ...} or , calls: ...
    #    This will NOT touch already-quoted keys
    t = re.sub(r'([\{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', t)
    return t

def load_json_array(path: Path, user_key: str = "user", assistant_key: str = "assistant"):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    items = []
    for ex in data:
        try:
            user_val = ex[user_key]
            assistant_val = ex[assistant_key]
        except KeyError as e:
            raise KeyError(f"{path}: missing key {e} in JSON object: {ex}")
        # Normalize assistant to a valid JSON string
        if isinstance(assistant_val, dict):
            assistant_str = json.dumps(assistant_val, ensure_ascii=False)
        elif isinstance(assistant_val, str):
            try:
                obj = json.loads(assistant_val)
            except Exception as e:
                raise ValueError(f"{path}: invalid assistant JSON string: {e}\nvalue={assistant_val!r}")
            assistant_str = json.dumps(obj, ensure_ascii=False)
        else:
            raise ValueError(f"{path}: assistant must be dict or JSON string")
        items.append({"user": user_val, "assistant": assistant_str})
    return items


def load_csv(path: Path, input_col: str = "input", output_col: str = "output_json", repair: bool = False):
    items = []
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if input_col not in r.fieldnames or output_col not in r.fieldnames:
            raise KeyError(
                f"{path.name}: CSV must contain columns '{input_col}', '{output_col}'. Found: {r.fieldnames}"
            )
        for row in r:
            user = (row.get(input_col) or "").strip()
            raw_out = (row.get(output_col) or "").strip()
            if not user or not raw_out:
                # skip empty/incomplete rows
                continue
            # Validate assistant JSON string and normalize
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
                        )
                else:
                    raise ValueError(f"{path.name}: invalid JSON in column '{output_col}': {e}\nvalue={raw_out!r}")
            assistant_str = json.dumps(out_obj, ensure_ascii=False)
            items.append({"user": user, "assistant": assistant_str})
    return items


def to_messages_item(user_text: str, tool_json_str: str, system_prompt: str):
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "assistant": tool_json_str,
    }


# ---- utilities ----

def parse_split(spec: str):
    """Parse split like '90,10' or '80,10,10' as percentages. Returns list of floats summing to 1.0."""
    try:
        parts = [float(x.strip()) for x in spec.split(",") if x.strip()]
    except Exception:
        raise ValueError("--split must be a comma-separated list of numbers, e.g., '90,10' or '80,10,10'")
    if not parts:
        raise ValueError("--split is empty")
    s = sum(parts)
    if s <= 1.001:  # already ratios
        ratios = parts
    else:           # treat as percentages
        ratios = [x / s for x in parts]
    if len(ratios) not in (2, 3):
        raise ValueError("--split must have 2 (train,eval) or 3 (train,eval,test) values")
    # normalize to exactly 1.0 to avoid rounding drift
    total = sum(ratios)
    ratios = [x / total for x in ratios]
    return ratios


def gather_files(root: Path, pattern: str, recursive: bool = True):
    """Return a list of files matching pattern under root. pattern may be comma-separated."""
    files = []
    if not pattern:
        return files
    for pat in [p.strip() for p in pattern.split(",") if p.strip()]:
        if recursive:
            files.extend(root.rglob(pat))
        else:
            files.extend(root.glob(pat))
    return sorted({p for p in files if p.is_file()})


def prepare_dataset(
    in_root: str = "assets/datasets",
    out_dir: str = "outputs/datasets",
    json_pattern: str = "json/*.json",
    csv_pattern: str = "csv/*.csv",
    csv_input_col: str = "input",
    csv_output_col: str = "output_json",
    system_prompt: str = 'You are a function calling agent. Output ONLY a single JSON object of the form: {"calls":[{"name":"...", "arguments":{}}]}',
    split: str = "80,10,10",
    seed: int = 42,
    no_shuffle: bool = False,
    limit: int = 0,
    train_name: str = "train.jsonl",
    eval_name: str = "eval.jsonl",
    test_name: str = "test.jsonl",
    repair: bool = False,
):
    in_root_path = Path(in_root)
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Collect inputs
    json_files = gather_files(in_root_path, json_pattern)
    csv_files = gather_files(in_root_path, csv_pattern)

    if not json_files and not csv_files:
        print(f"[ERR] No input files found under {in_root_path}. json_pattern='{json_pattern}', csv_pattern='{csv_pattern}'", file=sys.stderr)
        raise FileNotFoundError("No input files found")

    items = []
    for p in json_files:
        items.extend(load_json_array(p))
    for p in csv_files:
        items.extend(load_csv(p, input_col=csv_input_col, output_col=csv_output_col, repair=repair))

    # to messages
    msg_items = [
        to_messages_item(ex["user"], ex["assistant"], system_prompt)
        for ex in items
    ]

    # optional limit
    if limit and limit > 0:
        msg_items = msg_items[: limit]

    # shuffle
    if not no_shuffle:
        random.seed(seed)
        random.shuffle(msg_items)

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
        eval_ = msg_items[n_train:n_train+n_eval]
        test_ = msg_items[n_train+n_eval:]

    def dump_jsonl(path: Path, rows):
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    train_path = out_dir_path / train_name
    eval_path = out_dir_path / eval_name
    test_path = out_dir_path / test_name if test_ else None

    dump_jsonl(train_path, train)
    dump_jsonl(eval_path,  eval_)
    if test_:
        dump_jsonl(test_path,  test_)

    # minimal reporting
    print(f"done: {len(train)} train, {len(eval_)} eval" + (f", {len(test_)} test" if test_ else ""))
    print(f"from {len(json_files)} JSON file(s), {len(csv_files)} CSV file(s)")
    print(f"in_root={in_root_path} out_dir={out_dir_path}")

    result = {
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "test_path": str(test_path) if test_path else None,
        "counts": {"train": len(train), "eval": len(eval_), "test": len(test_) if test_ else 0},
        "sources": {"json_files": len(json_files), "csv_files": len(csv_files)},
        "repair": repair,
    }
    print(result)
    return result


def resolve_args(args):
    """Return a normalized dict of CLI args for prepare_dataset."""
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

# ---- main ----

def main():
    ap = argparse.ArgumentParser(description="Prepare dataset (CSV/JSON -> chat JSONL) with train/eval(/test) splits.")
    ap.add_argument("--in-root", default="assets/datasets", help="input root folder that contains json/ and/or csv/ subfolders")
    ap.add_argument("--out-dir", default="outputs/datasets", help="output folder for train/eval(/test).jsonl")
    ap.add_argument("--json-pattern", default="json/*.json", help="glob under in-root for JSON arrays (comma-separated allowed)")
    ap.add_argument("--csv-pattern", default="csv/*.csv", help="glob under in-root for CSV files (comma-separated allowed)")

    ap.add_argument("--csv-input-col", default="input", help="CSV column for user text")
    ap.add_argument("--csv-output-col", default="output_json", help="CSV column for assistant JSON string")

    ap.add_argument("--system-prompt", default='You are a function calling agent. Output ONLY a single JSON object of the form: {"calls":[{"name":"...", "arguments":{}}]}',
                    help="system prompt inserted in messages[0]")
    ap.add_argument("--split", default="80,10,10", help="train,eval(,test) percentages. e.g., '90,10' or '80,10,10'")
    ap.add_argument("--seed", type=int, default=42, help="random seed for shuffling and split")
    ap.add_argument("--no-shuffle", action="store_true", help="do not shuffle before split")
    ap.add_argument("--limit", type=int, default=0, help="optional cap on total items before split (0=all)")

    ap.add_argument("--train-name", default="train.jsonl")
    ap.add_argument("--eval-name", default="eval.jsonl")
    ap.add_argument("--test-name", default="test.jsonl")
    ap.add_argument("--repair", action="store_true", help="try to auto-repair malformed JSON strings in CSV (best-effort)")

    args = ap.parse_args()

    R = resolve_args(args)
    # Optional: brief config echo
    print(f"in_root={R['in_root']} out_dir={R['out_dir']} split={R['split']} json={R['json_pattern']} csv={R['csv_pattern']} repair={R['repair']}")
    prepare_dataset(**R)

if __name__ == "__main__":
    main()