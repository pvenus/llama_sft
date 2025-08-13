#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    # prefer package-style import
    from .eval_fc_cli import evaluate_fc
except Exception:
    # fallback
    from .eval_fc_cli import evaluate_fc  # type: ignore

__all__ = [
    "metrics_from_jsonl",
    "run_eval_suite",
]


def metrics_from_jsonl(path: Path) -> Dict[str, float]:
    """Read the JSONL produced by eval_fc_cli and compute metrics/fallback usage."""
    total = 0
    em = 0
    name1 = 0
    state = 0
    fb_used = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            total += 1
            gold, pred = r.get("gold"), r.get("pred")
            g_calls = (gold or {}).get("calls") or []
            p_calls = (pred or {}).get("calls") or []
            if pred and gold and json.dumps(pred, sort_keys=True) == json.dumps(gold, sort_keys=True):
                em += 1
            if p_calls and g_calls:
                name1 += int(p_calls[0].get("name") == g_calls[0].get("name"))

            # state compare
            def apply(state, calls):
                s = dict(state)
                for c in calls:
                    n = c.get("name")
                    if n == "power_on":
                        s["power"] = "on"
                    elif n == "power_off":
                        s["power"] = "off"
                    elif n == "enable_night_mode":
                        s["night_mode"] = True
                return s

            s0 = {"power": "off", "night_mode": False}
            state += int(apply(s0, p_calls) == apply(s0, g_calls))

            # fallback usage (eval_fc_cli writes `source` field)
            if r.get("source") == "fallback":
                fb_used += 1

    return dict(
        total=total,
        exact=em / total if total else 0.0,
        name1=name1 / total if total else 0.0,
        state=state / total if total else 0.0,
        fb=fb_used,
        fb_rate=(fb_used / total if total else 0.0),
    )


def run_eval_suite(
    *,
    model: str = "mlx-community/Bio-Medical-Llama-3-2-1B-CoT-012025",
    adapters: Optional[str] = None,
    file: str = "outputs/datasets/sample/test.jsonl",
    out_dir: str = "outputs/eval_suite/sample",
    max_tokens: int = 32,
    spec: str = "assets/train/fc_patterns.json",
) -> Tuple[Path, List[Tuple[str, Optional[Dict[str, float]]]]]:
    """Run a 4-way eval (baseline/finetuned × fallback on/off) and return results.

    Returns (out_root_dir, [(case_name, metrics_or_None), ...])
    """
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_root = Path(out_dir) / stamp
    out_root.mkdir(parents=True, exist_ok=True)

    combos = [
        dict(name="baseline_fb_on", no_fb=False, finetuned=False),
        dict(name="baseline_nofb", no_fb=True, finetuned=False),
        dict(name="finetuned_fb_on", no_fb=False, finetuned=True),
        dict(name="finetuned_nofb", no_fb=True, finetuned=True),
    ]

    results: List[Tuple[str, Optional[Dict[str, float]]]] = []

    for cb in combos:
        out_file = out_root / f"{cb['name']}.jsonl"

        # decide adapters for this run
        adp_path = adapters if cb["finetuned"] else None
        if cb["finetuned"] and not adapters:
            # skip finetuned cases if no adapters are provided
            results.append((cb["name"], None))
            continue

        # delegate to eval function (imported from eval_fc_cli)
        evaluate_fc(
            model=model,
            file=file,
            out=str(out_file),
            adapters=adp_path,
            max_tokens=max_tokens,
            no_fallback=cb["no_fb"],
            spec=spec,
        )

        metrics = metrics_from_jsonl(out_file)
        results.append((cb["name"], metrics))

    return out_root, results


def _print_summary(out_root: Path, results: List[Tuple[str, Optional[Dict[str, float]]]]) -> None:
    def fmt(x: float) -> str:
        return f"{x:.3f}"

    print("\n=== SUMMARY ===")
    print(f"Out dir: {out_root}")
    header = ["case", "Total", "ExactMatch", "FuncName@1", "StateMatch", "FallbackUsed"]
    print("\t".join(header))
    for name, m in results:
        if m is None:
            print(f"{name}\t(skipped)")
            continue
        print("\t".join([
            name,
            str(m["total"]),
            fmt(m["exact"]),
            fmt(m["name1"]),
            fmt(m["state"]),
            f"{m['fb']} ({m['fb_rate']:.1%})",
        ]))


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run a 4-case eval suite (baseline/finetuned × fb on/off)")
    ap.add_argument("--model", default="mlx-community/Bio-Medical-Llama-3-2-1B-CoT-012025", help="Model name or path")
    ap.add_argument("--adapters", default=None, help="Path to finetuned adapters directory; if omitted, finetuned cases are skipped")
    ap.add_argument("--file", default="outputs/datasets/sample/test.jsonl", help="Path to eval jsonl")
    ap.add_argument("--out_dir", default="outputs/eval_suite/sample", help="Directory to store evaluation outputs")
    ap.add_argument("--max_tokens", type=int, default=32, help="Max generation tokens per sample")
    ap.add_argument("--spec", default="assets/train/fc_patterns.json", help="Prompt/regex spec JSON file")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    out_root, results = run_eval_suite(
        model=args.model,
        adapters=args.adapters,
        file=args.file,
        out_dir=args.out_dir,
        max_tokens=args.max_tokens,
        spec=args.spec,
    )
    _print_summary(out_root, results)


if __name__ == "__main__":
    main()