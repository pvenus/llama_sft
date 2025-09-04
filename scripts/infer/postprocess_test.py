# scripts/infer/postprocess_test.py
# FunctionCall 보정기 테스트 (함수=시그니처 스키마, 인자=타겟별 규칙 스키마)
# 결과 컬럼: raw,repaired,tags,status

from __future__ import annotations
import os, json, csv, argparse, re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional

from scripts.infer.postprocess import repair_functioncall_format

try:
    import yaml
except Exception:
    yaml = None

END_TOKEN = "<end>"
FUNC_CALL_RX = re.compile(r"^<(function_[A-Za-z0-9_]+)>\((.*?)\)$", re.DOTALL)

@dataclass
class Case:
    raw: str

@dataclass
class Row:
    raw: str
    repaired: str
    tags: List[str]
    status: str  # "repaired_success" | "already_ok" | "still_broken"

# ---------------------------------------------------------------------------
def validate_format(s: str) -> bool:
    if not s: return False
    txt = s.strip()
    if not txt.endswith(END_TOKEN): return False
    head = txt[: -len(END_TOKEN)].rstrip()
    if head.endswith(";"): head = head[:-1]
    head = head.strip()
    if not head: return False
    parts = [p.strip() for p in head.split(";")]
    if any((not p) for p in parts): return False
    for p in parts:
        if not FUNC_CALL_RX.match(p): return False
    if re.search(r"\)\s*<function_", head): return False
    return True

def load_yaml(path: str) -> Optional[Dict[str, Any]]:
    if not (yaml and os.path.exists(path)): return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_sample_cases() -> List[Case]:
    ok = [
        Case("<function_BP>(enable=True)"),
        Case("<function_BP>(enable=True);<function_MR>()   "),
        Case("<function_BP(enable=True);<function_MR>()<end>"),
        Case("＜function_BP＞(enable=True);＜function_MR＞()<end>"),
        Case("<function_BP>(enable=True;<function_MR>()<end>"),
        Case("<function_BP>(enable=True)<end>"),
        Case("<function_BP>(enable=True)<end>  noise  "),
        Case("<function_BP>(enable=True)<end>   <end>"),
        Case("<function_BP>(enable=True)<end>;<function_MR>()"),
        Case("〈function_MR〉()<end>"),
        Case("<function_MR() <end>"),
        Case("＜function_MR＞(<end>"),
        Case("<function_BP>(enable=True);<function_MR()<end>"),
        Case("  <function_BP>(enable=True)   ;   <function_MR>()    <end>   "),
    ]
    ng = [
        Case("<function_BP>(enable=True)<function_MR>()<end>"),
        Case("function_BP(enable=True);<function_MR>()<end>"),
        Case("[function_BP](enable=True);<function_MR>()<end>"),
        Case("hello world"),
        Case("<function_BP>(enable=True);<function_MR()><end>"),
        Case("﹤function_BP﹥(enable=True);﹤function_MR﹥()<end>"),
    ]
    return ok + ng

def save_sample_csv(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["raw"])
        for c in build_sample_cases():
            w.writerow([c.raw])

def load_cases(path: str) -> List[Case]:
    if not os.path.exists(path): raise FileNotFoundError(f"input file not found: {path}")
    cases: List[Case] = []
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        header = [h.strip().lower() for h in rows[0]]
        idx = header.index("raw") if "raw" in header else header.index("text")
        for row in rows[1:]:
            cases.append(Case(raw=row[idx]))
    elif ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                obj = json.loads(line)
                cases.append(Case(raw=obj.get("raw") or obj.get("text") or ""))
    else:
        raise ValueError("Only .csv or .jsonl is supported")
    return cases

def run(args):
    out_root = args.outdir or "assets/repair"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_root, ts); os.makedirs(out_dir, exist_ok=True)

    # sample 생성/입력 경로 결정
    created = None
    default_csv = "assets/repair/repair_test.csv"
    if args.input:
        if not os.path.exists(args.input) and args.init_sample:
            os.makedirs(os.path.dirname(args.input), exist_ok=True)
            save_sample_csv(args.input); created = args.input
            print(f"[OK] sample created -> {created}")
    else:
        if args.init_sample:
            if os.path.exists(default_csv):
                print(f"[OK] using existing sample -> {default_csv}")
            else:
                os.makedirs(os.path.dirname(default_csv), exist_ok=True)
                save_sample_csv(default_csv); created = default_csv
                print(f"[OK] sample created -> {created}")

    in_path = args.input or created or (default_csv if os.path.exists(default_csv) else "assets/repair/repair_test.jsonl")

    # 스키마 로드
    func_schema = load_yaml(args.func_schema) if args.func_schema else None
    arg_schema  = load_yaml(args.arg_schema)  if args.arg_schema  else None

    # 실행
    cases = load_cases(in_path)
    rows: List[Row] = []
    for c in cases:
        rep = repair_functioncall_format(
            c.raw, max_rounds=args.max_rounds,
            func_schema=func_schema, arg_schema=arg_schema
        )
        va = validate_format(rep.text)
        status = ("already_ok" if va and rep.text.strip() == c.raw.strip()
                  else "repaired_success" if va
                  else "still_broken")
        rows.append(Row(raw=c.raw, repaired=rep.text, tags=rep.tags, status=status))

    # 저장
    out_csv = os.path.join(out_dir, "results.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["raw","repaired","tags","status"])
        for r in rows:
            w.writerow([r.raw, r.repaired, ";".join(r.tags), r.status])

    succ_jsonl = os.path.join(out_dir, "success.jsonl")
    fail_jsonl = os.path.join(out_dir, "failure.jsonl")
    with open(succ_jsonl, "w", encoding="utf-8") as fs, open(fail_jsonl, "w", encoding="utf-8") as ff:
        for r in rows:
            obj = {"raw": r.raw, "repaired": r.repaired, "tags": r.tags, "status": r.status}
            (fs if r.status in {"already_ok","repaired_success"} else ff).write(json.dumps(obj, ensure_ascii=False) + "\n")

    total = len(rows)
    valid_after  = sum(1 for r in rows if validate_format(r.repaired))
    repaired_success = sum(1 for r in rows if r.status == "repaired_success")
    still_broken = sum(1 for r in rows if r.status == "still_broken")
    already_ok = sum(1 for r in rows if r.status == "already_ok")
    summary = f"""# FunctionCall Repair Test Summary
- total: {total}
- valid_after: {valid_after}
- repaired_success: {repaired_success}
- already_ok: {already_ok}
- still_broken: {still_broken}
files:
- {os.path.relpath(out_csv)}
- {os.path.relpath(succ_jsonl)}
- {os.path.relpath(fail_jsonl)}
func_schema: {args.func_schema or "(none)"} | arg_schema: {args.arg_schema or "(none)"}
"""
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary)
    print(summary); print(f"[OK] saved to: {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=None, help="assets/repair/repair_test.csv|jsonl")
    ap.add_argument("--outdir", type=str, default="assets/repair", help="결과 저장 루트 폴더")
    ap.add_argument("--max_rounds", type=int, default=2, help="보정 반복 라운드")
    ap.add_argument("--init-sample", action="store_true", help="입력 파일이 없으면 샘플 생성")
    ap.add_argument("--func-schema", type=str, default="assets/repair/function_schema.yaml", help="함수 스키마 YAML 경로")
    ap.add_argument("--arg-schema",  type=str, default="assets/repair/arg_schema.yaml",      help="인자 스키마 YAML 경로")
    run(ap.parse_args())
