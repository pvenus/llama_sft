# -*- coding: utf-8 -*-
"""
function_parser.py — LLM Output 태그 파싱 → records 생성 (+ 선택적 규칙 치환)

입력(원본 파일: CSV/JSON/JSONL)
- CSV/JSON/JSONL 모두 'Query(한글)'와 'LLM Output' 컬럼(키)을 갖는다고 가정.
  예)
    "Query(한글)": "월요일 자카르타 대기질을 화면에 보여줘...",
    "LLM Output": "<function_IO>(timeframe=3, location=자카르타);<function_MR>()<end>"

출력(records): JSON/JSONL (확장자에 따라)
[
  {
    "query": "...",
    "calls": [
      {"func": "IO", "args": {"timeframe": 3, "location": "자카르타"}},
      {"func": "MR", "args": {}}
    ]
  },
  ...
]

옵션
- --rules + --rule-format (simple|by_func) + --direction (forward|reverse)
  규칙 치환(정/역변환)을 records.calls에 적용.
  * simple: {raw_func, when, new_func, args} 목록(JSONL/YAML/JSON)
  * by_func: 기존 function_rules.yaml 포맷(rules_by_func / rules / mappings)
"""
from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import re
import sys
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# =========================
# k=v 문자열 파서 (args)
# =========================
_BOOL_T = {"true"}
_BOOL_F = {"false"}

def _to_bool(s: str) -> Optional[bool]:
    t = (s or "").strip().lower()
    if t in _BOOL_T: return True
    if t in _BOOL_F: return False
    return None

def _to_num(s: str) -> Optional[int | float]:
    try:
        return int(s)
    except Exception:
        try:
            return float(s)
        except Exception:
            return None

def _parse_args_from_text(text: str) -> Dict[str, Any]:
    """
    'a=1, b=true, name=자카르타' → {'a':1, 'b':True, 'name':'자카르타'}
    범위/특수표현(예: '0~5', '지역|0(현재)')은 문자열로 유지.
    """
    s = (text or "").strip()
    if not s:
        return {}
    out: Dict[str, Any] = {}
    for part in [p.strip() for p in s.split(",") if p.strip()]:
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k, v = k.strip(), v.strip()
        bv = _to_bool(v)
        if bv is not None:
            out[k] = bv
            continue
        nv = _to_num(v)
        if nv is not None:
            out[k] = nv
            continue
        out[k] = v
    return out

# =========================
# <function_...>(...) 태그 파서
# =========================
_TAG_RE = re.compile(r"<function_([A-Za-z0-9_]+)>\((.*?)\)", re.S)

def parse_function_tags(text: str) -> List[Dict[str, Any]]:
    """
    '<function_IO>(timeframe=3, location=자카르타);<function_MR>()<end>'
      -> [{'func':'IO','args':{'timeframe':3,'location':'자카르타'}},
          {'func':'MR','args':{}}]
    """
    if not text:
        return []
    s = text.replace("<end>", "")
    calls: List[Dict[str, Any]] = []
    for code, arg_text in _TAG_RE.findall(s):
        args = _parse_args_from_text(arg_text)
        calls.append({"func": code, "args": args})
    return calls


# =========================
# 입력 리더 (CSV/JSON/JSONL)
# =========================
def _read_from_csv(path: str) -> List[Dict[str, Any]]:
    """
    CSV → [{"query": <Query(한글)>, "calls": [ {func, args}, ... ]}, ...]
    - 인코딩: utf-8-sig
    - 헤더: "Query(한글)", "LLM Output" (대소문자 무시)
    - Query 비어 있으면 스킵
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    out: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8-sig", newline="") as rf:
        reader = csv.DictReader(rf)
        headers = reader.fieldnames or []
        low = {h.lower(): h for h in headers}

        qh = low.get("query(한글)") or low.get("query") or low.get("message")
        eh = low.get("llm output") or low.get("expected")

        for row in reader:
            query = (row.get(qh, "") if qh else "").strip()
            if not query:
                continue
            llm = (row.get(eh, "") if eh else "").strip()
            calls = parse_function_tags(llm)
            out.append({"query": query, "calls": calls})
    return out


def _read_from_json(path: str) -> List[Dict[str, Any]]:
    """
    JSON/JSONL → [{"query": <Query(한글)>, "calls": [ {func, args}, ... ]}, ...]
    - 각 객체에 "Query(한글)" 과 "LLM Output" 키가 직접 존재한다고 가정
    - 이미 {"query","calls"} 형태라면 그대로 수용
    - Query 비어있으면 스킵
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    def _norm(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # 이미 records 형태면 그대로 통과
        if "query" in obj and "calls" in obj:
            calls = obj.get("calls") or []
            # calls 정규화: code -> func
            fixed: List[Dict[str, Any]] = []
            for c in calls:
                if not isinstance(c, dict):
                    continue
                fn = c.get("func") or c.get("code")
                fixed.append({"func": str(fn or ""), "args": c.get("args") or {}})
            return {"query": str(obj.get("query") or "").strip(), "calls": fixed}

        # flat 형태 처리
        query = str(obj.get("Query(한글)") or obj.get("query") or obj.get("message") or "").strip()
        if not query:
            return None
        llm = str(obj.get("LLM Output") or obj.get("expected") or "").strip()
        return {"query": query, "calls": parse_function_tags(llm)}

    out: List[Dict[str, Any]] = []
    if p.suffix.lower() == ".jsonl":
        with p.open("r", encoding="utf-8") as rf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    rec = _norm(obj)
                    if rec:
                        out.append(rec)
        return out

    with p.open("r", encoding="utf-8") as rf:
        data = json.load(rf)
    if isinstance(data, list):
        for obj in data:
            if isinstance(obj, dict):
                rec = _norm(obj)
                if rec:
                    out.append(rec)
    elif isinstance(data, dict):
        rec = _norm(data)
        if rec:
            out.append(rec)
    return out


def read_records(path: str) -> List[Dict[str, Any]]:
    """입력 경로 확장자에 따라 CSV/JSON/JSONL 자동 판별."""
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return _read_from_csv(path)
    return _read_from_json(path)

# =========================
# records 후처리: 같은 query끼리 calls 합치기
# =========================
def group_by_query(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for r in records:
        q = r.get("query", "")
        if not q:
            continue
        calls = r.get("calls") or []
        if q not in grouped:
            grouped[q] = {"query": q, "calls": []}
        # calls를 func/args 정규화하여 추가
        for c in calls:
            if not isinstance(c, dict):
                continue
            fn = c.get("func") or c.get("code") or ""
            grouped[q]["calls"].append({"func": str(fn), "args": c.get("args") or {}})
    return list(grouped.values())

# =========================
# 규칙 모델/유틸 + 로더 + 치환
# =========================
@dataclass
class Rule:
    raw_func: str              # 예) "BP"
    when: Dict[str, Any]       # 예) {"enable":0,"type":-1}
    new_func: str              # 예) "mode" 또는 기존 map_func
    args: Dict[str, Any]       # 예) {"target":"AI_MODE","enable":"on"} 또는 {"enum":"AI_Mode"}

def _is_wild(v: Any) -> bool:
    return isinstance(v, str) and v in {"*", "__ANY__"}

def _subset_match(base: Dict[str, Any], cond: Dict[str, Any]) -> bool:
    base = base or {}
    for k, expect in (cond or {}).items():
        if k not in base:
            return False
        if _is_wild(expect):
            continue
        if base.get(k) != expect:
            return False
    return True

# 아직 사용 불가
# def load_rules_v2(path: str) -> list[Rule]:
#     """
#     simple 포맷: 각 항목이 {raw_func, when, new_func, args}
#     - JSONL / YAML / JSON 모두 지원
#     """
#     p = Path(path)
#     text = p.read_text(encoding="utf-8")
# 
#     # JSONL 우선
#     items: list[dict] | None = []
#     saw_line = False
#     for line in text.splitlines():
#         s = line.strip()
#         if not s:
#             continue
#         saw_line = True
#         try:
#             obj = json.loads(s)
#         except Exception:
#             items = None
#             break
#         if not isinstance(obj, dict):
#             items = None
#             break
#         items.append(obj)
# 
#     # YAML/JSON
#     if items is None or (saw_line and not items):
#         if yaml is not None:
#             docs = list(yaml.safe_load_all(io.StringIO(text)))
#         else:  # yaml 미설치 시 JSON만 시도
#             docs = [json.loads(text)]
#         flat: list = []
#         for d in docs:
#             if d is None:
#                 continue
#             if isinstance(d, list): flat.extend(d)
#             else: flat.append(d)
#         items = [it for it in flat if isinstance(it, dict)]
# 
#     rules: list[Rule] = []
#     for it in (items or []):
#         rules.append(Rule(
#             raw_func=str(it.get("raw_func") or "").strip(),
#             when=dict(it.get("when") or {}),
#             new_func=str(it.get("new_func") or "").strip(),
#             args=dict(it.get("args") or {}),
#         ))
#     return rules


def load_rules_v1(path: str) -> list[Rule]:
    """
    기존 포맷 지원:
      rules_by_func:
        <map_func>:
          <map_enum>: { code: BP, enable: 0, type: -1 }  # dict | list[dict]
      또는 flat rules/mappings 배열
    → Rule(raw_func=code, when=..., new_func=map_func, args={'enum': map_enum})
    """
    if yaml is None:
        raise RuntimeError("by_func 포맷을 사용하려면 PyYAML이 필요합니다. pip install pyyaml")
    p = Path(path)
    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    out: list[Rule] = []

    if "rules_by_func" in doc:
        by_func = doc.get("rules_by_func") or {}
        for map_func, enums in by_func.items():
            for map_enum, spec in (enums or {}).items():
                if isinstance(spec, list):
                    for item in (spec or []):
                        if not isinstance(item, dict):
                            continue
                        code = (item.get("code") or item.get("func") or "").strip()
                        if not code:
                            continue
                        when = item.get("when")
                        when = dict(when) if isinstance(when, dict) else {k: v for k, v in item.items() if k not in {"code","func","when"}}
                        out.append(Rule(code, when, map_func, {"enum": map_enum}))
                elif isinstance(spec, dict):
                    code = (spec.get("code") or spec.get("func") or "").strip()
                    if not code:
                        continue
                    when = spec.get("when")
                    when = dict(when) if isinstance(when, dict) else {k: v for k, v in spec.items() if k not in {"code","func","when"}}
                    out.append(Rule(code, when, map_func, {"enum": map_enum}))
        return out

    # flat rules/mappings
    flat = doc.get("rules") or doc.get("mappings") or []
    for it in (flat or []):
        if not isinstance(it, dict):
            continue
        map_func = str(it.get("map_func") or "").strip()
        map_enum = str(it.get("map_enum") or "").strip()
        code = str(it.get("code") or it.get("func") or "").strip()
        when = it.get("when")
        when = dict(when) if isinstance(when, dict) else {k: v for k, v in it.items() if k not in {"map_func","map_enum","code","func","when"}}
        if map_func and map_enum and code:
            out.append(Rule(code, when, map_func, {"enum": map_enum}))
    return out


def load_rules(path: str, rule_format: str) -> list[Rule]:
    """rule_format: 'v1' | 'v2'"""
    if rule_format == "v1":
        return load_rules_v1(path)
    # if rule_format == "v2":
    #     return load_rules_v2(path)

    raise ValueError(f"Unsupported rule_format: {rule_format}")

# =========================
# 변환 함수 (dict in → dict out)
# =========================
def convert_forward(func: str, args: dict, rules: list[Rule]) -> dict:
    """(raw func,args) → (new func,args) dict 반환. 미매칭 시 입력 그대로."""
    func = func or ""
    args = args or {}
    for r in rules:
        if func == r.raw_func and _subset_match(args, r.when):
            return {"func": r.new_func, "args": copy.deepcopy(r.args)}
    return {"func": func, "args": copy.deepcopy(args)}


def convert_reverse(func: str, args: dict, rules: list[Rule]) -> dict:
    """(new func,args) → (raw func,args) dict 반환. 미매칭 시 입력 그대로."""
    func = func or ""
    args = args or {}
    for r in rules:
        if func == r.new_func and _subset_match(args, r.args):
            return {"func": r.raw_func, "args": copy.deepcopy(r.when)}
    return {"func": func, "args": copy.deepcopy(args)}


def apply_rules_to_records(records: list[dict], rules: list[Rule], direction: str = "forward") -> list[dict]:
    """
    records: [{"query":"...", "calls":[{"code"/"func":..., "args":{...}}, ...]}, ...]
    반환   : [{"query":"...", "calls":[{"func":..., "args":{...}}, ...]}, ...]
    """
    out: list[dict] = []
    for rec in (records or []):
        q = rec.get("query", "")
        calls = rec.get("calls") or []
        new_calls: list[dict] = []
        for c in calls:
            in_func = c.get("func") or c.get("code") or ""
            in_args = c.get("args") or {}
            if direction == "reverse":
                new_calls.append(convert_reverse(in_func, in_args, rules))
            else:
                new_calls.append(convert_forward(in_func, in_args, rules))
        out.append({"query": q, "calls": new_calls})
    return out

# =========================
# calls 적용기 (리스트 변환)
# =========================
def apply_forward_to_calls(calls: List[Dict[str, Any]], rules: List[Rule]) -> List[Dict[str, Any]]:
    """
    calls: [{'code'|'func': str, 'args': dict}, ...] → [{'func': str, 'args': dict}, ...]
    """
    out: List[Dict[str, Any]] = []
    for c in calls or []:
        in_func = c.get("func") or c.get("code") or ""
        in_args = c.get("args") or {}
        out.append(convert_forward(in_func, in_args, rules))
    return out

def apply_reverse_to_calls(calls: List[Dict[str, Any]], rules: List[Rule]) -> List[Dict[str, Any]]:
    """
    calls: [{'func': str, 'args': dict}, ...] → [{'func': str, 'args': dict}]  (원시 형태로 복원)
    """
    out: List[Dict[str, Any]] = []
    for c in calls or []:
        in_func = c.get("func") or ""
        in_args = c.get("args") or {}
        out.append(convert_reverse(in_func, in_args, rules))
    return out

# =========================
# I/O: records(JSON/JSONL) 읽고/쓰기
# =========================
def _read_records(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    out: List[Dict[str, Any]] = []
    if p.suffix.lower() == ".jsonl":
        with p.open("r", encoding="utf-8") as rf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    out.append(obj)
        return out
    # .json
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _write_records(path: Optional[str], records: List[Dict[str, Any]]) -> None:
    if not path:
        json.dump(records, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".jsonl":
        with p.open("w", encoding="utf-8") as wf:
            for r in records:
                wf.write(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + "\n")
    else:
        with p.open("w", encoding="utf-8") as wf:
            json.dump(records, wf, ensure_ascii=False, indent=2)


# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser(
        description="Parse LLM Output tags into records and optionally apply function-substitution rules.")

    # ↓↓↓ 기본값을 요청대로 설정 (옵션 없이 실행해도 동작)
    ap.add_argument("--in", dest="inp",
                    default="assets/convert/func_samples_10.csv",
                    help="input CSV/JSON/JSONL path (default: assets/convert/func_samples_10.csv)")
    ap.add_argument("--out",
                    default="assets/prompt/test_convert_msg.jsonl",
                    help="output records path (.json or .jsonl) (default: assets/prompt/test_convert_msg.jsonl)")
    ap.add_argument("--rules",
                    default="assets/convert/function_rules.yaml",
                    help="rules file path (default: assets/convert/function_rules.yaml)")
    ap.add_argument("--rule-format", choices=["simple", "by_func", "v1"],
                    default="v1",
                    help="rules format type (default: v1 == by_func)")
    ap.add_argument("--direction", choices=["forward", "reverse"],
                    default="forward",
                    help="substitution direction (default: forward)")
    ap.add_argument("--no-group", action="store_true", help="do not group by query (rarely needed)")
    args = ap.parse_args()

    # 1) 원본 읽기 → records (query/calls)
    records = read_records(args.inp)

    # 2) 같은 query 묶기(기본 on)
    if not args.no_group:
        records = group_by_query(records)

    # 3) 규칙 치환(항상 수행: 기본값 세팅)
    rules = load_rules(args.rules, args.rule_format)
    records = apply_rules_to_records(records, rules, direction=args.direction)

    # 4) 출력
    _write_records(args.out, records)


if __name__ == "__main__":
    main()