# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict

TRACE = False  # set True via --trace to see rule matching details

@dataclass
class Call:
    func: str
    args: Dict[str, Any]

# -*- coding: utf-8 -*-

import re
from typing import Any, Dict, List, Optional

_BOOL_T = {"true"}
_BOOL_F = {"false"}

def _to_bool(s: str) -> Optional[bool]:
    t = (s or "").strip().lower()
    if t in _BOOL_T:
        return True
    if t in _BOOL_F:
        return False
    return None

def _to_num(s: str):
    try:
        return int(s)
    except Exception:
        try:
            return float(s)
        except Exception:
            return None

def parse_kv_args(text: str) -> Dict[str, Any]:
    s = (text or "").strip()
    if not s:
        return {}
    out: Dict[str, Any] = {}
    for part in [p.strip() for p in s.split(",") if p.strip()]:
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k, v = k.strip(), v.strip()
        # keep quoted scalars as strings (do not numeric-cast)
        quoted = (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'"))
        if quoted:
            v = v[1:-1]
        # prevent accidental numeric cast for non-ASCII words like '하남'
        if not quoted and all(ch.isdigit() or ch in "+-." for ch in v):
            pass  # allow numeric parsing below
        elif not quoted and isinstance(v, str) and not v.isascii():
            out[k] = v
            continue
        if not quoted:
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

_TAG_RE = re.compile(r"<function_([A-Za-z0-9_]+)>\((.*?)\)", re.S)

def parse_function_tags(text: str) -> List[Call]:
    if not text:
        return []
    s = text.replace("<end>", "")
    calls: List[Call] = []
    for name, arg_text in _TAG_RE.findall(s):
        args = parse_kv_args(arg_text)
        calls.append(Call(func=name, args=args))
    return calls

# -*- coding: utf-8 -*-

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

def _is_wild(v: Any) -> bool:
    return isinstance(v, str) and v == "*"

def _args_equal(inp: Dict[str, Any], rule: Dict[str, Any]) -> bool:
    """rule의 각 키를 inp가 만족하는지 검사. '*'는 와일드카드. (부분 매칭)"""
    inp = inp or {}
    for k, expect in (rule or {}).items():
        if k not in inp:
            return False
        if _is_wild(expect):
            continue
        if inp[k] != expect:
            return False
    # rule에 정의된 키 외에 더 있는 경우는 허용 (부분 매칭)
    return True

@dataclass
class _Pair:
    func: str
    args: Dict[str, Any]
    json_obj: Dict[str, Any]

class PairMapper:
    """
    pairs.jsonl 파일을 사용해 태그(Call) → JSON 객체로 매핑한다.
    JSONL 각 라인은 {"tag": "<function_X>(...)", "json": {...}} 형태여야 한다.
    """
    def __init__(self, pairs: List[_Pair]):
        self._by_func: Dict[str, List[_Pair]] = {}
        for p in pairs or []:
            self._by_func.setdefault(p.func, []).append(p)

    @classmethod
    def from_jsonl(cls, path: str) -> "PairMapper":
        pairs: List[_Pair] = []
        with Path(path).open("r", encoding="utf-8") as rf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                tag = obj.get("tag")
                json_obj = obj.get("json")
                if not isinstance(tag, str) or not isinstance(json_obj, dict):
                    continue
                calls = parse_function_tags(tag)
                if not calls:
                    continue
                c = calls[0]  # 한 줄에 한 개의 태그를 가정
                pairs.append(_Pair(func=c.func, args=c.args, json_obj=json_obj))
        return cls(pairs)

    def tag_to_json_list(self, llm_output: str) -> List[Dict[str, Any]]:
        calls = parse_function_tags(llm_output)
        out: List[Dict[str, Any]] = []
        for c in calls:
            cand = self._by_func.get(c.func) or []
            mapped = None
            for p in cand:
                if _args_equal(c.args, p.args):
                    mapped = p.json_obj
                    break
            if mapped is None:
                # 아이덴티티 폴백
                mapped = {"functionName": c.func, "arguments": c.args or {}}
            out.append(mapped)
        return out

# -*- coding: utf-8 -*-

import yaml  # type: ignore
from typing import Any, Dict, List

def _is_wild(v: Any) -> bool:
    return isinstance(v, str) and v == "*"

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

def _fmt_value(v: Any, vars_: Dict[str, Any]) -> Any:
    if isinstance(v, str):
        try:
            return v.format(**vars_)
        except Exception:
            return v
    return v

def _fmt_args(args: Dict[str, Any], vars_: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _fmt_value(v, vars_) for k, v in (args or {}).items()}

class CodeGroupedBidirectionalMapper:
    """
    function_map.yaml 형식 지원:
      function_XX:
        - when: { ... }
          new_name: ...
          args: { ... }   # 출력 JSON의 arguments로 사용
    forward만 사용 (태그 -> JSON). 역방향이 필요하면 PairMapper를 권장.
    """
    def __init__(self, rules: Dict[str, List[Dict[str, Any]]]):
        self.rules = rules or {}

    @classmethod
    def from_yaml(cls, path: str) -> "CodeGroupedBidirectionalMapper":
        doc = yaml.safe_load(open(path, "r", encoding="utf-8")) or {}

        # 경우 1) 루트에 function_map 키가 있는 구조
        if isinstance(doc, dict) and "function_map" in doc and isinstance(doc["function_map"], dict):
            doc = doc["function_map"]

        rules: Dict[str, List[Dict[str, Any]]] = {}
        if isinstance(doc, dict):
            for k, v in doc.items():
                if not isinstance(k, str):
                    continue
                # reverse_by_name 등은 무시
                if not k.startswith("function_"):
                    continue
                if isinstance(v, list):
                    rules[k] = v
        return cls(rules)

    def forward_call(self, call: Call) -> Dict[str, Any]:
        sect = f"function_{call.func}"
        if TRACE:
            logger.info(f"[TRACE] evaluating section={sect} call_args={call.args}")
        for rule in self.rules.get(sect, []):
            if TRACE:
                logger.info(f"[TRACE]  trying rule.when={rule.get('when')}")
            when = rule.get("when") or {}
            if _subset_match(call.args or {}, when):
                if TRACE:
                    logger.info(f"[TRACE]  matched rule → new_name={rule.get('new_name')} args={rule.get('args')}")
                vars_ = {"code": call.func, **(call.args or {})}
                return {
                    "functionName": rule.get("new_name") or call.func,
                    "arguments": _fmt_args(dict(rule.get("args") or {}), vars_),
                }
        if TRACE:
            logger.info(f"[TRACE]  no match in {sect}, falling back to identity")
        # 매칭 없으면 아이덴티티
        return {"functionName": call.func, "arguments": call.args or {}}

    def forward_llm_output(self, llm_output: str) -> List[Dict[str, Any]]:
        calls = parse_function_tags(llm_output)
        return [self.forward_call(c) for c in calls]

# -------------------------
# Runner: CSV → mapping → print/save
# -------------------------
import argparse
import csv
import logging

logger = logging.getLogger("function_parse")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

def _pick_header(row_keys, *candidates):
    lowered = {k.lower().strip(): k for k in row_keys}
    for cand in candidates:
        key = lowered.get(cand.lower().strip())
        if key:
            return key
    return None

def _stringify(obj):
    if isinstance(obj, str):
        return obj
    return json.dumps(obj, ensure_ascii=False)

class MapperFacade:
    def __init__(self, pairs_path: Optional[str] = None, yaml_path: Optional[str] = None):
        self.kind = None
        self.impl = None
        if pairs_path and Path(pairs_path).exists():
            self.impl = PairMapper.from_jsonl(pairs_path)
            self.kind = "pairs"
            logger.info(f"Loaded PairMapper from {pairs_path}")
        elif yaml_path and Path(yaml_path).exists():
            self.impl = CodeGroupedBidirectionalMapper.from_yaml(yaml_path)
            self.kind = "yaml"
            logger.info(f"Loaded YAML mapper from {yaml_path}")
        else:
            raise RuntimeError("매퍼 초기화 실패: --pairs 또는 --yaml 경로를 확인하세요.")

    def to_json_list(self, llm_output: str):
        if not llm_output:
            return []
        if self.kind == "pairs":
            return self.impl.tag_to_json_list(llm_output)
        return self.impl.forward_llm_output(llm_output)

def run(in_path: str, pairs_path: Optional[str], yaml_path: Optional[str],
        out_path: Optional[str], encoding: str = "utf-8", out_jsonl_path: Optional[str] = None) -> None:
    mapper = MapperFacade(pairs_path, yaml_path)
    with open(in_path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        idx_h = _pick_header(headers, "Index", "index", "id")
        q_h = _pick_header(headers, "Query(한글)", "Query", "query", "message")
        o_h = _pick_header(headers, "LLM Output", "llm output", "llm_output", "output")
        if not q_h or not o_h:
            raise RuntimeError(f"CSV 헤더를 찾을 수 없습니다. (헤더: {headers})")

        rows_out = []
        for row in reader:
            idx = row.get(idx_h) if idx_h else None
            query = (row.get(q_h) or "").strip()
            llm_out = (row.get(o_h) or "").strip()
            try:
                llm_out_json = mapper.to_json_list(llm_out)
            except Exception as e:
                logger.warning(f"매핑 실패(Index={idx}): {e}")
                llm_out_json = []

            rows_out.append({
                "Index": idx,
                "Query(한글)": query,
                "LLM Output": llm_out,
                "LLM Output JSON": llm_out_json,
            })

            print("Query :", query)
            print("LLM Output :", llm_out)
            print("LLM Output JSON :", _stringify(llm_out_json))
            print("-" * 60)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        if out_path.lower().endswith(".csv"):
            with open(out_path, "w", encoding=encoding, newline="") as wf:
                w = csv.DictWriter(wf, fieldnames=["Index", "Query(한글)", "LLM Output", "LLM Output JSON"])
                w.writeheader()
                for r in rows_out:
                    r2 = dict(r)
                    r2["LLM Output JSON"] = _stringify(r["LLM Output JSON"])
                    w.writerow(r2)
            logger.info(f"Wrote CSV: {out_path}")
        elif out_path.lower().endswith(".jsonl"):
            with open(out_path, "w", encoding=encoding) as wf:
                for r in rows_out:
                    wf.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
            logger.info(f"Wrote JSONL: {out_path}")
        else:
            with open(out_path, "w", encoding=encoding) as wf:
                json.dump(rows_out, wf, ensure_ascii=False, indent=2)
            logger.info(f"Wrote JSON: {out_path}")

    if out_jsonl_path:
        Path(out_jsonl_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_jsonl_path, "w", encoding=encoding) as wf:
            for r in rows_out:
                # expected: array JSON (serialized as string), because a row can map to multiple function calls
                expected_list = r["LLM Output JSON"] if isinstance(r.get("LLM Output JSON"), list) else []
                record = {
                    "message": r["Query(한글)"],
                    "expected": json.dumps(expected_list, ensure_ascii=False)
                }
                wf.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Wrote out_jsonl: {out_jsonl_path}")

def main():
    ap = argparse.ArgumentParser(description="CSV의 LLM Output을 파싱해 JSON 표준 포맷으로 매핑합니다.")
    ap.add_argument("--in", dest="inp", required=True, help="입력 CSV 경로")
    ap.add_argument("--pairs", default=None, help="pairs.jsonl 경로")
    ap.add_argument("--yaml", default=None, help="function_map.yaml 경로")
    ap.add_argument("--out", default=None, help="결과 저장 경로(.csv | .jsonl | .json)")
    ap.add_argument("--encoding", default="utf-8", help="입출력 인코딩(기본: utf-8)")
    ap.add_argument("--trace", action="store_true", help="룰 매칭 상세 로깅")
    ap.add_argument("--out_jsonl", default=None, help="jsonl 저장 (message/expected 필드만)")
    args = ap.parse_args()
    global TRACE
    TRACE = args.trace
    if TRACE:
        logger.setLevel(logging.DEBUG)
    run(args.inp, args.pairs, args.yaml, args.out, encoding=args.encoding, out_jsonl_path=args.out_jsonl)

if __name__ == "__main__":
    main()