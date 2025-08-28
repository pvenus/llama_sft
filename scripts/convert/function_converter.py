#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
function_converter.py

역할
- 입력 CSV(func, raw_args, query ...)와 규칙(config YAML)을 이용해
  map_func / map_enum / map_args를 생성하여 변환 CSV 출력
- (신규) 변환 결과를 이용해 message/expected(JSON 문자열) JSONL 생성
  * --emit-test-jsonl : test.jsonl 과 동일 스키마
    - 각 라인: {"message": <query>, "expected": "{\"name\":\"<map_func>\",\"arguments\":{...}}"}
    - arguments 는 map_args 를 k=v 형태에서 파싱(숫자/불리언 자동 캐스팅)

규칙(요약)
- 규칙 루트: rules_by_func (권장) / rules (대체) / mappings (레거시 별칭)
- 와일드카드: "*" 또는 "__ANY__" → 해당 키가 "존재"하면 값 무관
- 변환 CSV 컬럼(고정): func, raw_args, query, map_func, map_enum, map_args
- 인/아웃 인코딩 기본: utf-8-sig (Excel 한글 호환)

예)
PowerShell:
  python -m scripts.convert.function_converter `
    --in assets/convert/func_samples_10.csv `
    --out assets/convert/func_samples_10.converted.csv `
    --config assets/convert/function_rules.yaml `
    --emit-test-jsonl outputs/datasets/test_convert_function.jsonl

이미 변환된 CSV만 JSONL로 바꾸고 싶다면:
  python -m scripts.convert.function_converter `
    --from-converted assets/convert/func_samples_10.converted.csv `
    --emit-test-jsonl outputs/datasets/test_convert_function.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


# =============================
# 공통 유틸
# =============================
_BOOL_TRUE = {"true", "t", "yes", "y", "1"}
_BOOL_FALSE = {"false", "f", "no", "n", "0"}

def _to_bool(s: str) -> Optional[bool]:
    t = (s or "").strip().lower()
    if t in _BOOL_TRUE:
        return True
    if t in _BOOL_FALSE:
        return False
    return None

def _to_num(s: str) -> Optional[int | float]:
    try:
        return int(s)
    except Exception:
        try:
            return float(s)
        except Exception:
            return None

def parse_raw_args(text: str) -> Dict[str, Any]:
    """
    'a=1, b=true, name=foo' → {'a':1,'b':True,'name':'foo'}
    범위/특수표현(예: '0~5', '지역명|0(현재지역)')은 문자열로 그대로 둔다.
    """
    text = (text or "").strip()
    if not text:
        return {}
    out: Dict[str, Any] = {}
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        if "=" not in part:
            # 위치 인자 스타일은 스킵(필요시 확장)
            continue
        k, v = part.split("=", 1)
        k, v = k.strip(), v.strip()
        bv = _to_bool(v)
        if bv is not None:
            out[k] = bv; continue
        nv = _to_num(v)
        if nv is not None:
            out[k] = nv; continue
        out[k] = v
    return out

def dict_to_argtext(d: Dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in d.items())


# =============================
# 규칙 로딩/정규화
# =============================
def _load_yaml_or_json(path: Path) -> dict:
    if path.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("PyYAML이 필요합니다. pip install pyyaml") from e
        with path.open("r", encoding="utf-8") as rf:
            return yaml.safe_load(rf) or {}
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as rf:
            return json.load(rf) or {}
    else:
        raise ValueError(f"지원하지 않는 설정 파일 형식: {path}")

@dataclass
class RuleEntry:
    map_func: str
    map_enum: str
    code: str
    when: Dict[str, Any]

def _normalize_rules(doc: dict) -> List[RuleEntry]:
    """
    지원 스키마
    1) rules_by_func (권장)
       rules_by_func:
         <map_func>:
           <map_enum>: { code: BP, enable: 0, type: -1 }      # dict 1건
           <map_enum>:
             - { code: CE }                                   # list[dict] 여러 건
             - { code: IO, timeframe: 0, location: "*" }
    2) rules / mappings (대체, 평탄화 배열)
       - { map_func: ..., map_enum: ..., code: ..., when: {...} }

    반환: RuleEntry 리스트(한 규칙=한 Row)
    """
    out: List[RuleEntry] = []

    if "rules_by_func" in doc:
        by_func = doc.get("rules_by_func") or {}
        for map_func, enums in by_func.items():
            enums = enums or {}
            for map_enum, spec in enums.items():
                # 케이스 A: list[dict]
                if isinstance(spec, list):
                    for item in (spec or []):
                        if not isinstance(item, dict):
                            continue
                        code = (item.get("code") or item.get("func") or "").strip()
                        if not code:
                            continue
                        # item에 when이 중첩돼 있을 수도 있고, 키가 flat일 수도 있음
                        when_dict = item.get("when")
                        if isinstance(when_dict, dict):
                            when = dict(when_dict)
                        else:
                            when = {k: v for k, v in item.items() if k not in {"code", "func", "when"}}
                        out.append(RuleEntry(map_func=map_func, map_enum=map_enum, code=code, when=when))
                # 케이스 B: dict 1건
                elif isinstance(spec, dict):
                    code = (spec.get("code") or spec.get("func") or "").strip()
                    if not code:
                        continue
                    when_dict = spec.get("when")
                    if isinstance(when_dict, dict):
                        when = dict(when_dict)
                    else:
                        when = {k: v for k, v in spec.items() if k not in {"code", "func", "when"}}
                    out.append(RuleEntry(map_func=map_func, map_enum=map_enum, code=code, when=when))
                # 그 외 타입은 무시
        return out

    # 평탄화 배열 스키마 (rules / mappings)
    flat_arr = doc.get("rules") or doc.get("mappings") or []
    for it in (flat_arr or []):
        it = it or {}
        map_func = (it.get("map_func") or "").strip()
        map_enum = (it.get("map_enum") or "").strip()
        code = (it.get("code") or it.get("func") or "").strip()
        if not (map_func and map_enum and code):
            continue
        when_dict = it.get("when")
        if isinstance(when_dict, dict):
            when = dict(when_dict)
        else:
            when = {k: v for k, v in it.items() if k not in {"map_func", "map_enum", "code", "func", "when"}}
        out.append(RuleEntry(map_func=map_func, map_enum=map_enum, code=code, when=when))
    return out

# =============================
# 매칭
# =============================
def _wildcard_ok(v: Any) -> bool:
    return isinstance(v, str) and v in {"*", "__ANY__"}

def match_row(rule: RuleEntry, func_code: str, raw_args_dict: Dict[str, Any]) -> bool:
    if (func_code or "").strip() != (rule.code or "").strip():
        return False
    for k, expect in (rule.when or {}).items():
        if k not in raw_args_dict:
            return False
        if _wildcard_ok(expect):
            continue
        if raw_args_dict.get(k) != expect:
            return False
    return True


# =============================
# 변환 실행 (CSV→CSV)
# =============================
def convert_rows(reader: csv.DictReader, rules: List[RuleEntry]) -> List[Dict[str, str]]:
    out_rows: List[Dict[str, str]] = []
    for row in reader:
        func_code = (row.get("func") or row.get("code") or "").strip()
        raw_args = (row.get("raw_args") or "").strip()
        query = (row.get("query") or row.get("Query(한글)") or row.get("Query") or "").strip()

        raw_dict = parse_raw_args(raw_args)

        map_func, map_enum = "", ""
        map_args_dict: Dict[str, Any] = {}
        for r in rules:
            if match_row(r, func_code, raw_dict):
                map_func, map_enum = r.map_func, r.map_enum
                # 안전하게: 규칙 when 키만 출력 args로
                map_args_dict = {k: raw_dict[k] for k in r.when.keys() if k in raw_dict}
                break

        map_args_text = dict_to_argtext(map_args_dict)
        out_rows.append({
            "func": func_code,
            "raw_args": raw_args,
            "query": query,
            "map_func": map_func,
            "map_enum": map_enum,
            "map_args": map_args_text,
        })
    return out_rows

def write_converted_csv(rows: List[Dict[str, str]], outp: Path, encoding_out: str = "utf-8-sig") -> None:
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding=encoding_out, newline="") as wf:
        w = csv.DictWriter(wf, fieldnames=["func", "raw_args", "query", "map_func", "map_enum", "map_args"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


# =============================
# (신규) 변환본 CSV → test_convert_function.jsonl
# =============================
def emit_test_jsonl_from_converted(
    converted_csv: Path, out_jsonl: Path, encoding_in: str = "utf-8-sig"
) -> dict:
    """
    func_samples_10.converted.csv → test_convert_function.jsonl

    라인 스키마(테스트셋과 동일):
      {"message": <query>,
       "expected": "{\"name\":\"<map_func>\",\"arguments\":{...}}"}   # expected는 JSON 문자열
    """
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    empty_func = 0

    with converted_csv.open("r", encoding=encoding_in, newline="") as rf, \
            out_jsonl.open("w", encoding="utf-8") as wf:
        reader = csv.DictReader(rf)
        for row in reader:
            message = (row.get("query") or row.get("Query") or row.get("Query(한글)") or "").strip()
            map_func = (row.get("map_func") or "").strip()
            map_args_text = (row.get("raw_args") or "").strip()

            # 🔽 추가: 원본 func(code) 가져오기
            func_code = (row.get("func") or row.get("code") or "").strip()

            # 기존: map_args 파싱
            args_obj = parse_raw_args(map_args_text)  # 숫자/불리언 캐스팅

            # 🔽 변경: arguments = {"code": func} 를 먼저 넣고 이후 map_args 병합(순서 유지)
            arguments = {"code": func_code}
            arguments.update(args_obj)

            if map_func == "":
                empty_func += 1

            # expected에 name/map_func, arguments/위에서 만든 dict
            expected_obj = {"name": map_func, "arguments": arguments}
            expected_str = json.dumps(expected_obj, ensure_ascii=False, separators=(',', ':'))
            line = {
                "message": message,
                "expected": expected_str,
            }
            wf.write(json.dumps(line, ensure_ascii=False, separators=(',', ':')) + "\n")
            total += 1

    return {"total": total, "empty_map_func": empty_func, "out": str(out_jsonl)}


# =============================
# 메인
# =============================
def main():
    ap = argparse.ArgumentParser(description="Convert CSV with rules and optionally emit test.jsonl-format JSONL.")
    ap.add_argument("--in", dest="inp", help="입력 CSV (원본)")
    ap.add_argument("--out", dest="out_csv", help="변환 CSV 출력 경로 (converted.csv)")
    ap.add_argument("--config", help="규칙 YAML/JSON 경로")
    ap.add_argument("--encoding-in", default="utf-8-sig")
    ap.add_argument("--encoding-out", default="utf-8-sig")

    # 신규: 이미 변환된 CSV만 JSONL로 바꿀 때
    ap.add_argument("--from-converted", help="이미 변환된 CSV 경로를 지정하면, 변환 없이 JSONL만 생성")

    # 신규: test.jsonl 동일 스키마 JSONL 출력
    ap.add_argument("--emit-test-jsonl", help="생성할 JSONL 경로 (예: outputs/datasets/test_convert_function.jsonl)")

    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # Case A) 변환 CSV 생성 + (옵션) JSONL 생성
    if args.inp and args.out_csv and args.config:
        # 규칙 로딩
        rules_doc = _load_yaml_or_json(Path(args.config))
        rules = _normalize_rules(rules_doc)

        # 변환 실행
        with Path(args.inp).open("r", encoding=args.encoding_in, newline="") as rf:
            reader = csv.DictReader(rf)
            rows = convert_rows(reader, rules)

        write_converted_csv(rows, Path(args.out_csv), encoding_out=args.encoding_out)

        if args.verbose:
            blanks = sum(1 for r in rows if not r["map_func"])
            print(f"[convert] total={len(rows)}, matched={len(rows)-blanks}, unmatched={blanks}")
            if args.emit_test_jsonl:
                print(f"[emit] writing JSONL → {args.emit_test_jsonl}")

        # JSONL까지 요구되면, 방금 만든 rows를 그대로 사용
        if args.emit_test_jsonl:
            stats = emit_test_jsonl_from_converted(
                converted_csv=Path(args.out_csv),
                out_jsonl=Path(args.emit_test_jsonl),
                encoding_in=args.encoding_out,
            )
            if args.verbose:
                print(f"[emit] {json.dumps(stats, ensure_ascii=False)}")
        return

    # Case B) 변환 없이, 기존 변환본 CSV만 JSONL로
    if args.from_converted and args.emit_test_jsonl:
        stats = emit_test_jsonl_from_converted(
            converted_csv=Path(args.from_converted),
            out_jsonl=Path(args.emit_test_jsonl),
            encoding_in=args.encoding_in,
        )
        if args.verbose:
            print(f"[emit-only] {json.dumps(stats, ensure_ascii=False)}")
        return

    # 안내
    ap.print_help()


if __name__ == "__main__":
    main()
