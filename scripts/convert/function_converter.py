# scripts/convert/function_converter.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import argparse, json, re
import pandas as pd

# =========================
# 1) 설정 로드 & 정규화
# =========================

def _load_yaml_or_json(path: Path) -> dict:
    if path.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml  # pip install pyyaml
        except Exception as e:
            raise RuntimeError("PyYAML이 필요합니다. `pip install pyyaml`") from e
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError("지원 형식: .yaml/.yml/.json")

def _normalize_from_rules_list(rules_in: list) -> Dict[str, List[Dict[str, Any]]]:
    """
    rules: [
      { code: BP, when: {...}, map: { func: mode_on, enum: AI_Mode } },
      ...
    ] -> by_code dict
    """
    by_code = defaultdict(list)
    for i, r in enumerate(rules_in):
        if not isinstance(r, dict):
            raise ValueError(f"rules[{i}] 항목은 dict 여야 합니다.")
        code = r.get("code") or r.get("name") or r.get("func_code")
        when = r.get("when") or r.get("conditions") or {}
        if not isinstance(when, dict):
            raise ValueError(f"rules[{i}].when 은 dict 여야 합니다. (got {type(when).__name__})")
        map_ = r.get("map") or {}
        if isinstance(map_, str) and "." in map_:
            func, enum = map_.split(".", 1)
        else:
            func = map_.get("func") or map_.get("func_name")
            enum = map_.get("enum") or map_.get("func_enum")
        if not code or not func or not enum:
            raise ValueError(f"rules[{i}]에 code/func/enum 누락")
        by_code[code].append({"func": func, "enum": enum, "when": dict(when)})
    return dict(by_code)

def _normalize_from_rules_by_func(rbf: dict) -> Dict[str, List[Dict[str, Any]]]:
    """
    rules_by_func:
      func_name:
        enum_name: dict_or_list   # dict={code, 조건...} 또는 list[dict]
      또는
      func_name:
        - { enum_name: dict_or_list }  # 단일키 딕셔너리 리스트도 허용
    -> by_code dict
    """
    by_code = defaultdict(list)

    def _iter_enum_items(container):
        if isinstance(container, dict):
            for enum_name, payload in container.items():
                yield enum_name, payload
        elif isinstance(container, list):
            for i, item in enumerate(container):
                if not isinstance(item, dict) or len(item) != 1:
                    raise ValueError(
                        "rules_by_func[*] 리스트 항목은 단일 키 딕셔너리여야 합니다. "
                        f"(index={i}, got={type(item).__name__})"
                    )
                (enum_name, payload), = item.items()
                yield enum_name, payload
        else:
            raise ValueError("rules_by_func 하위는 dict 또는 list여야 합니다.")

    for func_name, enum_container in rbf.items():
        for enum_name, payload in _iter_enum_items(enum_container):
            entries = payload if isinstance(payload, list) else [payload]
            for j, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    raise ValueError(
                        f"rules_by_func.{func_name}.{enum_name}[{j}] 는 dict 여야 합니다."
                    )
                entry = dict(entry)
                code = (
                    entry.pop("code", None)
                    or entry.pop("name", None)
                    or entry.pop("raw", None)
                    or entry.pop("func", None)
                )
                when = entry.pop("when", None)
                conditions = dict(when) if isinstance(when, dict) else entry
                if not code:
                    raise ValueError(
                        f"rules_by_func.{func_name}.{enum_name} 항목에 "
                        "code/name/raw/func 중 하나가 필요합니다."
                    )
                by_code[code].append({"func": func_name, "enum": enum_name, "when": conditions})
    return dict(by_code)

def _normalize_from_old_mappings(old: dict) -> Dict[str, List[Dict[str, Any]]]:
    """
    (옵션) legacy:
      mappings: { "BP": [ {func_name, func_enum, conditions}, ... ] }
    """
    by_code = defaultdict(list)
    for code, rules in old.items():
        if not isinstance(rules, list):
            raise ValueError(f"mappings.{code} 는 list 여야 합니다.")
        for r in rules:
            r = dict(r)
            func = r.get("func_name") or r.get("func")
            enum = r.get("func_enum") or r.get("enum")
            cond = dict(r.get("conditions") or {})
            if not func or not enum:
                raise ValueError(f"mappings.{code} 규칙에 func/enum 누락")
            by_code[code].append({"func": func, "enum": enum, "when": cond})
    return dict(by_code)

def load_config(config_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    반환: by_code = { "BP": [ {func, enum, when}, ... ], ... }
    rules / rules_by_func / (옵션) mappings 를 모두 지원
    """
    data = _load_yaml_or_json(config_path)
    if not isinstance(data, dict):
        raise ValueError("설정 파일 파싱 실패")

    # 흔한 오타/별칭도 허용
    if "mappings_by_func" in data and "rules_by_func" not in data:
        data["rules_by_func"] = data["mappings_by_func"]

    by_code_all = defaultdict(list)

    if isinstance(data.get("rules"), list):
        tmp = _normalize_from_rules_list(data["rules"])
        for k, v in tmp.items(): by_code_all[k].extend(v)

    if isinstance(data.get("rules_by_func"), dict):
        tmp = _normalize_from_rules_by_func(data["rules_by_func"])
        for k, v in tmp.items(): by_code_all[k].extend(v)

    if isinstance(data.get("mappings"), dict):
        tmp = _normalize_from_old_mappings(data["mappings"])
        for k, v in tmp.items(): by_code_all[k].extend(v)

    if not by_code_all:
        raise ValueError("설정에 'rules' 또는 'rules_by_func' (또는 'mappings') 중 하나가 필요합니다.")

    return dict(by_code_all)

# =========================
# 2) 매칭 로직
# =========================

BOOL_TRUE = {"true","True","TRUE"}
BOOL_FALSE = {"false","False","FALSE"}

def _coerce_value(s: str) -> Any:
    s = s.strip()
    if s in BOOL_TRUE: return True
    if s in BOOL_FALSE: return False
    if s == "xxxx": return "*"   # 기존 CSV 와일드카드 보정
    if re.fullmatch(r"-?\d+", s):
        try: return int(s)
        except ValueError: pass
    if re.fullmatch(r"-?\d+\.\d+", s):
        try: return float(s)
        except ValueError: pass
    return s

def parse_raw_args(arg_string: Optional[str]) -> Dict[str, Any]:
    if arg_string is None: return {}
    s = str(arg_string).strip()
    if s == "" or s.lower() == "nan": return {}
    out: Dict[str, Any] = {}
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk: continue
        if "=" in chunk:
            k, v = chunk.split("=", 1)
            out[k.strip()] = _coerce_value(v.strip())
        else:
            out[chunk] = True
    return out

def _want_ok(got: Any, want: Any) -> bool:
    # want="*" / "__ANY__" → 값 무관(키 존재만)
    return True if want in ("*","__ANY__") else (got == want)

def _conditions_match(got: Dict[str, Any], want: Dict[str, Any]) -> bool:
    for k, v in (want or {}).items():
        if k not in got: return False
        if not _want_ok(got[k], v): return False
    return True

def map_from_code(
    code: str,
    raw_args: Dict[str, Any],
    by_code: Dict[str, List[Dict[str, Any]]],
) -> Tuple[Optional[str], Optional[str], Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    (map_func, map_enum, matched_args, rule_when) 반환
    매칭 실패 시 (None, None, {}, None)
    """
    rules = by_code.get(code, [])
    for rule in rules:
        when = rule["when"]
        if _conditions_match(raw_args, when):
            matched = {k: raw_args.get(k) for k in when.keys() if k in raw_args}
            return rule["func"], rule["enum"], matched, when
    return None, None, {}, None

def map_args_to_string(matched_args: Dict[str, Any], when: Optional[Dict[str, Any]]) -> str:
    if not matched_args:
        return ""
    keys = list(when.keys()) if when else list(matched_args.keys())
    def _fmt(v: Any) -> str:
        if isinstance(v, bool): return "True" if v else "False"
        return str(v)
    return ", ".join(f"{k}={_fmt(matched_args[k])}" for k in keys if k in matched_args)

# =========================
# 3) 변환 실행
# =========================

def convert_csv(
    in_csv: Path,
    out_csv: Path,
    config_path: Path,
    encoding_in: str = "utf-8-sig",
    encoding_out: str = "utf-8-sig",
    minimal: bool = True,
) -> pd.DataFrame:
    by_code = load_config(config_path)

    # Windows/Excel 호환: 입력도 utf-8-sig 우선
    df = pd.read_csv(in_csv, encoding=encoding_in)

    if "func" not in df.columns or "raw_args" not in df.columns:
        raise ValueError("입력 CSV에는 'func'와 'raw_args' 컬럼이 필요합니다.")

    map_funcs, map_enums, map_args = [], [], []
    for _, row in df.iterrows():
        code = str(row["func"]).strip()
        rargs = parse_raw_args(row["raw_args"])
        f, e, matched, when = map_from_code(code, rargs, by_code)
        map_funcs.append("" if f is None else f)
        map_enums.append("" if e is None else e)
        map_args.append(map_args_to_string(matched, when))

    df["map_func"] = map_funcs
    df["map_enum"] = map_enums
    df["map_args"] = map_args

    # 결과 포맷 고정: func / raw_args / query / map_func / map_enum / map_args
    if minimal:
        cols = [c for c in ["func", "raw_args", "query", "map_func", "map_enum", "map_args"] if c in df.columns]
        df = df[cols]

    df.to_csv(out_csv, index=False, encoding=encoding_out)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", required=True, help="입력 CSV 경로 (func, raw_args 필수)")
    ap.add_argument("--out", dest="out_csv", required=True, help="출력 CSV 경로")
    ap.add_argument("--config", dest="config", required=True, help="설정 파일(.yaml/.yml/.json)")
    ap.add_argument("--encoding-in", default="utf-8-sig", help="입력 CSV 인코딩(기본: utf-8-sig)")
    ap.add_argument("--encoding-out", default="utf-8-sig", help="출력 CSV 인코딩(기본: utf-8-sig; Excel 호환)")
    ap.add_argument("--full", action="store_true", help="원본 모든 컬럼 유지(기본은 6컬럼만 출력)")
    args = ap.parse_args()

    convert_csv(
        in_csv=Path(args.in_csv),
        out_csv=Path(args.out_csv),
        config_path=Path(args.config),
        encoding_in=args.encoding_in,
        encoding_out=args.encoding_out,
        minimal=not args.full,
    )
    print(f"[OK] saved: {args.out_csv}")

if __name__ == "__main__":
    main()
