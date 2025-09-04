# scripts/infer/postprocess.py
# 경연 전용 FunctionCall 포맷 보정기
# - 함수 스키마(function_schema.yaml): 각 함수의 시그니처(허용 인자 조합; one-of)
# - 인자 스키마(arg_schema.yaml): arg별 규칙 리스트(타겟 함수 집합/타입/enum|range)
# - 최종 포맷: "<function_XX>(args?);<function_YY>(...)<end>"

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

END_TOKEN = "<end>"
FUNC_CALL_RX = re.compile(r"<(function_[A-Za-z0-9_]+)>\((.*?)\)", re.DOTALL)

FULLWIDTH_MAP = str.maketrans({
    "＜": "<", "＞": ">", "〈": "<", "〉": ">",
    "﹤": "<", "﹥": ">", "《": "<", "》": ">"
})

@dataclass
class RepairReport:
    text: str
    tags: List[str] = field(default_factory=list)  # 적용 순서대로 기록

# ---------------------------------------------------------------------------
# 0) 공백 규칙: 괄호 내부 쉼표 뒤만 공백 허용, 그 외 공백/탭 제거
# ---------------------------------------------------------------------------
def _normalize_ws_strict(s: str, report: Optional[RepairReport] = None) -> str:
    out: List[str] = []
    paren = 0
    changed = False
    for ch in s:
        if ch == "(":
            paren += 1
            out.append(ch)
            continue
        if ch == ")":
            paren = max(0, paren - 1)
            if out and out[-1] == " ":  # ", )" → ",)"
                out.pop(); changed = True
            out.append(ch)
            continue
        if ch.isspace():
            if paren > 0 and out and out[-1] == ",":
                out.append(" ")
            else:
                changed = True
            continue
        out.append(ch)
    res = "".join(out)
    if report and changed:
        report.tags.append("whitespace")
    return res

# ---------------------------------------------------------------------------
# A) 브래킷/꺾쇠/순서 보정
# ---------------------------------------------------------------------------
def fix_fullwidth_brackets(s: str, report: RepairReport) -> str:
    if any(ch in s for ch in ("＜","＞","〈","〉","﹤","﹥","《","》")):
        s2 = s.translate(FULLWIDTH_MAP)
        if s2 != s:
            report.tags.append("fullwidth")
        return s2
    return s

def fix_square_to_angle(s: str, report: RepairReport) -> str:
    s2, n = re.subn(r"\[(function_[A-Za-z0-9_]+)\]", r"<\1>", s)
    if n > 0:
        report.tags.append("square-angle")
    return s2

def fix_misordered_angle_and_paren(s: str, report: RepairReport) -> str:
    s2, n = re.subn(r"<(function_[A-Za-z0-9_]+)\((.*?)\)>", r"<\1>(\2)", s, flags=re.DOTALL)
    if n > 0:
        report.tags.append("order-fix")
    return s2

def fix_missing_gt_before_paren(s: str, report: RepairReport) -> str:
    s2, n = re.subn(r"<(function_[A-Za-z0-9_]+)\(", r"<\1>(", s)
    if n > 0:
        report.tags.append("missing-gt")
    return s2

def fix_missing_angle_pair(s: str, report: RepairReport) -> str:
    s2, n = re.subn(r"(?<!<)(function_[A-Za-z0-9_]+)\(", r"<\1>(", s)
    if n > 0:
        report.tags.append("wrap-angle")
    return s2

# ---------------------------------------------------------------------------
# B) 세미콜론/괄호 누락 보정
# ---------------------------------------------------------------------------
def fix_missing_semicolons(s: str, report: RepairReport) -> str:
    s2, n = re.subn(r"\)\s*<function_", r");<function_", s)
    if n > 0:
        report.tags.append("semicolon-insert")
    return s2

def fix_missing_close_before_next_call(s: str, report: RepairReport) -> str:
    s2, n = re.subn(r"(<function_[A-Za-z0-9_]+>\([^)]*?);<function_", r"\1);<function_", s, flags=re.DOTALL)
    if n > 0:
        report.tags.append("closeparen-insert")
    return s2

# ---------------------------------------------------------------------------
# C) 스키마 로딩/유틸
#   function_schema.yaml
#     functions:
#       - name: function_BP
#         signatures:
#           - [get]
#           - [enable, type]
#   arg_schema.yaml
#     args:
#       enable:
#         - target: [function_BP]  # 특정 함수 전용 규칙
#           type: integer
#           enum: [0,1]
#         - target: []             # 전역 규칙
#           type: boolean
#           enum: [False, True]
# ---------------------------------------------------------------------------
def _funcschema_to_map(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    fmap: Dict[str, Dict[str, Any]] = {}
    if not schema:
        return fmap
    for it in (schema.get("functions") or []):
        nm = it.get("name")
        if not nm:
            continue
        sigs = it.get("signatures") or []
        norm: List[List[str]] = []
        for s in sigs:
            if isinstance(s, (list, tuple)):
                norm.append([str(k) for k in s])
        allowed = sorted(set(k for sig in norm for k in sig))
        fmap[nm] = {"signatures": norm, "allowed": allowed}
    return fmap

def _argschema_to_index(schema: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    arg_map[arg] = [ {target:[fn...], type:str, enum|min|max...}, ... ]
    target: [] 는 전역 규칙(모든 함수에 적용)
    """
    if not schema:
        return {}
    args = schema.get("args") or {}
    out: Dict[str, List[Dict[str, Any]]] = {}
    for arg, rules in args.items():
        rules_list: List[Dict[str, Any]] = []
        if isinstance(rules, list):
            for r in rules:
                if not isinstance(r, dict):
                    continue
                entry = dict(r)
                entry["target"] = list(entry.get("target") or [])
                rules_list.append(entry)
        out[arg] = rules_list
    return out

def _pick_arg_spec(arg_rules: List[Dict[str, Any]], fname: str) -> Dict[str, Any]:
    """
    우선순위: (1) target에 fname이 포함된 규칙 → (2) target == [] 전역 규칙 → (3) 기본 string
    """
    if not arg_rules:
        return {"type": "string"}
    # specific
    for r in arg_rules:
        if r.get("target"):
            if fname in r["target"]:
                return r
    # global
    for r in arg_rules:
        if not r.get("target"):
            return r
    return {"type": "string"}

def _split_kv_parts(inner: str) -> List[Tuple[str, str]]:
    parts = [p.strip() for p in inner.split(",") if p.strip()]
    out: List[Tuple[str, str]] = []
    for p in parts:
        m = re.match(r"^\s*([A-Za-z0-9_]+)\s*=\s*(.+?)\s*$", p)
        if m:
            out.append((m.group(1), m.group(2)))
    return out

# ---- 타입 캐스팅/검증 (boolean|integer 유니온 등도 허용) -------------------
def _try_coerce_primitive(raw: str, typ: str) -> Tuple[Optional[Any], Optional[str]]:
    typ = (typ or "string").lower()
    if typ == "integer":
        try:
            return int(raw), "cast-integer"
        except Exception:
            if raw.lower() in {"true", "false"}:
                return (1 if raw.lower() == "true" else 0), "cast-integer"
            return None, None
    if typ == "number":
        try:
            return float(raw), "cast-float"
        except Exception:
            return None, None
    if typ == "boolean":
        low = raw.lower()
        if low in {"true", "false"}:
            return (low == "true"), "cast-boolean"
        if raw in {"0", "1"}:
            return (raw == "1"), "cast-boolean"
        return None, None
    # string
    return raw, None

def _coerce_value(v: str, type_spec: Any) -> Tuple[Optional[Any], Optional[str]]:
    raw = v.strip()
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        raw = raw[1:-1]
    if isinstance(type_spec, list):
        candidates = [str(t).lower() for t in type_spec]
    else:
        ts = (str(type_spec or "string")).lower()
        candidates = [t.strip() for t in ts.split("|")] if "|" in ts else [ts]
    for t in candidates:
        val, tag = _try_coerce_primitive(raw, t)
        if val is not None:
            return val, tag
    return None, None

def _validate_range_enum(val: Any, spec: Dict[str, Any]) -> bool:
    if "enum" in spec:
        enum = spec["enum"]
        if isinstance(val, bool):
            # allow True/False to match 1/0 in numeric enums
            if enum and all(isinstance(x, (int, float, bool)) for x in enum):
                as_int = 1 if val else 0
                return (as_int in enum) or (val in enum)
        return val in enum
    if isinstance(val, (int, float)):
        if "min" in spec and val < spec["min"]:
            return False
        if "max" in spec and val > spec["max"]:
            return False
    if isinstance(val, str) and "regex" in spec:
        try:
            return re.fullmatch(spec["regex"], val) is not None
        except re.error:
            return True
    return True

def _default_value_for_spec(spec: Dict[str, Any]) -> Any:
    if not spec:
        return ""
    if "enum" in spec and isinstance(spec["enum"], list) and spec["enum"]:
        return spec["enum"][0]
    type_spec = spec.get("type", "string")
    types = [t.strip() for t in (type_spec if isinstance(type_spec, str) else "|".join(type_spec)).lower().split("|")]
    for t in types:
        if t in {"integer", "number"}:
            if "min" in spec:
                return int(spec["min"]) if t == "integer" else float(spec["min"])
            return 0 if t == "integer" else 0.0
        if t == "boolean":
            return False
        if t == "string":
            return ""
    return ""

def _format_args_minimal_from_pairs(pairs: List[Tuple[str, Any]]) -> str:
    xs: List[str] = []
    for k, v in pairs:
        if isinstance(v, bool):
            vv = "True" if v else "False"  # 대문자 표기 통일
        elif isinstance(v, (int, float)):
            vv = str(v)
        else:
            vv = v
            if not re.fullmatch(r"[A-Za-z0-9_\-\.]+", vv or ""):
                vv = f"\"{vv}\""
        xs.append(f"{k}={vv}")
    return ", ".join(xs)

# ---------------------------------------------------------------------------
# D) 시그니처 선택/적용 (부분 일치 허용 + 누락 기본값 채우기)
# ---------------------------------------------------------------------------
def _select_signature_best(present: set, signatures: List[List[str]]) -> Optional[List[str]]:
    """
    부분 일치도 기반 선택:
      - overlap = |present ∩ sig|
      - 우선순위: (overlap DESC, len(sig) DESC, index ASC)
    overlap 0이어도 가장 '길고 앞선' 시그니처를 선택해서 누락 채우기 시도
    """
    best = None
    best_score = (-1, -1, 10**9)
    for idx, sig in enumerate(signatures):
        sset = set(sig)
        overlap = len(present & sset)
        score = (overlap, len(sig), -idx)
        if score > best_score:
            best_score = score
            best = sig
    return best

def _apply_schema_to_call(
    fname: str,
    inner: str,
    func_map: Dict[str, Dict[str, Any]],
    arg_index: Dict[str, List[Dict[str, Any]]],
    report: RepairReport
) -> Optional[str]:
    rules = func_map.get(fname)
    if not rules:
        return _format_args_minimal_inner(inner)

    allowed: List[str] = rules.get("allowed") or []
    signatures: List[List[str]] = rules.get("signatures") or []

    kvs = _split_kv_parts(inner)
    kept: Dict[str, Any] = {}

    # 1) 허용/타입검증/캐스팅
    for k, vraw in kvs:
        if allowed and (k not in allowed):
            report.tags.append("drop-unknown"); continue
        # pick arg spec for this function
        spec = _pick_arg_spec(arg_index.get(k, []), fname)
        val, cast_tag = _coerce_value(vraw, spec.get("type") or "string")
        if val is None:
            report.tags.append("drop-invalid"); continue
        if not _validate_range_enum(val, spec):
            report.tags.append("drop-invalid"); continue
        if cast_tag:
            report.tags.append(cast_tag)
        if k not in kept:
            kept[k] = val

    present = set(kept.keys())
    # 2) 최적 시그니처 선택
    chosen = _select_signature_best(present, signatures)
    if not chosen:
        # 시그니처가 비어있다면(정의누락) 현재 인자만 유지
        return _format_args_minimal_from_pairs(list(kept.items()))

    # 3) 시그니처 외 키 드랍
    extra = present - set(chosen)
    if extra:
        report.tags.append("drop-extra")
        for k in list(extra):
            kept.pop(k, None)

    # 4) 누락 인자 기본값 채우기 (함수별/전역 규칙에서 추론)
    missing = [k for k in chosen if k not in kept]
    if missing:
        for k in missing:
            spec = _pick_arg_spec(arg_index.get(k, []), fname)
            kept[k] = _default_value_for_spec(spec)
        report.tags.append("signature-fill")

    # 5) 시그니처 순서대로 포맷
    ordered = [(k, kept[k]) for k in chosen]
    return _format_args_minimal_from_pairs(ordered)

def _format_args_minimal_inner(inner: str) -> str:
    if inner is None:
        return ""
    core = re.sub(r"\s+", "", inner)
    return core.replace(",", ", ")

# ---------------------------------------------------------------------------
# E) 유효한 함수콜만 추출해 재조립
# ---------------------------------------------------------------------------
def rebuild_from_valid_calls(
    s: str,
    report: RepairReport,
    func_schema: Optional[Dict[str, Any]],
    arg_schema: Optional[Dict[str, Any]],
) -> str:
    calls: List[str] = []
    func_map = _funcschema_to_map(func_schema or {})
    arg_index = _argschema_to_index(arg_schema or {})
    for m in FUNC_CALL_RX.finditer(s):
        fname = m.group(1)
        inner = m.group(2)
        if func_map:
            inner_fmt = _apply_schema_to_call(fname, inner, func_map, arg_index, report)
        else:
            inner_fmt = _format_args_minimal_inner(inner)
        if inner_fmt is None:
            # 현재 로직상 발생하지 않음
            continue
        calls.append(f"<{fname}>({inner_fmt})")
    if not calls:
        report.tags.append("default-mr")
        return "<function_MR>()" + END_TOKEN
    rebuilt = ";".join(calls) + END_TOKEN
    report.tags.append("rebuild-calls")
    return rebuilt

# ---------------------------------------------------------------------------
# 메인 진입점
# ---------------------------------------------------------------------------
def repair_functioncall_format(
    raw_text: str,
    max_rounds: int = 2,
    func_schema: Optional[Dict[str, Any]] = None,
    arg_schema: Optional[Dict[str, Any]] = None,
) -> RepairReport:
    """
    파이프라인:
      0) 공백 정규화
      A) 꺾쇠/브래킷/순서 보정
      B) 세미콜론 & 닫는 괄호 보정
      C) 유효한 함수콜만 추출하여 재조립(+<end>)
      ※ 시그니처 미충족 시: 누락 인자를 기본값으로 채움 (signature-fill)
    """
    rep = RepairReport(text=_normalize_ws_strict(raw_text, None))
    rep.text = _normalize_ws_strict(rep.text, rep)

    for _ in range(max_rounds):
        before = rep.text

        # A단계
        rep.text = fix_fullwidth_brackets(rep.text, rep)
        rep.text = fix_square_to_angle(rep.text, rep)
        rep.text = fix_misordered_angle_and_paren(rep.text, rep)
        rep.text = fix_missing_gt_before_paren(rep.text, rep)
        rep.text = fix_missing_angle_pair(rep.text, rep)

        # B단계
        rep.text = fix_missing_close_before_next_call(rep.text, rep)
        rep.text = fix_missing_semicolons(rep.text, rep)

        # C단계(재조립)
        rep.text = rebuild_from_valid_calls(rep.text, rep, func_schema, arg_schema)

        if rep.text == before:
            break

    # '<end>' 뒤 노이즈 제거
    if rep.text.endswith(END_TOKEN):
        idx = rep.text.rfind(END_TOKEN)
        rep.text = rep.text[: idx + len(END_TOKEN)]

    return rep
