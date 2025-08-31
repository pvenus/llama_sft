from __future__ import annotations
import time
import pandas as pd  # used inside Streamlit table rendering
import re
from pathlib import Path
import os
import csv, json
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from collections import deque, defaultdict
import threading
 # --- Per-file write locks to support multi-thread shards writing into one file ---
_FILE_LOCKS = defaultdict(threading.Lock)

def _read_jsonl(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with p.open("r", encoding="utf-8") as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out

def _read_jsonl_head(path: str, limit: int = 300) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with p.open("r", encoding="utf-8") as rf:
        for i, line in enumerate(rf):
            if i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out

# --- Read test message file and normalize to [{"message":..., "expected":...}] ---
def _read_msg_file(path: str) -> List[Dict[str, str]]:
    """
    Read (CSV or JSONL) and normalize to:
      {"message": <text>, "expected": <LLM Output>}
    지원:
      - CSV 헤더: "Query(한글)"|"Query"|"query"|"MESSAGE"|"message"
                 + "LLM Output"|"expected"|"assistant"|"EXPECT"|"Expect"
      - JSONL 라인:
          {"message":"...", "expected":"..."}            # 권장
        또는 {"messages":[{"role":"user","content":"..."}], "assistant":"..."}
        또는 {"calls":[...]} → expected로 간주(문자열 JSON으로 변환)
    """
    p = Path(path)
    if not p.exists():
        return []
    suf = p.suffix.lower()
    if suf in {".jsonl", ".json"}:
        return _read_msg_jsonl_internal(p)
    return _read_msg_csv_internal(p)

# ---- 내부 구현 (변경 없음) ----
_MSG_CAND = ["message","Message","query","Query(한글)","Query","MESSAGE"]
_EXP_CAND = ["expected","Expected","LLM Output","assistant","EXPECT","Expect"]

def _pick_col(headers: List[str], cands: List[str]) -> str:
    lower = {h.lower(): h for h in headers}
    for c in cands:
        if c.lower() in lower:
            return lower[c.lower()]
    # 못 찾으면 첫 후보를 그대로 반환(값이 비면 스킵됨)
    return cands[0]

def _read_msg_csv_internal(p: Path) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    with p.open("r", encoding="utf-8-sig", newline="") as rf:
        reader = csv.DictReader(rf)
        headers = reader.fieldnames or []
        mcol = _pick_col(headers, _MSG_CAND)
        ecol = _pick_col(headers, _EXP_CAND)
        for row in reader:
            msg = (row.get(mcol) or "").strip()
            exp = (row.get(ecol) or "").strip()
            if not msg:
                continue
            out.append({"message": msg, "expected": exp})
    return out

def _read_msg_jsonl_internal(p: Path) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    with p.open("r", encoding="utf-8") as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            # message 추출
            msg: Optional[str] = None
            if isinstance(obj.get("message"), str):
                msg = obj["message"]
            elif isinstance(obj.get("messages"), list) and obj["messages"]:
                for m in obj["messages"]:
                    if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str):
                        msg = m["content"]; break
                if msg is None:
                    m0 = obj["messages"][0]
                    if isinstance(m0, dict) and isinstance(m0.get("content"), str):
                        msg = m0["content"]

            # expected 추출 → 문자열로 표준화(공백 최소화)
            exp_val: Any = obj.get("expected")
            if exp_val is None and "assistant" in obj:
                exp_val = obj["assistant"]
            if exp_val is None and "calls" in obj:
                exp_val = {"calls": obj["calls"]}

            if isinstance(exp_val, (dict, list)):
                expected = json.dumps(exp_val, ensure_ascii=False, separators=(',', ':'))
            elif isinstance(exp_val, str):
                expected = exp_val.strip()
            else:
                expected = ""

            if msg:
                out.append({"message": msg.strip(), "expected": expected})
    return out

# Default static system prompt (since SPEC is removed)
def build_prompt(sys_text: str, user_text: str) -> str:
    # Llama 3.2 chat formatting with explicit role sections
    # System block -> User block -> Assistant header. We keep the JSON priming
    # so downstream logic that expects a continuation starting after '{"name":'
    # continues to work.
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{sys_text}\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{user_text}\n"
        "<|eot_id|><|start_header_id|>content<|end_header_id|>\n"
        '[{"functionName":'
    )

# --- Import runners from package (no local caching) ---
try:
    from ..common.runners import get_runner, release_runner  # type: ignore
except Exception:
    try:
        from ..runners import get_runner, release_runner  # type: ignore
    except Exception:
        try:
            from runners import get_runner, release_runner  # type: ignore
        except Exception:
            import sys as _sys  # ensure alias exists
            from pathlib import Path as _Path
            _sys.path.append(str(_Path(__file__).resolve().parents[2]))
            from scripts.common.runners import get_runner, release_runner  # type: ignore

def parse_json(text: str):
    """문자열 JSON을 안전하게 파싱(실패 시 None)."""
    if not text: return None
    try: return json.loads(text)
    except Exception: return None

def _undouble_quotes(s: str) -> str:
    return s.replace('""', '"') if s.count('""') > s.count('"')//2 else s

# (legacy) Triple-brace collapse helper (kept for compatibility if used elsewhere).
# NOTE: We no longer call this unconditionally; we only apply it after the first parse attempt fails.
def _collapse_triple_braces(s: str) -> str:
    if not isinstance(s, str):
        return s
    return s.replace("}}}", "}}")

# Try to decode, and only apply triple-brace fix after the first failure
def _decode_with_triple_brace_fix(s: str):
    """
    Try to decode either the new function-call format or legacy JSON.
    If the first attempt fails and the text contains '}}}', retry after collapsing to '}}'.
    Returns (name, args, used_text).
    """
    if not isinstance(s, str):
        return (None, None, s or "")
    # First attempt
    name, args = decode_call_any(s)
    if name:
        return name, args, s
    # Retry after fixing triple braces once
    if "}}}" in s:
        fixed = _collapse_triple_braces(s)
        name2, args2 = decode_call_any(fixed)
        if name2:
            return name2, args2, fixed
    return None, None, s

# --- helpers for <function_XX>(arg=val) format ---
def _coerce_scalar(v: str):
    s = v.strip()
    if not s:
        return ""
    # strip surrounding quotes if present
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    ls = s.lower()
    if ls in ("true", "false"):
        return ls == "true"
    # int
    try:
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return int(s)
    except Exception:
        pass
    # float
    try:
        return float(s)
    except Exception:
        pass
    return s

def _parse_args_kv(arg_str: str) -> dict:
    """Parse 'a=1, b="x"' to {a: 1, b: 'x'}. Empty -> {}"""
    if arg_str is None:
        return {}
    s = arg_str.strip()
    if not s:
        return {}
    out = {}
    # allow trailing <end>
    s = re.sub(r"<\s*end\s*>\s*$", "", s, flags=re.IGNORECASE)
    # split by commas that are not inside quotes
    parts = re.split(r",(?=(?:[^\\\"]|\\.|\"[^\"]*\")*$)", s)
    for p in parts:
        if not p.strip():
            continue
        if "=" not in p:
            # bare flag -> treat as True
            k = p.strip()
            out[k] = True
            continue
        k, v = p.split("=", 1)
        out[k.strip()] = _coerce_scalar(v)
    return out

def parse_function_call(s: str):
    """Return (name, args) from strings like '<function_WN>(brightness=2)' or '<function_IO>(timeframe=1, location="0")'.
    If multiple calls are present (separated by ';'), parse the first. Returns (None, None) on failure.
    """
    if not s:
        return None, None
    text = s.strip()
    # keep only first segment before ';'
    text = text.split(";", 1)[0].strip()
    m = re.search(r"<\s*function_([A-Za-z0-9]+)\s*>\s*\((.*?)\)\s*(?:<\s*end\s*>\s*)?$", text)
    if not m:
        # also allow no-arg form like <function_KI>() or <function_KI>
        m2 = re.search(r"<\s*function_([A-Za-z0-9]+)\s*>\s*(?:\((.*?)\))?", text)
        if not m2:
            return None, None
        fname = f"function_{m2.group(1)}"
        arg_str = m2.group(2) or ""
        return fname, _parse_args_kv(arg_str)
    fname = f"function_{m.group(1)}"
    arg_str = m.group(2) or ""
    return fname, _parse_args_kv(arg_str)

def decode_call_any(s: str):
    """Decode either legacy JSON {"name":..., "arguments":{...}} or the new '<function_XX>(...)' string.
    Returns (name, args) or (None, None).
    """
    if not s:
        return None, None
    # try JSON first
    obj = parse_json(s)
    if isinstance(obj, dict) and "name" in obj:
        return obj.get("name"), obj.get("arguments") if isinstance(obj.get("arguments"), dict) else {}
    # try function-call format
    return parse_function_call(s)
def _finalize_pred_continuation(cont: str):
    """
    cont: model continuation that starts right after '{"name":'
    Goal:
      - Prefer to complete a valid single-object JSON: {"name":<cont>}
      - Specifically, trim right after the JSON 'arguments' section and the closing '}' that follows it.
      - If the closing '}' after arguments is missing, add it.
    Returns (obj, used_text) where obj is parsed dict or None.
    """
    # 1) sanitize (⚠️ 기존 코드의 `_undou...le_quotes` 오타를 `_undouble_quotes`로 교체)
    cont = _undouble_quotes(cont.strip())

    # 2) quick attempts
    candidates = [
        '{"name":' + cont,
        '{"name":' + cont + "}",
        '{"name":' + cont.rstrip(", ") + "}",
    ]
    for c in candidates:
        try:
            return json.loads(c), c
        except Exception:
            continue

    # 3) Targeted trim around "arguments": find the arguments object closing '}', then the next '}' (outer)
    def _find_matching_brace(s: str, start_idx: int) -> int:
        """
        Find the index of the matching '}' for the '{' at start_idx.
        Ignores braces inside strings and handles escapes.
        Returns -1 if not found.
        """
        depth = 0
        in_str = False
        esc = False
        for i in range(start_idx, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            else:
                if ch == '"':
                    in_str = True
                elif ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        return i
        return -1

    def _fix_arguments_subjson(arg_obj_text: str) -> str:
        """
        Heuristically repair a broken JSON object text for "arguments".
        - Trim leading spaces inside quoted values like: "key": "  value" -> "key": "value"
        - Replace hard newlines inside an open string with a single space
        - If the number of double-quotes is odd (unbalanced), append a closing quote just before the final '}'.
        Returns repaired text (still an object string starting with '{' and ending with '}').
        """
        if not isinstance(arg_obj_text, str) or len(arg_obj_text) < 2:
            return arg_obj_text

        s = arg_obj_text

        # 1) Normalize CR/LF within likely string contexts by replacing raw newlines between quotes with a space.
        #    We keep it simple: collapse any newline directly followed by non-quote text until the next quote.
        s = s.replace("\r\n", "\n")
        # Prevent accidental breakage of JSON by removing newlines that are likely inside a string value.
        # This is heuristic but safe for our truncated 'detail' case.
        s = re.sub(r'(":\s*")\s*\n+\s*', r'\1', s)     # newline right after an opening value quote
        s = re.sub(r'\s*\n+\s*(")', r' \1', s)         # newline right before a closing quote

        # 2) Trim leading spaces inside quoted values: "key": "  value" -> "key": "value"
        s = re.sub(r'(":\s*")\s+', r'\1', s)

        # 3) If quotes are unbalanced (odd count), append a closing quote just before the final '}'
        #    to handle cases like:  "detail": " ... 까지로 끊겨도
        if s.count('"') % 2 == 1:
            # insert one quote before the last '}' (if present)
            rpos = s.rfind('}')
            if rpos != -1:
                s = s[:rpos] + '"' + s[rpos:]

        return s

    def _try_json_variants_from_cont(base_cont: str):
        """
        Try to build valid JSON by appending/repairing closers after the continuation.
        User-specified normalization sequence:
          - 정상: arguments 뒤에 "}}" 가 있으면 기존 시도에서 잡힘.
          - 비정상 보정:
            1) arguments 뒤에 '}' 하나만 있는 경우 → '}}' 로 보정
            2) 실패 시 → '}}' 추가
            3) 실패 시 → '"}}' 추가
            4) 실패 시 → 마지막 한 글자 삭제 후 '"}}' 추가
        각 시도는
          A) 그대로
          B) 끝에 외부용 '}' 1개 추가
          C) 끝 공백/콤마 제거 후 '}' 추가
        를 순차적으로 시도.
        """
        # --- Normalize cases ending with triple braces '}}}' (or more) ---
        # If the continuation ends with '}}}' we collapse to exactly '}}' so that
        # the outer object can close cleanly. Also, if '}}}' appears anywhere,
        # prepare a "front-only" variant that keeps content up to the first '}}}'
        # and replaces it with '}}' to enable partial parsing from the beginning.
        base_norm = base_cont
        # Collapse any run of 3+ closing braces at the very end to exactly '}}'
        m_tail = re.search(r'\}\}\}+\s*$', base_norm)
        if m_tail:
            base_norm = base_norm[:m_tail.start()] + '}}'

        # If '}}}' occurs inside the string (not necessarily at the end),
        # create a head-only variant up to the first '}}}' and normalize to '}}'.
        head_only = None
        if '}}}' in base_norm:
            head_only = base_norm.split('}}}', 1)[0] + '}}'

        variants = []
        # Prefer head-only normalized variant first if available
        if head_only:
            variants.append(head_only)
        # Then the fully normalized base
        variants.append(base_norm)
        # Original as-is (in case the above normalization was over-eager)
        variants.append(base_cont)
        # Common auto-closing attempts
        variants.append(base_norm + "}")
        variants.append(base_norm + "}}")
        variants.append(base_norm + '"}}')
        if base_norm:
            variants.append(base_norm[:-1] + '"}}')

        # De-duplicate while preserving order
        _seen = set()
        variants = [v for v in variants if not (v in _seen or _seen.add(v))]

        for v in variants:
            # try without extra outer brace first
            cand = '{"name":' + v
            try:
                return json.loads(cand), cand
            except Exception:
                pass

            # try with one outer brace
            cand2 = '{"name":' + v + "}"
            try:
                return json.loads(cand2), cand2
            except Exception:
                pass

            # try with rstrip of trailing comma/space then one outer brace
            cand3 = '{"name":' + v.rstrip(", ") + "}"
            try:
                return json.loads(cand3), cand3
            except Exception:
                pass

        return None, None
        """
        Heuristically repair a broken JSON object text for "arguments".
        - Trim leading spaces inside quoted values like: "key": "  value" -> "key": "value"
        - Replace hard newlines inside an open string with a single space
        - If the number of double-quotes is odd (unbalanced), append a closing quote just before the final '}'.
        Returns repaired text (still an object string starting with '{' and ending with '}').
        """
        if not isinstance(arg_obj_text, str) or len(arg_obj_text) < 2:
            return arg_obj_text

        s = arg_obj_text

        # 1) Normalize CR/LF within likely string contexts by replacing raw newlines between quotes with a space.
        #    We keep it simple: collapse any newline directly followed by non-quote text until the next quote.
        s = s.replace("\r\n", "\n")
        # Prevent accidental breakage of JSON by removing newlines that are likely inside a string value.
        # This is heuristic but safe for our truncated 'detail' case.
        s = re.sub(r'(":\s*")\s*\n+\s*', r'\1', s)     # newline right after an opening value quote
        s = re.sub(r'\s*\n+\s*(")', r' \1', s)         # newline right before a closing quote

        # 2) Trim leading spaces inside quoted values: "key": "  value" -> "key": "value"
        s = re.sub(r'(":\s*")\s+', r'\1', s)

        # 3) If quotes are unbalanced (odd count), append a closing quote just before the final '}'
        #    to handle cases like:  "detail": " ... 까지로 끊겨도
        if s.count('"') % 2 == 1:
            # insert one quote before the last '}' (if present)
            rpos = s.rfind('}')
            if rpos != -1:
                s = s[:rpos] + '"' + s[rpos:]

        return s

    # Find "arguments" (supports "arguments" or 'arguments')
    m = re.search(r'["\']arguments["\']\s*:\s*{', cont)
    if m:
        brace_open = cont.find('{', m.end() - 1)  # the '{' just after "arguments":
        if brace_open != -1:
            brace_close = _find_matching_brace(cont, brace_open)
            if brace_close != -1:
                # Extract the arguments object slice and try to heuristically fix common truncations:
                # e.g., {"arguments":{"data":" 청정 결과 보고서","detail":" 청정 결과 보고서는 발표의 중요성에 따라
                args_slice = cont[brace_open:brace_close + 1]
                fixed_args_slice = _fix_arguments_subjson(args_slice)

                # Rebuild 'cont' with the fixed arguments object
                cont = cont[:brace_open] + fixed_args_slice + cont[brace_close + 1:]

                # Recompute close index because 'cont' may have changed length
                brace_close = brace_open + len(fixed_args_slice) - 1

                # After arguments object, trim everything after the first consecutive '}}'
                after_args = cont[brace_close + 1:]
                # 1) Prefer the first literal '}}' as the boundary (outer close right after inner close)
                jj = after_args.find('}}')
                if jj != -1:
                    # Keep content up to and including the first '}}' and drop any trailing garbage
                    trunc = cont[:brace_close + 1] + after_args[:jj + 2]
                else:
                    # 2) If only a single '}' exists, add one more to complete the object
                    j1 = after_args.find('}')
                    if j1 != -1:
                        trunc = cont[:brace_close + 1] + after_args[:j1 + 1] + '}'
                    else:
                        # 3) Neither '}}' nor '}' found -> force a proper close
                        trunc = cont[:brace_close + 1] + '}}'
                cand = '{"name":' + trunc
                try:
                    return json.loads(cand), cand
                except Exception:
                    # Try without trailing commas/spaces and ensure one final brace
                    cand2 = '{"name":' + trunc.rstrip(", ") + "}"
                    try:
                        return json.loads(cand2), cand2
                    except Exception:
                        pass

    # 4) Fallback: cut at first '}}' and close once
    i = cont.find("}}")
    if i != -1:
        cand = '{"name":' + cont[:i+2]
        try:
            return json.loads(cand), cand
        except Exception:
            pass

    # 4-bis) User-specified normalization heuristics on the tail after "arguments"
    obj, used = _try_json_variants_from_cont(cont)
    if obj is not None:
        return obj, used

    # 5) Last resort: return partially formed variant
    return None, '{"name":' + cont

def _infer_once(sys_text:str, user_text: str, model: str, adapters: str | None, max_tokens: int, allow_fallback: bool, runner_index: int) -> dict:
    prompt = build_prompt(sys_text, user_text)
    try:
        print(f"[infer_once] runner_index={runner_index} thread={threading.current_thread().name}")
    except Exception:
        pass
    raw = get_runner(model, adapters, index=runner_index).generate(prompt, max_tokens=max_tokens)
    pred_txt = raw.strip()

    # Prefer new '<function_XX>(...)' format; if that fails, try triple-brace fix once,
    # then fallback to legacy JSON completion heuristics.
    fname, fargs, pred_txt_used = _decode_with_triple_brace_fix(pred_txt)
    used_fallback = False
    if fname:
        pred = {"name": fname, "arguments": fargs or {}}
        final_pred_text = pred_txt_used
    else:
        pred, _used = _finalize_pred_continuation(pred_txt_used)
        final_pred_text = pred_txt_used
        if ((not pred) or (not isinstance(pred, dict)) or ("name" not in pred)) and allow_fallback:
            fb = fallback_route_ko(user_text)
            if fb:
                pred = {"name": fb, "arguments": {}}
                final_pred_text = json.dumps(pred, ensure_ascii=False)
                used_fallback = True
    return {
        "pred": pred,
        "pred_text": final_pred_text,
        "raw": raw,
        "source": ("fallback" if used_fallback else "model"),
    }

def _write_jsonl_line(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as wf:
        wf.write(json.dumps(obj, ensure_ascii=False) + "\n")


# Shared iterator for per-message inference used by both _infer_worker and run_batch
def _iter_infer_rows(index: int, s_prompt: str, msg_list: list[dict], model: str, adapters: str | None, mt: int, allow_fb: bool, runner_index: int):
    """
    Yield unified per-message inference rows so run_batch and _infer_worker share the same logic.
    Produces dict with keys:
      index, runner_idx, ts, sys_prompt, user, expected, pred, pred_text, elapsed_ms, name_match, arg_match, source
    """
    for m in msg_list:
        user_text = (m.get("message") or "").strip()
        if not user_text:
            continue
        print(f"[infer] prompt#{index:02d} runner_idx={runner_index} thread={threading.current_thread().name} -> start")
        t0 = time.perf_counter()
        out = _infer_once(s_prompt, user_text, model, adapters, int(mt), bool(allow_fb), runner_index)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        print(f"[infer] prompt#{index:02d} runner_idx={runner_index} thread={threading.current_thread().name} -> done {elapsed_ms} ms")

        expected_raw = (m.get("expected") or "").strip()
        expected_name, expected_args = decode_call_any(expected_raw)

        pred_obj = out.get("pred")
        pred_name = None
        pred_args = None
        if isinstance(pred_obj, dict) and "name" in pred_obj:
            pred_name = pred_obj.get("name")
            pred_args = pred_obj.get("arguments") if isinstance(pred_obj.get("arguments"), dict) else {}
        else:
            # try to decode from pred_text (new format)
            ptxt = out.get("pred_text") or ""
            pn, pa = decode_call_any(ptxt)
            pred_name, pred_args = pn, pa

        name_match = (pred_name == expected_name)
        arg_match = _args_equal(pred_args, expected_args)

        yield {
            "index": index,
            "runner_idx": runner_index,
            "ts": datetime.now().isoformat(timespec="seconds"),
            "sys_prompt": s_prompt,
            "user": user_text,
            "expected": expected_raw,
            "pred": out.get("pred"),
            "pred_text": out.get("pred_text"),
            "elapsed_ms": elapsed_ms,
            "name_match": name_match,
            "arg_match": arg_match,
            "source": out.get("source"),
        }

def _infer_worker(index: int, s_prompt: str, msg_list: list[dict], model: str, adapters: str | None, mt: int, allow_fb: bool, out_dir: Path):
    """
    Process all messages for one sys_prompt and write to outputs/result_{index:02d}.jsonl
    """
    out_path = out_dir / f"result_{index:02d}.jsonl"
    start_ts = datetime.now().isoformat(timespec="seconds")
    meta = {"meta": {"index": index, "started_at": start_ts, "model": model, "adapters": adapters or "", "sys_prompt": s_prompt}}
    _write_jsonl_line(out_path, meta)

    for row in _iter_infer_rows(index, s_prompt, msg_list, model, adapters, mt, allow_fb, index):
        _write_jsonl_line(out_path, row)

    _write_jsonl_line(out_path, {"meta": {"index": index, "finished_at": datetime.now().isoformat(timespec="seconds")}})
    import gc as _gc
    _gc.collect()
    return str(out_path)

# ===== 간단 폴백 라우터 =====
def fallback_route_ko(text: str):
    """한국어 텍스트에 대해 간단한 규칙 기반 함수명 라우팅."""
    s = text.strip().lower()
    # 나이트(우선순위↑)
    if re.search(r'(야간|수면|night|잠|quiet|휴식)', s): return "enable_night_mode"
    # 끄기
    if re.search(r'(끄|꺼|off|전원\s*꺼)', s):         return "power_off"
    # 켜기
    if re.search(r'(켜|on|전원\s*켜)', s):             return "power_on"
    return None

def _args_equal(a, b) -> bool:
    if a is None: a = {}
    if b is None: b = {}
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return False
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        va, vb = a[k], b[k]
        if str(va) != str(vb):
            return False
    return True

def _arrow_safe_df(df: pd.DataFrame, force_str_cols: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Coerce problematic/mixed-type columns to strings so Streamlit -> Arrow conversion won't fail.
    Only columns in force_str_cols are coerced; if None, all object dtype columns are coerced.
    """
    if df is None:
        return df
    df2 = df.copy()
    if force_str_cols is None:
        force_str_cols = [c for c in df2.columns if str(df2[c].dtype) == "object"]
    for c in force_str_cols:
        if c not in df2.columns:
            continue
        def _to_str(v):
            if v is None:
                return ""
            # Keep simple scalars as string; JSON-encode dict/list for readability.
            if isinstance(v, (dict, list)):
                try:
                    return json.dumps(v, ensure_ascii=False)
                except Exception:
                    return str(v)
            return str(v)
        try:
            df2[c] = df2[c].map(_to_str)
        except Exception:
            # As a last resort
            df2[c] = df2[c].astype("string")
    return df2

def render(
    st,
    model: str,
    adapters: str | None,
    default_max_tokens: int,
    default_no_fallback: bool
):
    st.markdown("---")
    st.subheader("Settings")
    st.write(f"**Model:** `{model}`")
    if adapters:
        st.write(f"**Adapters:** `{adapters}`")
    mt = st.number_input("max_tokens", min_value=1, max_value=512, value=int(default_max_tokens), step=1)
    allow_fb = st.checkbox("Enable fallback (rule-based)", value=(not default_no_fallback))

    colx1, colx2, colx3 = st.columns([2,4,4])
    with colx1:
        max_batch = st.number_input("MAX_BATCH", min_value=1, max_value=128, value=8, step=1, help="한 번에 모델에 넣을 프롬프트 수(배치 크기)")
    with colx2:
        out_dir_str = st.text_input("Output dir", value="outputs", help="Results will be saved as result_{index:02d}.jsonl")
    with colx3:
        monitor = st.checkbox("Monitor output dir", value=True)
        lite_table = st.checkbox("Lightweight table (show minimal columns)", value=True, help="Reduces memory by omitting large text fields.")
        max_rows_keep = st.number_input("Max table rows kept", min_value=50, max_value=5000, value=500, step=50, help="Older rows will be dropped from view to save memory.")

    st.markdown("---")
    st.subheader("Batch Inference (sys_prompt.jsonl × test_convert_msg.jsonl)")

    colb1, colb2, colb3 = st.columns([5, 5, 1])
    with colb1:
        sys_path = st.text_input("sys_prompt.jsonl 경로", value="assets/prompt/sys_prompt.jsonl",
                                 key="sys_jsonl_path")
    with colb2:
        msg_path = st.text_input("test_convert_msg.jsonl 경로", value="assets/prompt/test_convert_msg.jsonl",
                                 key="msg_jsonl_path")
    run_batch = st.button("Batch Infer", type="primary", key="btn_batch_infer")



    if run_batch:
        try:
            # 1) 데이터 로드
            sys_list = _read_jsonl(sys_path)  # 기대: [{"prompt":"..."}] * N
            msg_list = _read_msg_file(msg_path)  # 기대: [{"message":"..." , "expected":"..."}] * M
            if not sys_list:
                st.error("sys_prompt.jsonl에서 항목을 찾지 못했습니다. (빈 파일 또는 경로 확인)")
            if not msg_list:
                st.error("test_convert_msg.jsonl에서 항목을 찾지 못했습니다. (빈 파일 또는 경로 확인)")

            cols = [
                "name_match",
                "arg_match",
                "pred_name",
                "pred_args",
                "pred_text",  # raw model output string in <function_XX>(...) format
                "expected_name",
                "expected_args",
                "sys_prompt",
                "user_message",
                "elapsed_ms",
                "source",
            ]
            df_buffer = deque(maxlen=int(max_rows_keep))
            table_ph = st.empty()
            total = len(sys_list) * len([m for m in msg_list if (m.get("message") or "").strip()])
            done = 0
            prog = st.progress(0)
            # --- Per-sys_prompt summary and details ---
            summary_rows = []  # per sys_prompt aggregates
            details_by_key: dict[str, list[dict]] = {}
            summary_ph = st.empty()  # placeholder for the summary table
            details_ph = st.empty()  # clearable container to render expanders under the table

            for s in sys_list:
                s_prompt = s.get("prompt", "").strip()
                if not s_prompt:
                    continue
                # per-prompt counters and detail collector
                all_match_sum = 0
                name_only_match_sum = 0
                argument_only_match_sum = 0
                detail_rows = []

                for row in _iter_infer_rows_batched(0, s_prompt, msg_list, model, adapters, int(mt), bool(allow_fb), 0, int(max_batch)):
                    name_match = row["name_match"]
                    arg_match = row["arg_match"]
                    pred_obj = row.get("pred")

                    # Decode expected from either '<function_XX>(...)' or legacy JSON
                    expected_name, expected_args = decode_call_any(row.get("expected") or "")

                    # Predicted: prefer already-parsed dict; else decode from pred_text
                    if isinstance(pred_obj, dict) and "name" in pred_obj:
                        pred_name = pred_obj.get("name")
                        pred_args = pred_obj.get("arguments") if isinstance(pred_obj.get("arguments"), dict) else {}
                    else:
                        pred_name, pred_args = decode_call_any(row.get("pred_text") or "")

                    # update per-prompt counters (mutually exclusive buckets)
                    if name_match and arg_match:
                        all_match_sum += 1
                    elif name_match and not arg_match:
                        name_only_match_sum += 1
                    elif arg_match and not name_match:
                        argument_only_match_sum += 1

                    # collect detailed row for this sys_prompt
                    detail_rows.append({
                        "ts": row["ts"],
                        "user": row["user"],
                        "expected_name": expected_name,
                        "expected_args": expected_args,
                        "pred_name": pred_name,
                        "pred_args": pred_args,
                        "pred_text": row.get("pred_text"),
                        "name_match": name_match,
                        "arg_match": arg_match,
                        "elapsed_ms": row["elapsed_ms"],
                        "source": row.get("source"),
                    })

                    row_min = {
                        "name_match": "✅" if name_match else "❌",
                        "arg_match": "✅" if arg_match else "❌",
                        "pred_name": pred_name,
                        "pred_args": json.dumps(pred_args, ensure_ascii=False) if isinstance(pred_args, dict) else str(pred_args),
                        "pred_text": row.get("pred_text"),
                        "expected_name": expected_name,
                        "expected_args": json.dumps(expected_args, ensure_ascii=False) if isinstance(expected_args, dict) else str(expected_args),
                        "sys_prompt": s_prompt,
                        "user_message": row["user"],
                        "elapsed_ms": row["elapsed_ms"],
                        "source": row.get("source"),
                    }
                    df_buffer.append(row_min)
                    if df_buffer:
                        if lite_table:
                            df_view = pd.DataFrame(list(df_buffer), columns=cols)
                        else:
                            row_full = dict(row_min)
                            row_full["pred_text"] = row.get("pred_text")
                            df_view = pd.DataFrame(list(df_buffer))
                        # Arrow-safe render (avoid mixed bool/int/object columns)
                        df_render = _arrow_safe_df(
                            df_view,
                            force_str_cols=[
                                "pred_args",
                                "expected_args",
                                "pred_text",
                                "sys_prompt",
                                "user_message",
                                "source",
                                "pred_name",
                                "expected_name",
                            ],
                        )
                        table_ph.dataframe(df_render, use_container_width=True)

                    done += 1
                    prog.progress(min(1.0, done / max(1, total)))

                # record per-prompt summary
                detail_key = f"sys_{hash(s_prompt) & 0xffff:04x}"
                summary_rows.append({
                    "all match sum": all_match_sum,
                    "name only match sum": name_only_match_sum,
                    "argument only match sum": argument_only_match_sum,
                    "sys message": s_prompt,
                    "detail": f"click to expand: {detail_key}",
                })
                details_by_key[detail_key] = detail_rows

                # --- Incremental render: update summary table and details per prompt ---
                if summary_rows:
                    summary_df = pd.DataFrame(summary_rows, columns=[
                        "all match sum",
                        "name only match sum",
                        "argument only match sum",
                        "sys message",
                        "detail",
                    ])
                    summary_df = _arrow_safe_df(summary_df, force_str_cols=["sys message", "detail"])
                    with summary_ph:
                        st.markdown("---")
                        st.subheader("Summary by sys message")
                        st.dataframe(summary_df, use_container_width=True)
                    with details_ph.container():
                        for sr in summary_rows:
                            key_label = sr["detail"].split(":", 1)[-1].strip()
                            with st.expander(f"Details — {key_label}"):
                                det = details_by_key.get(key_label, [])
                                if det:
                                    det_df = pd.DataFrame(det)
                                    det_df = _arrow_safe_df(
                                        det_df,
                                        force_str_cols=[
                                            "ts",
                                            "user",
                                            "expected_name",
                                            "expected_args",
                                            "pred_name",
                                            "pred_args",
                                            "pred_text",
                                            "source",
                                        ],
                                    )
                                    st.dataframe(det_df, use_container_width=True)
                                else:
                                    st.write("No details.")


            # release runner for index 0 after batch
            release_runner(0)

            if not df_buffer:
                st.info("생성된 결과가 없습니다. 입력 파일 내용을 확인하세요.")
        except Exception as e:
            st.error(f"Batch inference error: {e}")


# New batched iterator for batch inference
def _iter_infer_rows_batched(index: int, s_prompt: str, msg_list: list[dict], model: str, adapters: str | None, mt: int, allow_fb: bool, runner_index: int, batch_size: int):
    """
    Yield per-message rows using batched generation when possible.
    Tries runner.generate(list_of_prompts), falls back to per-item generate on failure.
    """
    # Pre-build (user, expected_raw) pairs while skipping empty messages
    pairs = []
    for m in msg_list:
        user_text = (m.get("message") or "").strip()
        if not user_text:
            continue
        expected_raw = (m.get("expected") or "").strip()
        pairs.append((user_text, expected_raw))
    if not pairs:
        return

    r = get_runner(model, adapters, index=runner_index)
    N = len(pairs)
    i = 0
    while i < N:
        chunk = pairs[i:i+int(batch_size)]
        prompts = [build_prompt(s_prompt, u) for (u, _) in chunk]
        print(f"[batch] prompt#{index:02d} runner_idx={runner_index} size={len(prompts)} thread={threading.current_thread().name} -> start")
        t0 = time.perf_counter()
        outputs = None
        # Try true batch first
        try:
            outputs = r.generate(prompts, max_tokens=int(mt))  # type: ignore
            if not isinstance(outputs, (list, tuple)) or len(outputs) != len(prompts):
                raise RuntimeError("Runner returned non-list or wrong size")
        except Exception:
            # Fallback to per-item
            outputs = []
            for p in prompts:
                try:
                    outputs.append(r.generate(p, max_tokens=int(mt)))
                except Exception as _e:
                    outputs.append("")
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        print(f"[batch] prompt#{index:02d} runner_idx={runner_index} size={len(prompts)} -> done {elapsed_ms} ms")

        # Parse each output and yield rows
        for (user_text, expected_raw), raw in zip(chunk, outputs):
            pred_txt = (raw or "").strip()
            # Try normal parse, then retry once with triple-brace collapse on failure.
            fname, fargs, pred_txt_used = _decode_with_triple_brace_fix(pred_txt)
            out = {"pred_text": pred_txt_used, "source": "model"}

            if fname:
                pred = {"name": fname, "arguments": fargs or {}}
            else:
                pred, _used = _finalize_pred_continuation(pred_txt_used)

            if ((not pred) or (not isinstance(pred, dict)) or ("name" not in pred)) and allow_fb:
                fb = fallback_route_ko(user_text)
                if fb:
                    pred = {"name": fb, "arguments": {}}
                    out["pred_text"] = json.dumps(pred, ensure_ascii=False)
                    out["source"] = "fallback"

            out["pred"] = pred

            expected_name, expected_args = decode_call_any(expected_raw)

            if isinstance(pred, dict) and "name" in pred:
                pred_name = pred.get("name")
                pred_args = pred.get("arguments") if isinstance(pred.get("arguments"), dict) else {}
            else:
                pred_name, pred_args = decode_call_any(out.get("pred_text") or "")

            name_match = (pred_name == expected_name)
            arg_match = _args_equal(pred_args, expected_args)

            yield {
                "index": index,
                "runner_idx": runner_index,
                "ts": datetime.now().isoformat(timespec="seconds"),
                "sys_prompt": s_prompt,
                "user": user_text,
                "expected": expected_raw,
                "pred": out.get("pred"),
                "pred_text": out.get("pred_text"),
                "elapsed_ms": elapsed_ms,
                "name_match": name_match,
                "arg_match": arg_match,
                "source": out.get("source"),
            }
        i += int(batch_size)
