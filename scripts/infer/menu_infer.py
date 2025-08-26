from __future__ import annotations
import time
import pandas as pd  # used inside Streamlit table rendering
import json
import re
from pathlib import Path
import os
import csv
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

# --- Read test_user_message.csv and normalize to [{"message":..., "expected":...}] ---
def _read_msg_csv(path: str) -> list[dict]:
    """
    Read test_user_message.csv and normalize to a list of dicts:
      {"message": <Query text>, "expected": <LLM Output>}
    Expected headers (case-sensitive preferred, case-insensitive fallback):
      - "Query(한글)" or "Query"
      - "LLM Output" or "expected"
    Rows with empty Query are skipped.
    """
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict] = []
    with p.open("r", encoding="utf-8") as rf:
        reader = csv.DictReader(rf)
        for row in reader:
            # Header fallbacks
            msg = (row.get("Query(한글)")
                   or row.get("Query")
                   or row.get("query")
                   or row.get("MESSAGE")
                   or row.get("message")
                   or "").strip()
            expected = (row.get("LLM Output")
                        or row.get("expected")
                        or row.get("EXPECT")
                        or row.get("Expect")
                        or "").strip()
            if not msg:
                continue
            out.append({"message": msg, "expected": expected})
    return out

# Default static system prompt (since SPEC is removed)
def build_prompt(sys_text: str, user_text: str) -> str:
    return f"{sys_text}, User:{user_text}, Assistant: " + '{"name":'

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
    Try a few candidate closures to build a valid single-object JSON:
    {"name":<cont>}
    {"name":<cont>}}
    {"name":<trimmed_cont>}}
    Returns (obj, used_text) where obj is parsed dict or None.
    """
    cont = _undouble_quotes(cont.strip())
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
    # Heuristic: if we can find the first occurrence of '}}', cut there and close once
    i = cont.find("}}")
    if i != -1:
        cand = '{"name":' + cont[:i+2]
        try:
            return json.loads(cand), cand
        except Exception:
            pass
    return None, '{"name":' + cont

def _infer_once(sys_text:str, user_text: str, model: str, adapters: str | None, max_tokens: int, allow_fallback: bool, runner_index: int) -> dict:
    prompt = build_prompt(sys_text, user_text)
    try:
        print(f"[infer_once] runner_index={runner_index} thread={threading.current_thread().name}")
    except Exception:
        pass
    raw = get_runner(model, adapters, index=runner_index).generate(prompt, max_tokens=max_tokens)
    pred_txt = raw.strip()

    # Prefer new '<function_XX>(...)' format; fallback to legacy JSON; then heuristic JSON completion
    fname, fargs = decode_call_any(pred_txt)
    used_fallback = False
    if fname:
        pred = {"name": fname, "arguments": fargs or {}}
    else:
        pred, _used = _finalize_pred_continuation(pred_txt)
        if ((not pred) or (not isinstance(pred, dict)) or ("name" not in pred)) and allow_fallback:
            fb = fallback_route_ko(user_text)
            if fb:
                pred = {"name": fb, "arguments": {}}
                pred_txt = json.dumps(pred, ensure_ascii=False)
                used_fallback = True
    return {
        "pred": pred,
        "pred_text": pred_txt,
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
    st.subheader("Batch Inference (sys_prompt.jsonl × test_user_message.csv)")

    colb1, colb2, colb3 = st.columns([5, 5, 1])
    with colb1:
        sys_path = st.text_input("sys_prompt.jsonl 경로", value="assets/prompt/sys_prompt.jsonl",
                                 key="sys_jsonl_path")
    with colb2:
        msg_path = st.text_input("test_user_message.csv 경로", value="assets/prompt/test_user_message.csv",
                                 key="msg_csv_path")
    run_batch = st.button("Batch Infer", type="primary", key="btn_batch_infer")



    if run_batch:
        try:
            # 1) 데이터 로드
            sys_list = _read_jsonl(sys_path)  # 기대: [{"prompt":"..."}] * N
            msg_list = _read_msg_csv(msg_path)  # 기대: [{"message":"..." , "expected":"..."}] * M
            if not sys_list:
                st.error("sys_prompt.jsonl에서 항목을 찾지 못했습니다. (빈 파일 또는 경로 확인)")
            if not msg_list:
                st.error("test_user_message.csv에서 항목을 찾지 못했습니다. (빈 파일 또는 경로 확인)")

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
                        table_ph.dataframe(df_view, use_container_width=True)

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
            out = {"pred_text": pred_txt, "source": "model"}

            # Prefer new '<function_XX>(...)' format; fallback to legacy JSON; then heuristic JSON completion
            fname, fargs = decode_call_any(pred_txt)
            if fname:
                pred = {"name": fname, "arguments": fargs or {}}
            else:
                pred, _used = _finalize_pred_continuation(pred_txt)

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
