#!/usr/bin/env python3
"""Streamlit UI for single-load function-call inference.

This module provides a simple Streamlit app:
- Load the model once (MLX or HF backend auto-detected)
- Type a prompt and run inference
- See parsed function-call JSON and raw outputs

No evaluation/metrics or FastAPI server mode is included.
"""
import argparse
import json
import re
from pathlib import Path
import time
import os
import sys as _sys

try:
    from ..common.env import backend
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.common.env import backend

# Optional Streamlit import (safe fallback if not installed)
try:
    import streamlit as st  # type: ignore
except Exception:
    st = None  # type: ignore

# Optional YAML import
try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore

# Import build_functions_section from system_template (relative first, then fallbacks)
try:
    from ..data.system_template import build_functions_section  # type: ignore
except Exception:
    try:
        from ..system_template import build_functions_section  # type: ignore
    except Exception:
        try:
            from system_template import build_functions_section  # type: ignore
        except Exception:
            # last resort: modify sys.path to locate scripts/data/system_template.py
            import sys as _sys
            _sys.path.append(str(Path(__file__).resolve().parents[2]))
            from scripts.data.system_template import build_functions_section  # type: ignore




__all__ = [
    "load_patterns",
    "run_streamlit",
]

# Runtime spec (must be loaded via --spec or spec=)
SPEC = None
COMPILED_RE = None

# Singleton text-generation runner (loaded once and reused)
RUNNER = None

class _HFRunner:
    def __init__(self, model: str, adapters: str | None = None):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        try:
            from peft import PeftModel  # optional
        except Exception:
            PeftModel = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model)
        self.m = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=(torch.float16 if self.device == "cuda" else None),
        )
        if adapters and PeftModel is not None:
            self.m = PeftModel.from_pretrained(self.m, adapters)
        if self.device == "cuda":
            self.m = self.m.to("cuda")

    def generate(self, prompt: str, max_tokens: int = 32) -> str:
        inputs = self.tok(prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        out = self.m.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        return self.tok.decode(out[0], skip_special_tokens=True)

#
# --- Compatibility shim: handle broken/old tqdm variants ----------------------
def _patch_tqdm_if_broken():
    """Ensure a minimally compatible tqdm exists so HF/MLX progress hooks don't crash.
    Creates/patches a dummy tqdm with `.tqdm` and `._lock` attributes if missing,
    and disables progress bars via env vars.
    """
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")
    try:
        import tqdm as _tq
        # Some environments expose a module without `tqdm` attr or without `_lock`
        if not hasattr(_tq, "tqdm"):
            class _Dummy:
                _lock = None
                def __init__(self, *a, **k): pass
                def update(self, *a, **k): pass
                def close(self, *a, **k): pass
                def set_description(self, *a, **k): pass
            _tq.tqdm = _Dummy  # type: ignore
        if not hasattr(_tq, "_lock"):
            try:
                _tq._lock = None  # type: ignore
            except Exception:
                pass
    except Exception:
        # Provide a minimal stub module if import fails entirely
        class _Dummy:
            _lock = None
            def __init__(self, *a, **k): pass
            def update(self, *a, **k): pass
            def close(self, *a, **k): pass
            def set_description(self, *a, **k): pass
        stub = type("_TQDMStub", (), {"tqdm": _Dummy, "_lock": None})()
        _sys.modules["tqdm"] = stub  # type: ignore


class _MLXRunner:
    def __init__(self, model: str, adapters: str | None = None):
        # Uses mlx_lm Python API (no subprocess). Adapters may be unsupported; ignored if not available.
        _patch_tqdm_if_broken()
        from mlx_lm import load as mlx_load
        try:
            # Newer mlx_lm uses `load(model)` and an optional adapter path
            if adapters:
                self.m, self.tok = mlx_load(model, adapter_path=adapters)
            else:
                self.m, self.tok = mlx_load(model)
        except TypeError:
            # Fallback if adapter_path is unsupported
            self.m, self.tok = mlx_load(model)
        except Exception as e:
            # If tqdm is the culprit, patch and retry once
            if "tqdm" in str(e).lower():
                _patch_tqdm_if_broken()
                if adapters:
                    self.m, self.tok = mlx_load(model, adapter_path=adapters)
                else:
                    self.m, self.tok = mlx_load(model)
            else:
                raise RuntimeError(f"Failed to load MLX model: {e}")

    def generate(self, prompt: str, max_tokens: int = 32) -> str:
        from mlx_lm import generate as mlx_generate
        # Use only minimal args to maximize compatibility across mlx_lm versions
        try:
            return mlx_generate(self.m, self.tok, prompt, max_tokens=max_tokens)
        except TypeError:
            try:
                # Some versions use `max_new_tokens` instead of `max_tokens`
                return mlx_generate(self.m, self.tok, prompt, max_new_tokens=max_tokens)
            except TypeError:
                # Fallback: rely on library defaults
                return mlx_generate(self.m, self.tok, prompt)


def init_runner(model: str, adapters: str | None = None):
    """Initialize the global RUNNER once based on backend()."""
    global RUNNER
    try:
        if backend() == "mlx":
            RUNNER = _MLXRunner(model, adapters)
        else:
            RUNNER = _HFRunner(model, adapters)
    except Exception as e:
        raise RuntimeError(f"Model load failed: {e}")
    return RUNNER

def load_patterns(spec_path: str | None = None):
    """Load SPEC and compile regexes strictly from an external JSON file.
    Required JSON schema:
    {
      "sys_prompt": "...",                    # string
      "fewshot": "..." or ["line1", ...],     # string or list of strings
      "regex_full": "...",                    # string (regex)
      "regex_tail_with_name": "...",          # string (regex)
      "regex_tail_nameless": "..."            # string (regex)
    }
    No built-in defaults are used. Missing or invalid keys raise an error.
    """
    if not spec_path:
        raise ValueError("--spec (or spec=) is required: external JSON spec must be provided.")
    p = Path(spec_path)
    if not p.exists():
        raise FileNotFoundError(f"Spec JSON not found: {spec_path}")
    with open(p, "r", encoding="utf-8") as f:
        spec = json.load(f)

    # Validate required keys
    required = [
        "sys_prompt",
        "fewshot",
        "regex_full",
        "regex_tail_with_name",
        "regex_tail_nameless",
    ]
    for k in required:
        if k not in spec:
            raise ValueError(f"Spec JSON missing required key: {k}")

    # Normalize fewshot
    fs = spec["fewshot"]
    if isinstance(fs, list):
        spec["fewshot"] = "\n".join(fs) + ("\n" if fs else "")
    elif not isinstance(fs, str):
        raise ValueError("'fewshot' must be a string or list of strings")

    # Types check for strings
    for k in ["sys_prompt", "regex_full", "regex_tail_with_name", "regex_tail_nameless"]:
        if not isinstance(spec[k], str):
            raise ValueError(f"Spec key '{k}' must be a string")

    # Set globals
    global SPEC, COMPILED_RE
    SPEC = spec
    COMPILED_RE = {
        "full": re.compile(spec["regex_full"], re.S),
        "tail_with_name": re.compile(spec["regex_tail_with_name"], re.S),
        "tail_nameless": re.compile(spec["regex_tail_nameless"], re.S),
    }

def build_prompt(user_text: str) -> str:
    """주어진 사용자 텍스트로 프롬프트를 생성."""
    if SPEC is None:
        raise RuntimeError("Spec not loaded. Provide --spec (or spec=) JSON with sys_prompt/fewshot/regexes.")
    sys_prompt = SPEC.get("sys_prompt", "")
    fewshot = SPEC.get("fewshot", "")
    return f"{sys_prompt}\n\n{fewshot}User: {user_text}\nAssistant: " + '{"calls":[{"name":"'

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


def parse_json(text: str):
    """문자열 JSON을 안전하게 파싱(실패 시 None)."""
    if not text: return None
    try: return json.loads(text)
    except Exception: return None

def calls_from(obj):
    """객체에서 'calls' 리스트 추출(없으면 빈 리스트)."""
    return (obj or {}).get("calls", [])

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

# ===== Sys-prompt composer from base text + function YAML =====

def compose_sys_prompt_from_files(base_txt_path: str, func_yaml_path: str) -> str:
    if yaml is None:
        raise RuntimeError("PyYAML is not installed. Please `pip install pyyaml`.")
    p_txt = Path(base_txt_path)
    p_yaml = Path(func_yaml_path)
    if not p_txt.exists():
        raise FileNotFoundError(f"Base system text not found: {base_txt_path}")
    if not p_yaml.exists():
        raise FileNotFoundError(f"Function YAML not found: {func_yaml_path}")
    base_text = p_txt.read_text(encoding="utf-8").strip()
    spec = yaml.safe_load(p_yaml.read_text(encoding="utf-8")) or {}
    funcs = spec.get("functions", [])
    section = build_functions_section(funcs)
    # Join with a blank line between
    return base_text + ("\n\n" + section if section else "")


# ===== Prompt parts loaders and composer =====

def _load_datas_list(path: str) -> list[str]:
    """Load a JSON file shaped like {"datas": [ ... ]}.
    Elements may be strings or objects; objects are compact-dumped to strings.
    Returns a list of string options.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt part JSON not found: {path}")
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
    datas = obj.get("datas", [])
    out: list[str] = []
    for item in datas:
        if isinstance(item, str):
            out.append(item)
        else:
            # Compact JSON for minimal tokens
            out.append(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
    return out


def compose_preamble(sel_what: str, sel_how: str, sel_format: str) -> str:
    """Compose a minimal sys_prompt using chosen WHAT/HOW/FORMAT strings.
    Keep tokens low; no markdown fences.
    """
    # Order: FORMAT first so structure is primed, then WHAT, then HOW rules
    parts = []
    if sel_what:
        parts.append(f"{sel_what}")
    if sel_how:
        parts.append(f"{sel_how}")
    if sel_format:
        parts.append(f"FORMAT={sel_format}")
    # Final guard
    parts.append("Return only the FORMAT as valid JSON. No extra text.")
    return "\n".join(parts)

# ===== Extraction regex (full and tail recovery) =====

def _norm_args_braces(txt: str) -> str:
    """'arguments' 필드의 빈 배열을 빈 객체로 정규화."""
    return (
        txt.replace('"arguments":[]', '"arguments":{}')
           .replace('"arguments": []', '"arguments": {}')
    )

def extract_calls_json(raw: str) -> str:
    """원시 텍스트에서 함수 호출 JSON 문자열 추출 및 꼬리 복구."""
    if COMPILED_RE is None:
        raise RuntimeError("Spec regexes not loaded. Provide --spec (or spec=) JSON.")
    # (A) 완전한 JSON 먼저
    m = COMPILED_RE["full"].search(raw)
    if m:
        return _norm_args_braces(m.group(0))
    # (B) "name": 포함된 꼬리 복구
    t = COMPILED_RE["tail_with_name"].search(raw)
    if t:
        name = t.group(1)
        return f'{{"calls":[{{"name":"{name}","arguments":{{}}}}]}}'
    # (C) name 라벨도 사라진 꼬리 복구
    u = COMPILED_RE["tail_nameless"].search(raw)
    if u:
        name = u.group(1)
        return f'{{"calls":[{{"name":"{name}","arguments":{{}}}}]}}'
    # (D) 최후 수단: 함수명만 탐지해 구성
    v = re.search(r'\b(power_on|power_off|enable_night_mode)\b', raw)
    if v:
        name = v.group(1)
        return f'{{"calls":[{{"name":"{name}","arguments":{{}}}}]}}'
    return ""

# ===== Generation via mlx_lm CLI =====
_HF_CACHE = {}

def _gen_hf(model: str, prompt: str, max_tokens=32, adapter_path=None) -> str:
    """Transformers/PEFT generation path for CUDA/CPU backends."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    try:
        from peft import PeftModel  # optional
    except Exception:
        PeftModel = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = (model, adapter_path, device)
    tok, m = _HF_CACHE.get(key, (None, None))
    if tok is None:
        tok = AutoTokenizer.from_pretrained(model)
        m = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=(torch.float16 if device == "cuda" else None),
        )
        if adapter_path and PeftModel is not None:
            # PEFT adapters directory
            m = PeftModel.from_pretrained(m, adapter_path)
        if device == "cuda":
            m = m.to("cuda")
        _HF_CACHE[key] = (tok, m)

    inputs = tok(prompt, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    out = m.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    return tok.decode(out[0], skip_special_tokens=True)

def gen_cli(model, prompt, max_tokens=32, adapter_path=None):
    # Reuse single-loaded runner (no per-request subprocess/model reload)
    runner = init_runner(model, adapter_path)
    return runner.generate(prompt, max_tokens=max_tokens)


def _ensure_spec_loaded(spec_path: str):
    if SPEC is None:
        load_patterns(spec_path)

def _infer_once(user_text: str, model: str, adapters: str | None, max_tokens: int, allow_fallback: bool) -> dict:
    prompt = build_prompt(user_text)
    raw = gen_cli(model, prompt, max_tokens=max_tokens, adapter_path=adapters)
    cut = raw.split("\nUser:", 1)[0]
    pred_txt = cut.strip()  # 가공 없이 모델 원본(JSON 가정) 그대로 사용
    closing = "}}]}"
    idx = pred_txt.find(closing)
    if idx != -1:
        pred = parse_json('{"calls":[{"name":"' + pred_txt[:idx + len(closing)])
    else:
        closing = "}}]"
        idx = pred_txt.find(closing)
        if idx != -1:
            pred = parse_json('{"calls":[{"name":"' + pred_txt[:idx + len(closing)] + "}")
        else:
            pred = parse_json('{"calls":[{"name":"' + pred_txt)
    used_fallback = False
    if ((not pred) or (not calls_from(pred))) and allow_fallback:
        fb = fallback_route_ko(user_text)
        if fb:
            pred = {"calls": [{"name": fb, "arguments": {}}]}
            pred_txt = json.dumps(pred, ensure_ascii=False)
            used_fallback = True
    return {
        "pred": pred,
        "pred_text": pred_txt,
        "raw": raw,
        "source": ("fallback" if used_fallback else "model"),
    }

def main():
    """Launch Streamlit UI.
    If executed via `python server.py` or `python -m ...` (bare mode),
    auto-bootstraps `streamlit run` on this file so the UI opens properly.
    """
    if st is None:
        raise SystemExit("Streamlit is not installed. Please `pip install streamlit`.")

    # Detect whether we are running inside a Streamlit runner
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore
        _ctx = get_script_run_ctx()
    except Exception:
        _ctx = None

    if _ctx is None:
        # Bare mode: re-exec with `streamlit run` so the app opens in a browser
        import subprocess
        from pathlib import Path
        script_path = Path(__file__).resolve()
        cmd = ["streamlit", "run", str(script_path)]
        print("[bootstrap] Executing:", " ".join(cmd))
        raise SystemExit(subprocess.call(cmd))

    # Already under Streamlit: run the UI
    run_streamlit(
        model="mlx-community/Llama-3.2-1B-Instruct-bf16",
        adapters=None,
        spec_path="assets/train/fc_patterns.json",
        default_max_tokens=32,
        default_no_fallback=False,
    )

# ===== Streamlit UI (optional) =====

def run_streamlit(model: str,
                  adapters: str | None = None,
                  spec_path: str = "assets/train/fc_patterns.json",
                  default_max_tokens: int = 32,
                  default_no_fallback: bool = False):
    """Run a minimal Streamlit UI that reuses the single-loaded model.
    Execute with: `streamlit run server.py -- --model ... --spec ... --max_tokens 32 --streamlit`
    """
    if st is None:
        raise RuntimeError("Streamlit is not installed. Please `pip install streamlit`.")

    # Ensure spec and model are loaded once
    _ensure_spec_loaded(spec_path)
    init_runner(model, adapters)

    st.set_page_config(page_title="Function-Call Inference", layout="centered")
    st.title("🔧 Function-Call Inference — Streamlit (Single-load)")
    st.caption("모델은 한 번만 로드되어 계속 재사용됩니다. 좌측 Settings에서 파라미터를 조정하세요.")
    with st.sidebar:
        st.subheader("메뉴")
        menu = st.radio("기능 선택", ["변환", "추론"], index=0)

    if menu == "변환":
        st.markdown("---")
        st.subheader("Prompt parts · WHAT / HOW / FORMAT")

        # Paths + loader (main area)
        colp1, colp2, colp3, colp4 = st.columns([4, 4, 4, 1])
        with colp1:
            what_path = st.text_input("WHAT JSON path", value="assets/prompt/prompt_what.json", key="what_path_main")
        with colp2:
            how_path = st.text_input("HOW JSON path", value="assets/prompt/prompt_how.json", key="how_path_main")
        with colp3:
            fmt_path = st.text_input("FORMAT JSON path", value="assets/prompt/prompt_format.json", key="fmt_path_main")
        with colp4:
            if st.button("Convert", key="btn_convert_parts"):
                try:
                    whats = _load_datas_list(what_path)
                    hows = _load_datas_list(how_path)
                    fmts = _load_datas_list(fmt_path)

                    # Fixed function section from YAML (hidden)
                    fixed_yaml = Path("assets/prompt/function.yaml")
                    if yaml is None:
                        raise RuntimeError("PyYAML is not installed. Please `pip install pyyaml`.")
                    if not fixed_yaml.exists():
                        raise FileNotFoundError(f"Function YAML not found: {fixed_yaml}")
                    spec_yaml = yaml.safe_load(fixed_yaml.read_text(encoding="utf-8")) or {}
                    funcs = spec_yaml.get("functions", [])
                    fn_section = build_functions_section(funcs)

                    # Build all combinations and write to JSONL
                    out_path = Path("assets/prompt/sys_prompt.jsonl")
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    count = 0
                    with out_path.open("w", encoding="utf-8") as wf:
                        for f in fmts:
                            for w in whats:
                                for h in hows:
                                    preamble = compose_preamble(w, h, f)
                                    full_text = preamble + ("\n\n" + fn_section if fn_section else "")
                                    line = {"prompt": full_text}
                                    wf.write(json.dumps(line, ensure_ascii=False) + "\n")
                                    count += 1
                    st.success(f"Converted and saved {count} prompts → {out_path}")
                    # Optional quick preview of first prompt
                    with out_path.open("r", encoding="utf-8") as rf:
                        first_line = rf.readline().strip()
                    if first_line:
                        st.caption("First line preview:")
                        st.code(first_line, language="json")
                except Exception as e:
                    st.error(f"Convert error: {e}")

        st.info("Set the JSON paths and click **Convert** to generate all combinations and save them to `assets/prompt/sys_prompt.jsonl`.")

    elif menu == "추론":
        st.markdown("---")
        st.subheader("Settings")
        st.write(f"**Model:** `{model}`")
        if adapters:
            st.write(f"**Adapters:** `{adapters}`")
        mt = st.number_input("max_tokens", min_value=1, max_value=512, value=int(default_max_tokens), step=1)
        allow_fb = st.checkbox("Enable fallback (rule-based)", value=(not default_no_fallback))

        st.markdown("---")
        st.subheader("Batch Inference (sys_prompt.jsonl × test_user_message.jsonl)")

        colb1, colb2, colb3 = st.columns([5, 5, 1])
        with colb1:
            sys_path = st.text_input("sys_prompt.jsonl 경로", value="assets/prompt/sys_prompt.jsonl",
                                     key="sys_jsonl_path")
        with colb2:
            msg_path = st.text_input("test_user_message.jsonl 경로", value="assets/prompt/test_user_message.jsonl",
                                     key="msg_jsonl_path")
        with colb3:
            run_batch = st.button("Batch Infer", type="primary", key="btn_batch_infer")

        if run_batch:
            try:
                # 1) 데이터 로드
                sys_list = _read_jsonl(sys_path)  # 기대: [{"prompt":"..."}] * N
                msg_list = _read_jsonl(msg_path)  # 기대: [{"message":"..."}] * M
                if not sys_list:
                    st.error("sys_prompt.jsonl에서 항목을 찾지 못했습니다. (빈 파일 또는 경로 확인)")
                if not msg_list:
                    st.error("test_user_message.jsonl에서 항목을 찾지 못했습니다. (빈 파일 또는 경로 확인)")

                # 2) 고정 SPEC 로딩
                _ensure_spec_loaded(spec_path)

                # 3) 전 조합 수행 (실시간 표 갱신)
                import pandas as pd  # Streamlit 내부에서 사용 가능
                df = pd.DataFrame(columns=[
                    "sys_prompt",
                    "user message",
                    "expected message",
                    "pred",
                    "pred_text",
                    "all",
                    "elapsed_ms",
                ])
                table_ph = st.empty()
                total = len(sys_list) * len(msg_list)
                done = 0
                prog = st.progress(0)

                for s in sys_list:
                    s_prompt = s.get("prompt", "").strip()
                    if not s_prompt:
                        continue
                    # sys_prompt 적용
                    SPEC["sys_prompt"] = s_prompt

                    for m in msg_list:
                        user_text = (m.get("message") or "").strip()
                        if not user_text:
                            continue
                        t0 = time.perf_counter()
                        out = _infer_once(user_text, model, adapters, int(mt), bool(allow_fb))
                        elapsed_ms = int((time.perf_counter() - t0) * 1000)
                        # 한 행 추가 후 즉시 갱신
                        df.loc[len(df)] = {
                            "sys_prompt": s_prompt,
                            "user message": user_text,
                            "expected message": (m.get("expected") or "").strip(),
                            "pred": out.get("pred"),
                            "pred_text": out.get("pred_text"),
                            "all": out,
                            "elapsed_ms": elapsed_ms,
                        }
                        table_ph.dataframe(df, use_container_width=True)

                        done += 1
                        prog.progress(min(1.0, done / max(1, total)))

                if df.empty:
                    st.info("생성된 결과가 없습니다. 입력 파일 내용을 확인하세요.")
            except Exception as e:
                st.error(f"Batch inference error: {e}")

if __name__ == "__main__":
    main()