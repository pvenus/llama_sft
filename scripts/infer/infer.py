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

__all__ = [
    "run_streamlit",
]

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
    # run_streamlit(
    #     model="meta-llama/Llama-3.2-1B-Instruct",
    #     adapters=None,
    #     spec_path="assets/train/fc_patterns.json",
    #     default_max_tokens=32,
    #     default_no_fallback=False,
    # )

    if __name__ == "__main__":
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--model", required=True, help="로컬 병합모델(혹은 베이스) 폴더 경로")
        ap.add_argument("--adapters", default=None, help="LoRA 어댑터 경로(병합모델이면 비움)")
        ap.add_argument("--spec", default="assets/train/fc_patterns.json")
        ap.add_argument("--max_tokens", type=int, default=32)
        ap.add_argument("--no_fallback", action="store_true")
        args = ap.parse_args()

        run_streamlit(
            model=args.model,
            adapters=args.adapters,
            spec_path=args.spec,
            default_max_tokens=args.max_tokens,
            default_no_fallback=args.no_fallback,
        )

# ===== Streamlit UI (optional) =====

# --- Import left-pane menu UIs from separate files ---
try:
    from .menu_convert import render as render_menu_convert  # type: ignore
except Exception:
    try:
        from ..menu_convert import render as render_menu_convert  # type: ignore
    except Exception:
        try:
            from menu_convert import render as render_menu_convert  # type: ignore
        except Exception:
            import sys as _sys
            from pathlib import Path as _Path
            _sys.path.append(str(_Path(__file__).resolve().parents[2]))
            from scripts.ui.menu_convert import render as render_menu_convert  # type: ignore
try:
    from .menu_infer import render as render_menu_infer  # type: ignore
except Exception:
    try:
        from ..menu_infer import render as render_menu_infer  # type: ignore
    except Exception:
        try:
            from menu_infer import render as render_menu_infer  # type: ignore
        except Exception:
            import sys as _sys
            from pathlib import Path as _Path
            _sys.path.append(str(_Path(__file__).resolve().parents[2]))
            from scripts.ui.menu_infer import render as render_menu_infer  # type: ignore

def run_streamlit(model: str,
                  adapters: str | None = None,
                  spec_path: str = "assets/train/fc_patterns.json",
                  default_max_tokens: int = 32,
                  default_no_fallback: bool = True):
    """Run a minimal Streamlit UI that reuses the single-loaded model.
    Execute with: `streamlit run server.py -- --model ... --spec ... --max_tokens 32 --streamlit`
    """
    if st is None:
        raise RuntimeError("Streamlit is not installed. Please `pip install streamlit`.")

    st.set_page_config(page_title="Function-Call Inference", layout="centered")
    st.title("🔧 Function-Call Inference — Streamlit (Single-load)")
    st.caption("모델은 한 번만 로드되어 계속 재사용됩니다. 좌측 Settings에서 파라미터를 조정하세요.")
    with st.sidebar:
        st.subheader("메뉴")
        menu = st.radio("기능 선택", ["변환", "추론"], index=0)

    if menu == "변환":
        render_menu_convert(st)
    else:
        render_menu_infer(
            st,
            model,
            adapters,
            default_max_tokens,
            default_no_fallback
        )

if __name__ == "__main__":
    main()