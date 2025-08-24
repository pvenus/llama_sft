from __future__ import annotations
import json
from pathlib import Path

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


def render(st):
    st.markdown("---")
    st.subheader("Prompt parts · WHAT / HOW / FORMAT")

    # Paths + loader (main area)
    colp1, colp2, colp3 = st.columns(3)
    with colp1:
        what_path = st.text_input("WHAT JSON path", value="assets/prompt/prompt_what.json", key="what_path_main")
    with colp2:
        how_path = st.text_input("HOW JSON path", value="assets/prompt/prompt_how.json", key="how_path_main")
    with colp3:
        fmt_path = st.text_input("FORMAT JSON path", value="assets/prompt/prompt_format.json", key="fmt_path_main")

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
            # Show table preview of all prompts
            try:
                import pandas as pd
                rows = []
                with out_path.open("r", encoding="utf-8") as rf:
                    for line in rf:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rows.append(json.loads(line))
                        except Exception:
                            rows.append({"prompt": line})
                if rows:
                    st.caption("Preview of sys_prompt.jsonl:")
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render table preview: {e}")
        except Exception as e:
            st.error(f"Convert error: {e}")

    st.info("Set the JSON paths and click **Convert** to generate all combinations and save them to `assets/prompt/sys_prompt.jsonl`.")