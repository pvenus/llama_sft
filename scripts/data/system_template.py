import streamlit as st
from pathlib import Path
import yaml
import json

def _load_datas_list(path: str) -> list[str]:
    """
    Load a JSON file shaped like {"datas": [...]}.
    Elements may be strings or objects; objects are serialized to compact JSON strings.
    If file is missing, returns [] (caller can surface a UI message).
    """
    p = Path(path)
    if not p.exists():
        return []
    text = p.read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
    except Exception:
        # Allow YAML too (optional)
        try:
            obj = yaml.safe_load(text)
        except Exception:
            return []
    datas = (obj or {}).get("datas", [])
    out: list[str] = []
    for item in datas:
        if isinstance(item, str):
            out.append(item)
        else:
            out.append(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
    return out

def _param_items_from_func(func: dict):
    """
    Return ordered list of (param_name, param_info_dict) from a function spec.
    Supports:
      NEW: func["args"] = [{name, type, enum?, minimum?, maximum?, order?}, ...]
      OLD: func["parameters"] = { name: {type, ...}, ... }
    """
    if not isinstance(func, dict):
        return []
    # New schema: list under 'args'
    if isinstance(func.get("args"), list):
        items = []
        for idx, p in enumerate(func["args"]):
            if not isinstance(p, dict):
                continue
            pname = str(p.get("name", f"arg{idx}"))
            pinfo = {k: v for k, v in p.items() if k != "name"}
            if "type" not in pinfo:
                pinfo["type"] = "string"
            pinfo.setdefault("order", idx)
            items.append((pname, pinfo))
        items.sort(key=lambda x: (x[1].get("order", 9999), x[0]))
        return items
    # Old schema: dict under 'parameters'
    params = func.get("parameters") or {}
    if isinstance(params, dict):
        items = []
        for pname, pinfo in params.items():
            if not isinstance(pinfo, dict):
                pinfo = {"type": "string"}
            pinfo.setdefault("type", "string")
            pinfo.setdefault("order", 9999)
            items.append((pname, pinfo))
        items.sort(key=lambda x: (x[1].get("order", 9999), x[0]))
        return items
    return []

def build_functions_section(functions):
    lines = []
    # Global output format
    lines.append("Output JSON format (fixed):")
    lines.append(" [{\"functionName\":\"...\", \"arguments\":{\"key_name\":\"...\"}}, {\"functionName\":\"...\", \"arguments\":{\"key_name\":\"...\"}} ]")
    lines.append(" The arguments object must always contain exactly 3 fields.")
    lines.append("  # Return JSON only. No extra text.")
    lines.append("")
    # Functions
    lines.append("Available functions:")
    for func in functions:
        name = func.get("name", "unknown")
        description = func.get("description")
        if description:
            lines.append(f"- {name}: {description}")
        else:
            lines.append(f"- {name}:")
        param_items = _param_items_from_func(func)
        if param_items:
            lines.append("  arguments (object):")
            for pname, pinfo in param_items:
                # Build a human-friendly constraint hint
                hint_parts = []
                # ENUM: join with "/" (no spaces) e.g., A/B/C
                if isinstance(pinfo.get("enum"), (list, tuple)):
                    enum_vals = [str(v) for v in pinfo["enum"]]
                    hint_parts.append("/".join(enum_vals))
                # Numeric/text constraints (optional)
                if "minimum" in pinfo:
                    hint_parts.append(f"min={pinfo['minimum']}")
                if "maximum" in pinfo:
                    hint_parts.append(f"max={pinfo['maximum']}")
                if "min" in pinfo and "minimum" not in pinfo:
                    hint_parts.append(f"min={pinfo['min']}")
                if "max" in pinfo and "maximum" not in pinfo:
                    hint_parts.append(f"max={pinfo['max']}")
                if "pattern" in pinfo:
                    hint_parts.append(f"pattern={pinfo['pattern']}")
                if hint_parts:
                    # Two spaces after colon to match the requested style
                    lines.append(f"    - {pname}:  {' '.join(hint_parts)}")
                else:
                    lines.append(f"    - {pname}:")
        else:
            lines.append("  (no parameters)")
        # Example with placeholders (compact JSON)
        if param_items:
            args_obj = {pname: "..." for pname, _ in param_items}
        else:
            args_obj = {}
        example = json.dumps({"name": name, "arguments": args_obj})
        lines.append(f"  example: {example}")
    return "\n".join(lines)

def compose_preamble(sel_what: str, sel_how: str) -> str:
    """
    Compose a minimal system prompt preamble using fixed output format.
    FORMAT is now fixed to a single JSON object: {"name": "...", "arguments": { ... }}
    We no longer accept FORMAT from outside.
    Order: FORMAT (fixed) → WHAT → HOW → final guard.
    """
    parts = []
    # Fixed format block (do not change from external input)
    parts.append("Output JSON format (fixed):")
    parts.append("  { \"functionName\": \"...\", \"arguments\":{\"arg1\": \"...\",\"arg2\": \"...\",\"arg3\": \"...\"} }")
    parts.append(" The arguments object must always contain exactly 3 fields.")
    parts.append("  # Return JSON only. No extra text.")
    if sel_what:
        parts.append(f"WHAT={sel_what}")
    if sel_how:
        parts.append(f"HOW={sel_how}")
    parts.append("Return only the JSON. No extra text.")
    return "\n".join(parts)

# ... other imports and code ...

def main():
    # ... other code ...

    st.header("Prompt parts · WHAT / HOW / FORMAT")
    what_path = "assets/prompt/what.json"
    how_path = "assets/prompt/how.json"

    if st.button("Load", key="btn_load_parts"):
        try:
            st.session_state["what_list"] = _load_datas_list(what_path)
            st.session_state["how_list"] = _load_datas_list(how_path)

            # Build full sys_prompt candidates by combining WHAT × HOW with a fixed FORMAT
            whats = st.session_state["what_list"]
            hows = st.session_state["how_list"]

            # Append hidden, fixed functions section from YAML once
            fixed_yaml = Path("assets/prompt/function_real_convert.yaml")
            if yaml is None:
                raise RuntimeError("PyYAML is not installed. Please `pip install pyyaml`.")
            if not fixed_yaml.exists():
                raise FileNotFoundError(f"Function YAML not found: {fixed_yaml}")
            spec_yaml = yaml.safe_load(fixed_yaml.read_text(encoding="utf-8")) or {}
            funcs = spec_yaml.get("functions", [])
            fn_section = build_functions_section(funcs)

            def _mk_label(w: str, h: str) -> str:
                def _short(s, n):
                    return (s[:n] + "…") if len(s) > n else s
                return f"W:{_short(w, 32)} | H:{_short(h, 32)}"

            opts = []
            for w in whats:
                for h in hows:
                    preamble = compose_preamble(w, h)
                    full_text = preamble + ("\n\n" + fn_section if fn_section else "")
                    opts.append({"label": _mk_label(w, h), "text": full_text})
            st.session_state["sys_prompt_options"] = opts
            st.success(f"Loaded {len(opts)} sys_prompt candidates.")
        except Exception as e:
            st.error(f"Load error: {e}")

    what_list = st.session_state.get("what_list")
    how_list = st.session_state.get("how_list")

    if what_list and how_list:
        # Use list-box to select among fully composed sys_prompts
        opts = st.session_state.get("sys_prompt_options", [])
        if not opts:
            st.warning("Click Load to build sys_prompt candidates.")
        else:
            labels = [o["label"] for o in opts]
            idx = st.selectbox("Select sys_prompt", options=list(range(len(labels))), format_func=lambda i: labels[i], key="sel_sys_prompt_idx")
            preview = opts[idx]["text"] if 0 <= idx < len(opts) else ""
            with st.expander("sys_prompt preview"):
                st.code(preview, language="text")
    else:
        st.info("Load WHAT/HOW JSON to enable selection.")

    # ... other code ...