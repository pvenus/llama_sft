#!/usr/bin/env python3
"""Function-call evaluation CLI and importable API.

This module evaluates function-calling behavior for baseline/finetuned models
using mlx_lm CLI generation. It can be used as a CLI or imported via
`evaluate_fc(...)` without changing behavior.
"""
import argparse
import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path


__all__ = [
    "evaluate_fc",
    "main",
    "load_patterns",
]

# Runtime spec (must be loaded via --spec or spec=)
SPEC = None
COMPILED_RE = None

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

# ===== State simulator & parsing =====
def new_state():
    """새로운 초기 상태 생성."""
    return {"power": "off", "night_mode": False}

def apply_calls(state, calls):
    """상태에 함수 호출을 적용하여 새 상태 반환."""
    s = copy.deepcopy(state)
    for c in calls or []:
        n = c.get("name")
        if n == "power_on": s["power"] = "on"
        elif n == "power_off": s["power"] = "off"
        elif n == "enable_night_mode": s["night_mode"] = True
    return s

def parse_json(text: str):
    """문자열 JSON을 안전하게 파싱(실패 시 None)."""
    if not text: return None
    try: return json.loads(text)
    except Exception: return None

def calls_from(obj):
    """객체에서 'calls' 리스트 추출(없으면 빈 리스트)."""
    return (obj or {}).get("calls", [])

def load_jsonl(path):
    """JSONL 파일을 로드하여 리스트로 반환."""
    rows=[]
    with open(path, "r", encoding="utf-8") as f:
        for line in f: rows.append(json.loads(line))
    return rows

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
def gen_cli(model, prompt, max_tokens=32, adapter_path=None):
    """Call mlx_lm CLI to generate text. Retries with alternate entrypoint and
    provides detailed diagnostics on failure."""
    base_args = [
        "--model", model,
        "--ignore-chat-template",
        "--temp", "0.0", "--top-k", "1",
        "--max-tokens", str(max_tokens),
        "--prompt", prompt,
    ]
    if adapter_path:
        base_args += ["--adapter-path", adapter_path]

    cmd1 = [sys.executable, "-m", "mlx_lm", "generate", *base_args]
    cmd2 = [sys.executable, "-m", "mlx_lm.generate", *base_args]  # deprecated path, but works on some installs

    # First attempt
    out = subprocess.run(cmd1, capture_output=True, text=True)
    if out.returncode == 0:
        return out.stdout.strip()

    # Retry with alternate invocation
    out2 = subprocess.run(cmd2, capture_output=True, text=True)
    if out2.returncode == 0:
        return out2.stdout.strip()

    # Compose rich error message
    def fmt(cmd, cp):
        return (
            "Command: " + " ".join([repr(c) for c in cmd]) + "\n" +
            f"Return code: {cp.returncode}\n" +
            ("STDERR:\n" + (cp.stderr or "<empty>") + "\n") +
            ("STDOUT:\n" + (cp.stdout or "<empty>") + "\n")
        )

    err = (
        "[mlx_lm generate] failed twice.\n\n" +
        fmt(cmd1, out) + "\n--- RETRY ---\n" + fmt(cmd2, out2)
    )
    raise RuntimeError(err)

# ===== Config resolution =====
def resolve_args(args):
    """Resolve final configuration from CLI Namespace (no YAML)."""
    model = args.model
    eval_file = args.file
    adapters = args.adapters
    max_tokens = args.max_tokens
    out_path = args.out

    if not model:
        raise ValueError("model이 비어있습니다. --model을 지정하세요.")
    if not eval_file:
        raise ValueError("file(평가 데이터)가 비어있습니다. --file을 지정하세요.")
    if not out_path:
        raise ValueError("out 경로가 비어있습니다. --out 을 지정하세요 (예: outputs/eval.jsonl)")
    if max_tokens is None:
        raise ValueError("max_tokens 값이 비어있습니다. --max_tokens 를 지정하세요 (예: --max_tokens 32)")

    # Auto-detect mode
    mode = "finetuned" if adapters else "baseline"

    return {
        "mode": mode,
        "model": str(model),
        "eval_file": str(eval_file),
        "out_path": str(out_path),
        "adapters": str(adapters) if adapters else None,
        "max_tokens": int(max_tokens),
        "no_fallback": bool(args.no_fallback),
    }

# ===== Core evaluation routine (shared by CLI & API) =====
def _run_eval(R):
    """설정에 따라 평가를 수행하고 결과 통계 및 파일 저장."""
    mode = R["mode"]
    model, eval_file = R["model"], R["eval_file"]
    out_path, adapters = R["out_path"], R["adapters"]
    max_tokens, no_fallback = R["max_tokens"], R["no_fallback"]

    print(f"[cfg] mode={mode} model={model}")
    print(f"[cfg] eval_file={eval_file} out={out_path} adapters={adapters}")
    print(f"[cfg] max_tokens={max_tokens} no_fallback={no_fallback}")

    data = load_jsonl(eval_file)
    total = len(data)
    em = name_acc = state_ok = 0
    fb_used = 0

    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    with open(outp, "w", encoding="utf-8") as w:
        for ex in data:
            user = ex["messages"][-1]["content"]
            gold = parse_json(ex["assistant"])
            g_calls = calls_from(gold)
            s0 = new_state()
            s_gold = apply_calls(s0, g_calls)

            prompt = build_prompt(user)
            if adapters:
                print(f"[dbg] adapters={adapters}")
            else:
                print("[dbg] adapters: <none>")
            raw = gen_cli(model, prompt, max_tokens=max_tokens, adapter_path=adapters)

            # 다음 턴 나오기 전까지만 잘라서 추출
            cut = raw.split("\nUser:", 1)[0]
            pred_txt = extract_calls_json(cut)

            # 1) 일단 파싱해서 pred 생성
            pred = parse_json(pred_txt)

            # 2) 폴백: pred가 없거나 calls가 비면 규칙으로 채움 (no_fallback이 False일 때만)
            used_fallback = False
            if ((not pred) or (not calls_from(pred))) and (not no_fallback):
                fb = fallback_route_ko(user)
                if fb:
                    pred = {"calls": [{"name": fb, "arguments": {}}]}
                    pred_txt = json.dumps(pred, ensure_ascii=False)
                    used_fallback = True
                    fb_used += 1  # ← 파일 전체 통계용 카운터(루프 바깥에서 0으로 초기화해 둔 것)

            p_calls = calls_from(pred)
            s_pred = apply_calls(s0, p_calls)

            if pred and gold and json.dumps(pred, sort_keys=True) == json.dumps(gold, sort_keys=True):
                em += 1
            if p_calls and g_calls:
                name_acc += int(p_calls[0].get("name") == g_calls[0].get("name"))
            state_ok += int(s_pred == s_gold)

            w.write(json.dumps({
                "user": user,
                "gold": gold,
                "pred": pred,
                "pred_text": pred_txt,  # 첫 JSON(또는 폴백) 반영
                "state_gold": s_gold,
                "state_pred": s_pred,
                "raw": raw,  # 원본 전체(디버그용)
                "source": "fallback" if used_fallback else "model"
            }, ensure_ascii=False) + "\n")

    print(f"TOTAL={total}")
    print(f"ExactMatch = {em/total:.3f}")
    print(f"FuncName@1 = {name_acc/total:.3f}")
    print(f"StateMatch = {state_ok/total:.3f}")
    print(f"Fallback used = {fb_used}/{total} ({(fb_used/total):.1%})")
    print("Saved →", outp)

    return {
        "total": total,
        "exact_match": em / total if total > 0 else 0.0,
        "funcname_at_1": name_acc / total if total > 0 else 0.0,
        "state_match": state_ok / total if total > 0 else 0.0,
        "fallback_used": fb_used,
        "out_path": out_path,
        "mode": mode,
        "model": model,
        "adapters": adapters,
        "eval_file": eval_file,
        "max_tokens": max_tokens,
        "no_fallback": no_fallback,
    }

# ===== Public importable API =====
def evaluate_fc(
    model="mlx-community/Bio-Medical-Llama-3-2-1B-CoT-012025",
    file="outputs/datasets/test.jsonl",
    out="outputs/eval/output.jsonl",
    adapters=None,
    max_tokens=32,
    no_fallback=False,
    spec="assets/train/fc_patterns.json",
):
    """함수 호출 평가를 수행하는 공개 API 함수."""
    if spec is None:
        raise ValueError("spec path is required when calling evaluate_fc(...) programmatically.")
    load_patterns(spec)
    args = argparse.Namespace(
        model=model,
        file=file,
        out=out,
        adapters=adapters,
        max_tokens=max_tokens,
        no_fallback=no_fallback,
    )
    R = resolve_args(args)
    return _run_eval(R)

# ===== CLI entrypoint =====
def main():
    """명령행 인터페이스 진입점."""
    ap = argparse.ArgumentParser(
        description="Evaluate function-calling on a dataset using mlx_lm generate. Mode is auto-detected: finetuned if --adapters is provided, otherwise baseline."
    )
    ap.add_argument(
        "--model",
        default="mlx-community/Bio-Medical-Llama-3-2-1B-CoT-012025",
        help="HF repo or local path of the base model (e.g., 'mlx-community/Bio-Medical-Llama-3-2-1B-CoT-012025'). Default: mlx-community/Bio-Medical-Llama-3-2-1B-CoT-012025",
    )
    ap.add_argument(
        "--file",
        default="outputs/datasets/eval.jsonl",
        help="Path to eval JSONL file (each line has {'messages': [...], 'assistant': '...'}). Default: outputs/datasets/eval.jsonl",
    )
    ap.add_argument(
        "--spec",
        default="assets/train/fc_patterns.json",
        help="Path to JSON file defining sys_prompt/fewshot/regex patterns. Default: scripts/specs/fc_patterns.json",
    )
    ap.add_argument(
        "--out",
        default="outputs/eval/output.jsonl",
        help="Output .jsonl path (single file). Default: outputs/eval.jsonl",
    )
    ap.add_argument(
        "--adapters",
        default=None,
        help="Path to LoRA adapters (e.g., 'outputs/.../adapters.npz'). If provided, mode auto-sets to finetuned. Default: None",
    )
    ap.add_argument(
        "--max_tokens",
        type=int,
        default=32,
        help="Maximum generation tokens for each sample. Default: 32",
    )
    ap.add_argument(
        "--no_fallback",
        action="store_true",
        default=False,
        help="Disable rule-based fallback when the model fails to emit a valid calls JSON. Default: False",
    )
    args = ap.parse_args()

    load_patterns(args.spec)
    R = resolve_args(args)
    _run_eval(R)

if __name__ == "__main__":
    main()