import re, ast, pandas as pd
from collections import Counter, defaultdict

# === 1) CSV 읽기 ===
df = pd.read_csv("assets/train_data.csv")  # 파일명/경로만 맞춰주세요
col_out = "LLM Output" if "LLM Output" in df.columns else df.columns[-1]

# === 2) 함수 호출 분해 ===
call_pat = re.compile(r"<function_([A-Z]{2})>\((.*?)\)")
def split_calls(s):
    if not isinstance(s, str): return []
    parts = [p.strip() for p in s.split(";") if p.strip()]
    calls = []
    for p in parts:
        m = call_pat.search(p)
        if not m:
            continue
        fn, argstr = m.group(1), m.group(2)
        calls.append((fn, argstr))
    return calls

# === 3) 인자 파서 ===
def parse_args(argstr):
    argstr = argstr.strip()
    if argstr == "" or argstr == "()":
        return {}
    # (a=b, c="...") 형태에서 괄호는 이미 제거되어 들어옴
    # value 정규화: True/False/None/숫자/"문자열" 처리
    args = {}
    # 빈 값 방지용: split by comma respecting quotes -> 간단케이스 우선
    # 따옴표 이중(""...""), 한글 위치명 등 처리
    tokens = []
    depth, buf, in_str = 0, [], False
    i=0
    while i < len(argstr):
        ch = argstr[i]
        if ch == '"' and i+1 < len(argstr) and argstr[i+1] == '"':
            # 연속 쌍따옴표 -> 하나의 따옴표로 치환
            buf.append('"')
            i += 2
            continue
        if ch == ',' and not in_str and depth == 0:
            tokens.append(''.join(buf).strip()); buf=[]
        else:
            if ch == '"':
                in_str = not in_str
            elif ch in ('(', ')'):
                depth += 1 if ch=='(' else -1
            buf.append(ch)
        i+=1
    if buf:
        tokens.append(''.join(buf).strip())

    for t in tokens:
        if '=' not in t:
            continue
        k, v = t.split('=', 1)
        k, v = k.strip(), v.strip()
        # 타입 캐스팅 시도
        if v in ("True", "False", "None"):
            val = ast.literal_eval(v)
        else:
            # 숫자
            try:
                if '.' in v:
                    val = float(v)
                else:
                    val = int(v)
            except:
                # 양끝 따옴표 제거
                if len(v) >= 2 and ((v[0] == '"' and v[-1] == '"') or (v[0]=="'" and v[-1]=="'")):
                    val = v[1:-1]
                else:
                    val = v
        args[k] = val
    return args

# === 4) 호출 테이블 만들기 ===
rows = []
for idx, row in df.iterrows():
    calls = split_calls(row[col_out])
    for order, (fn, argstr) in enumerate(calls):
        args = parse_args(argstr)
        rows.append({
            "row_id": idx,
            "call_order": order,
            "func": fn,
            "raw_args": argstr,
            **{f"arg.{k}": v for k, v in args.items()}
        })
calls_df = pd.DataFrame(rows)

# === 5) 지표: 함수 빈도 ===
func_freq = calls_df["func"].value_counts().rename_axis("func").reset_index(name="count")

# === 6) 지표: 전역 인자 키/값 타입 분포 ===
def type_name(v):
    if v is None: return "none"
    return type(v).__name__
arg_cols = [c for c in calls_df.columns if c.startswith("arg.")]
melt = calls_df.melt(id_vars=["func", "row_id", "call_order"], value_vars=arg_cols, var_name="arg", value_name="val").dropna(subset=["val"])
melt["arg_key"] = melt["arg"].str.replace("^arg\\.", "", regex=True)
melt["val_type"] = melt["val"].map(type_name)

arg_key_freq = melt["arg_key"].value_counts().rename_axis("arg_key").reset_index(name="count")
arg_type_freq = melt.groupby(["arg_key","val_type"]).size().reset_index(name="count")

# === 6-1) 지표: 함수별 인자 키 빈도 ===
func_arg_key_freq = (
    melt.groupby(["func", "arg_key"])
        .size()
        .reset_index(name="count")
        .sort_values(["func", "count", "arg_key"], ascending=[True, False, True])
)


# === 6-2) 지표: 함수별 인자 값 분포 ===
# (값이 많은 경우 테이블이 클 수 있음)
func_arg_val_freq = (
    melt.groupby(["func", "arg_key", "val"])
        .size()
        .reset_index(name="count")
        .sort_values(["func", "arg_key", "count"], ascending=[True, True, False])
)

# === 6-3) 지표: 함수별 인자 값 리스트 ===
func_arg_val_list = (
    melt.groupby(["func", "arg_key"])["val"]
        .apply(lambda x: sorted(set(map(str, x))))
        .reset_index(name="values")
)

# Save argument value list to JSON
func_arg_val_list.to_json("assets/train_data.json", orient="records", force_ascii=False, indent=2)

# Save query list JSON for quick reference
query_list = []
for _, r in func_arg_val_list.iterrows():
    func = r["func"]
    arg_key = r["arg_key"]
    for v in r["values"]:
        query_list.append({
            "query": f"{func}.{arg_key}={v}",
            "func": func,
            "arg_key": arg_key,
            "value": v
        })

import json
with open("assets/query_list.json", "w", encoding="utf-8") as f:
    json.dump(query_list, f, ensure_ascii=False, indent=2)
print("Saved query list to assets/query_list.json")

# === 6-4) Save YAML from func_arg_val_list ===
try:
    import yaml  # PyYAML
except ImportError:
    yaml = None

def _infer_type_and_cast(values):
    vals = [str(v).strip() for v in values]
    lower = [v.lower() for v in vals]
    # boolean (true/false)
    if set(lower).issubset({"true", "false"}):
        return "boolean", [v.lower() == "true" for v in vals]
    # boolean (0/1)
    if set(vals).issubset({"0", "1"}):
        return "boolean", [v == "1" for v in vals]
    # numeric
    def _is_num(s):
        try:
            float(s)
            return True
        except:
            return False
    if all(_is_num(v) for v in vals):
        casted = []
        for v in vals:
            f = float(v)
            # keep integers as int
            if v.isdigit() or (v.startswith("-") and v[1:].isdigit()):
                casted.append(int(f))
            else:
                casted.append(f)
        return "number", casted
    # default string
    return "string", vals

# Build YAML structure
func_to_args = {}
for _, r in func_arg_val_list.iterrows():
    func = r["func"]
    arg_key = r["arg_key"]
    values = r["values"]
    arg_type, enum_vals = _infer_type_and_cast(values)
    # de-duplicate while preserving order
    enum_vals = list(dict.fromkeys(enum_vals))
    func_to_args.setdefault(func, []).append({
        "name": arg_key,
        "type": arg_type,
        "enum": enum_vals,
    })

func_descriptions = {
    "CE": "Provides reports or results about the air purification process (status, duration, or outcome).",
    "BP": "Controls basic power or operational modes (enable/disable, type switching).",
    "BS": "Toggles simple enable/disable state of a subsystem.",
    "CS": "Selects content or category types among limited options.",
    "DW": "Switches display or device modes with four type options.",
    "EF": "Triggers a predefined action event.",
    "EN": "Selects environment or entity type across multiple levels (1–7).",
    "EW": "Retrieves or sets environmental type values (negative, neutral, positive).",
    "FF": "Specifies a detailed physical position or location name (rooms, facilities, seats).",
    "GN": "Controls or queries generic enable/disable state.",
    "GQ": "Queries or sets gender-related attributes (male, female, unknown).",
    "HG": "Queries or sets theme-related options (basic themes).",
    "HI": "Chooses a binary type (two-level classification).",
    "HS": "Selects a three-level classification (negative, neutral, positive).",
    "HW": "Sets type or mode with six available options.",
    "ID": "Retrieves identification state.",
    "IH": "Gets or scans input/hardware status.",
    "IO": "Specifies a geographic location and timeframe for queries (e.g., weather, air quality).",
    "JS": "Selects a numeric type (1–10 levels).",
    "KP": "Chooses type with two possible levels.",
    "MO": "Queries or specifies mobility/travel locations with timeframe.",
    "MV": "Toggles get/mute state for media or audio.",
    "NK": "Controls or queries speed levels (from -1 to 4).",
    "NN": "Toggles enable/disable state for a neural or network feature.",
    "PC": "Selects a type among multiple categorical values.",
    "QD": "Queries or classifies using discrete type values.",
    "QT": "Toggles or queries enable state for quick/test features.",
    "SB": "Selects among five type levels (scenario or setting).",
    "SC": "Toggles enable/disable state for a control feature.",
    "SV": "Sets type among five categories (service or state).",
    "WN": "Adjusts brightness or queries its state.",
    "YA": "Controls movement along defined steps (0–2).",
    "ZV": "Adjusts or queries volume levels.",
    "ZX": "Enables/disables a feature toggle.",
}

functions_yaml = []
for func in sorted(func_to_args.keys()):
    functions_yaml.append({
        "name": func,
        "description": func_descriptions.get(func, f"{func} function"),
        "args": func_to_args[func],
    })

yaml_obj = {"functions": functions_yaml}

out_yaml_path = "assets/function_from_train_data.yaml"
if yaml is not None:
    with open(out_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_obj, f, allow_unicode=True, sort_keys=False)
else:
    # Fallback: minimal YAML writer
    def _dump_yaml(obj, indent=0, f=None):
        sp = "  " * indent
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    f.write(f"{sp}{k}:\n")
                    _dump_yaml(v, indent + 1, f)
                else:
                    if isinstance(v, str):
                        vv = '"' + v.replace('"', '\\"') + '"'
                    elif isinstance(v, bool):
                        vv = str(v).lower()
                    else:
                        vv = str(v)
                    f.write(f"{sp}{k}: {vv}\n")
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    f.write(f"{sp}- \n")
                    _dump_yaml(item, indent + 1, f)
                else:
                    if isinstance(item, str):
                        vv = '"' + item.replace('"', '\\"') + '"'
                    elif isinstance(item, bool):
                        vv = str(item).lower()
                    else:
                        vv = str(item)
                    f.write(f"{sp}- {vv}\n")

    with open(out_yaml_path, "w", encoding="utf-8") as f:
        _dump_yaml(yaml_obj, f)

print(f"Saved YAML to {out_yaml_path}")

# === 7) 지표: 함수별 인자 사용률 ===
func_arg_pivot = (melt
    .assign(present=1)
    .pivot_table(index="func", columns="arg_key", values="present", aggfunc="sum", fill_value=0)
    .astype(int)
    .reset_index())

# === 8) 다중 호출 시퀀스(빅램) ===
pairs = []
for (rid), g in calls_df.groupby("row_id"):
    g2 = g.sort_values("call_order")
    for a, b in zip(g2["func"], g2["func"].shift(-1)):
        if pd.notna(b):
            pairs.append((a, b))
pair_freq = pd.DataFrame(pairs, columns=["func_A","func_B"]).value_counts().rename("count").reset_index()

# === 9) 이상치 후보 ===
# location이 "0" 또는 0, 빈 인자 호출
placeholder_loc = melt[(melt["arg_key"]=="location") & (melt["val"].astype(str).str.strip().isin(['0', '""0""']))]
empty_args = calls_df[calls_df["raw_args"].str.strip().eq("")]

# === 10) 결과 미리보기(상위 10개씩) ===
print("함수 사용 빈도 (Top 10)")
print(func_freq, "\n")

print("인자 키 빈도 (Top 10)")
print(arg_key_freq, "\n")

print("함수→함수 시퀀스 (Top 10)")
print(pair_freq, "\n")

print("placeholder location 예시 (최대 5건)")
print(placeholder_loc, "\n")

print("빈 인자 함수 호출 예시 (최대 5건)")
print(empty_args[["row_id","func","raw_args"]])

print("함수별 인자 키 빈도 (전체)")
print(func_arg_key_freq, "\n")

print("함수별 인자 값 분포 (전체)")
print(func_arg_val_freq, "\n")

print("함수별 인자 값 리스트 (전체)")
print(func_arg_val_list, "\n")

# === 11) 원본 데이터에서 쿼리/LLM Output을 함수별로 모으기 ===
# 컬럼 감지
index_col = None
for cand in ["Index", "id", "ID", "index"]:
    if cand in df.columns:
        index_col = cand
        break
query_col = None
for cand in ["Query(한글)", "Query", "query", "prompt", "input"]:
    if cand in df.columns:
        query_col = cand
        break

# calls_df(row_id, func, call_order, raw_args ...) -> 원본과 join
join_cols = {"row_id": [], "func": [], "call_order": [], "raw_args": [], "Index": [], "Query": [], "LLM Output": []}
for _, r in calls_df.iterrows():
    rid = r["row_id"]
    join_cols["row_id"].append(rid)
    join_cols["func"].append(r["func"])
    join_cols["call_order"].append(r["call_order"])
    join_cols["raw_args"].append(r["raw_args"])
    # 안전 접근
    join_cols["Index"].append(df.loc[rid, index_col] if index_col else rid)
    join_cols["Query"].append(df.loc[rid, query_col] if query_col else None)
    join_cols["LLM Output"].append(df.loc[rid, col_out])

gather_df = pd.DataFrame(join_cols)

# 함수별 그룹 저장: JSON(단일) + 함수별 CSV
by_func = {}
for func, g in gather_df.groupby("func"):
    g_sorted = g.sort_values(["row_id", "call_order"]).reset_index(drop=True)
    by_func[func] = g_sorted[["Index", "Query", "LLM Output", "call_order", "raw_args"]].to_dict(orient="records")

# 디렉토리 생성
import os
os.makedirs("assets/by_function", exist_ok=True)

# 단일 JSON
with open("assets/by_function/by_function.json", "w", encoding="utf-8") as f:
    json.dump(by_func, f, ensure_ascii=False, indent=2)

# 함수별 CSV
for func, recs in by_func.items():
    out_csv = os.path.join("assets/by_function", f"{func}.csv")
    pd.DataFrame(recs).to_csv(out_csv, index=False)

print("Saved grouped files to assets/by_function/ (JSON + per-function CSV)")

# === 12) 전체 리스트 JSON 저장 (요약 없음) ===
# calls_df + 원본 전체 컬럼을 한 레코드에 묶어 저장
full_records = []
arg_prefix = "arg."
for _, r in calls_df.iterrows():
    rid = int(r["row_id"]) if pd.notna(r["row_id"]) else None
    base = {
        "row_id": rid,
        "func": r["func"],
        "call_order": int(r["call_order"]) if pd.notna(r["call_order"]) else None,
        "raw_args": r["raw_args"],
        "args": {}
    }
    # 파싱된 인자들(arg.*)을 args 딕셔너리로 평탄화
    for c in calls_df.columns:
        if c.startswith(arg_prefix):
            k = c[len(arg_prefix):]
            v = r[c]
            if pd.notna(v):
                base["args"][k] = v
    # 원본 행 전체를 포함
    orig_row = df.loc[rid].to_dict() if rid is not None else {}
    base["original"] = orig_row
    full_records.append(base)

import json, os
os.makedirs("assets/by_function", exist_ok=True)
with open("assets/by_function/all_calls.json", "w", encoding="utf-8") as f:
    json.dump(full_records, f, ensure_ascii=False, indent=2)

# 함수별 풀 리스트(JSON)도 별도로 저장
by_func_full = {}
for rec in full_records:
    by_func_full.setdefault(rec["func"], []).append(rec)
with open("assets/by_function/by_function_full.json", "w", encoding="utf-8") as f:
    json.dump(by_func_full, f, ensure_ascii=False, indent=2)

print("Saved full lists to assets/by_function/all_calls.json and by_function_full.json")

# 요약 프린트: 함수별 샘플 수
func_summary = gather_df.groupby("func").size().rename("samples").reset_index().sort_values(["samples","func"], ascending=[False, True])
print("함수별 샘플 수 요약")
print(func_summary)