import json
import csv
import re
import itertools
from pathlib import Path

INPUT_JSON = "input.json"   # 사용자가 제공한 배열 구조 JSON
OUTPUT_CSV = "function_queries.csv"

def slot_type(slot: str) -> str:
    # 1) <s1>, <o>, <v>, <q> 형태 우선
    m = re.match(r"<([a-zA-Z]+)", slot)
    if m:
        return m.group(1)[0].lower()

    # 2) s_g_bp_1, s_l_1 같은 언더스코어 형태 지원
    m = re.match(r"([a-zA-Z])_", slot)  # 맨 앞 글자 + '_' 패턴
    if m:
        return m.group(1).lower()

    return ""

def combine_same_type_tokens(tokens):
    """
    같은 유형 묶음 내 토큰들 사이에 '붙여쓰기' 또는 '띄어쓰기' 조합을 모두 생성.
    예) ["에이아이","모드"] -> ["에이아이모드", "에이아이 모드"]
    예) ["에이아이","모드","설정"] -> 모든 구분자 조합(2^(n-1)) 생성
    """
    n = len(tokens)
    if n == 1:
        return [tokens[0]]

    # 각 경계에 대해 '' 또는 ' ' 조합 생성
    variants = []
    for seps_bits in itertools.product(["", " "], repeat=n-1):
        parts = [tokens[0]]
        for i, sep in enumerate(seps_bits, start=1):
            parts.append(sep + tokens[i])
        variants.append("".join(parts))
    return variants

def build_queries_from_format(fmt: str, tags: dict):
    """
    하나의 datas 포맷("<s1><s2><o><v>" 등)에 대해 tags를 채워
    규칙(동일 유형 붙/띄, 상이 유형 간 띄어쓰기 필수)을 적용한 쿼리들을 생성.
    """
    # 1) 포맷의 슬롯 순서 추출
    slots = re.findall(r"<[^>]+>", fmt)
    if not slots:
        return []

    # 2) 포맷에 필요한 모든 슬롯이 tags에 존재하는지 확인 (없으면 스킵)
    for sl in slots:
        if sl not in tags:
            return []

    # 3) 각 슬롯의 값 목록
    values_lists = [tags[sl] for sl in slots]

    # 4) 슬롯 유형 단위로 그룹핑 (연속 동일 유형 묶음)
    types = [slot_type(sl) for sl in slots]
    groups = []  # [(group_slots_indices)]
    start = 0
    for i in range(1, len(slots) + 1):
        if i == len(slots) or types[i] != types[start]:
            groups.append(list(range(start, i)))
            start = i

    # 5) 모든 슬롯 값 조합을 순회하면서, 그룹 단위 결합 변형 생성
    queries = []
    for combo in itertools.product(*values_lists):
        # combo는 슬롯별 선택된 값들의 튜플
        group_variants = []
        for g in groups:
            # 같은 유형 그룹의 실제 토큰들
            group_tokens = [combo[i] for i in g]
            # 같은 유형 내부는 붙여쓰기/띄어쓰기 모든 변형 생성
            group_variants.append(combine_same_type_tokens(group_tokens))

        # 6) 서로 다른 유형 그룹 간에는 띄어쓰기 필수이므로,
        #    그룹 변형들(product) 간에 ' '로 연결
        for gv_combo in itertools.product(*group_variants):
            queries.append(" ".join(gv_combo))

    return queries

def merge_tag_sources(*sources):
    """
    sources에는 dict 또는 dict 리스트를 전달.
    앞에 오는 소스일수록 우선(표현 우선순위), 값은 중복 제거 + 순서 보존.
    예) merge_tag_sources(block_tags_list, global_tags_list)
    """
    merged = {}
    for src in sources:
        if not src:
            continue
        dicts = src if isinstance(src, list) else [src]
        for d in dicts:
            if not isinstance(d, dict):
                continue
            for k, vals in d.items():
                if not isinstance(vals, list):
                    continue
                acc = merged.setdefault(k, [])
                for v in vals:
                    if v not in acc:
                        acc.append(v)
    return merged

def main():
    data = json.loads(Path(INPUT_JSON).read_text(encoding="utf-8"))

    # 최상위는 {"datas": [ ...block... ]}
    blocks = data.get("datas", [])
    common_tags = data.get("tags", [])
    rows, seen = [], set()
    idx = 1

    for block in blocks:
        func_name = block.get("fname", "")
        # 동일 포맷 중복 제거
        fmts = list(dict.fromkeys(block.get("formats", [])))
        if not block.get("tags"):
            continue
        tags = block.get("tags", [])

        merged_tags = merge_tag_sources(tags, common_tags)

        for fmt in fmts:
            for q in build_queries_from_format(fmt, merged_tags):
                key = (q, func_name)
                if key in seen:
                    continue
                seen.add(key)
                rows.append([idx, q, func_name])
                idx += 1

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "query", "function"])
        writer.writerows(rows)

    print(f"✅ CSV 생성 완료: {OUTPUT_CSV} (총 {len(rows)} 행)")

if __name__ == "__main__":
    main()
