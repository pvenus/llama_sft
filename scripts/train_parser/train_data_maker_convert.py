#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
폴더 내 CSV들을 모두 로드한 뒤, function 컬럼의 "<...>" 사이 함수명으로 그룹핑.
- 각 함수별 전체 행 수의 절반(n//2)만 '페어링 풀'에 사용.
- 풀에서 서로 다른 함수 2개를 무작위로 짝지어(순서도 랜덤) 2개짜리 레코드 생성.
- 사용하지 못한 샘플:
    1) 페어링 풀에 남은 잔여분
    2) 절반 샘플링에서 제외된 나머지 절반
  => 위 모두를 1개짜리로 출력하며 function 끝에 항상 "<end>"를 붙인다.
- 같은 행(샘플)은 재사용하지 않음.
- 최종 CSV 컬럼: index, query, function (디버그 컬럼 제거)
"""

import argparse
import glob
import os
import random
import re
import sys
from collections import defaultdict
from typing import List, Dict, Tuple

import pandas as pd

# "<something>"에서 something 추출
FUNC_NAME_RE = re.compile(r"<\s*([^>\s]+)\s*>")

def extract_func_name(func_cell: str) -> str:
    """function 셀에서 첫 번째 '<...>' 안의 텍스트를 함수명으로 추출."""
    if not isinstance(func_cell, str):
        return ""
    m = FUNC_NAME_RE.search(func_cell)
    return m.group(1) if m else ""

def ensure_end_suffix(fn_call: str) -> str:
    """function 문자열 끝에 '<end>'가 없으면 붙인다."""
    s = (fn_call or "").strip()
    return s if s.endswith("<end>") else f"{s}<end>"

# --- 안전 CSV 로더 ------------------------------------------------------------
def _safe_read_csv(path: str) -> pd.DataFrame:
    """
    CSV 파싱이 깨진 파일도 최대한 살려 읽는다.
    - 기본: utf-8-sig (C 엔진)
    - 실패 시: python 엔진 + on_bad_lines='skip' + escapechar='\\'
    - 그래도 실패 시: utf-8로 재시도
    - 모든 컬럼 dtype=str
    """
    try:
        return pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    except Exception:
        try:
            return pd.read_csv(
                path,
                encoding="utf-8-sig",
                dtype=str,
                engine="python",
                sep=",",
                quotechar='"',
                doublequote=True,
                escapechar="\\",
                on_bad_lines="skip",
            )
        except Exception:
            return pd.read_csv(
                path,
                encoding="utf-8",
                dtype=str,
                engine="python",
                sep=",",
                quotechar='"',
                doublequote=True,
                escapechar="\\",
                on_bad_lines="skip",
            )

def load_all_csv(input_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not paths:
        print(f"[ERROR] No CSV files found in: {input_dir}", file=sys.stderr)
        sys.exit(1)

    dfs = []
    for p in paths:
        df = _safe_read_csv(p)

        # 표준 컬럼 맞추기
        needed = {"index", "query", "function"}
        if not needed.issubset(df.columns):
            colmap = {}
            for c in df.columns:
                lc = c.strip().lower()
                if lc in ("idx", "id"):
                    colmap[c] = "index"
                elif lc in ("q", "utterance", "text"):
                    colmap[c] = "query"
                elif lc in ("func", "functions", "call"):
                    colmap[c] = "function"
            df = df.rename(columns=colmap)

        missing = needed - set(df.columns)
        if missing:
            print(f"[WARN] Skip file (missing columns {missing}): {p}", file=sys.stderr)
            continue

        # 개행/공백 정리
        for col in ["index", "query", "function"]:
            df[col] = df[col].astype(str).str.replace("\r\n", " ").str.replace("\n", " ").str.strip()

        df["__src_path"] = os.path.basename(p)
        dfs.append(df[["index", "query", "function", "__src_path"]])

    if not dfs:
        print("[ERROR] No usable CSVs after column checks.", file=sys.stderr)
        sys.exit(1)

    return pd.concat(dfs, ignore_index=True)

# --- 풀 구성: 절반은 페어링용, 절반은 단일 출력용으로 --------------------------------
def build_pools(df: pd.DataFrame, seed: int):
    """
    반환:
      - pair_pool: Dict[func_name, List[(row_id, query, fncall)]]
      - singles_initial: List[(query, fncall)]  # 절반 샘플링에서 제외된 나머지
    """
    random.seed(seed)
    df = df.copy()
    df["__func_name"] = df["function"].apply(extract_func_name)
    df = df[df["__func_name"] != ""].reset_index(drop=True)

    by_func: Dict[str, List[Tuple[int, str, str]]] = defaultdict(list)
    for ridx, row in df.iterrows():
        by_func[row["__func_name"]].append((ridx, str(row["query"]), str(row["function"])))

    pair_pool: Dict[str, List[Tuple[int, str, str]]] = {}
    singles_initial: List[Tuple[str, str]] = []

    for fname, items in by_func.items():
        n = len(items)
        use_n = n // 2  # 페어링 풀에 넣을 개수
        if n == 0:
            continue
        random.shuffle(items)
        # 절반은 페어링 풀
        selected = items[:use_n]
        if selected:
            pair_pool[fname] = selected.copy()
        # 남은 절반은 처음부터 단일 출력 후보
        for _, q, fn in items[use_n:]:
            singles_initial.append((q, fn))

    return pair_pool, singles_initial
from collections import deque
import heapq

def make_pairs_and_collect_singles(pair_pool: Dict[str, List[Tuple[int, str, str]]], seed: int):
    """
    pair_pool에서 만들 수 있는 '서로 다른 함수' 페어를 최대 개수로 생성.
    - 내부적으로 각 함수 리스트를 섞고(deque), 남은 개수가 가장 많은 두 함수를
      매 스텝마다 꺼내 1개씩 사용 → 페어 생성.
    - 더 이상 서로 다른 두 함수가 남지 않으면 종료.
    - 남은 항목은 단일로 반환.
    반환:
      - pairs: List[Dict[str, str]]  # index/query/function
      - leftovers_single: List[(query, fncall)]
    """
    random.seed(seed)

    # 1) 각 함수 아이템을 섞고 deque로 변환(무작위 선택 보장)
    pools: Dict[str, deque] = {}
    for f, items in pair_pool.items():
        items = items.copy()
        random.shuffle(items)
        pools[f] = deque(items)

    # 2) (남은개수, 함수명) 최대 힙 구성. 파이썬 heapq는 최소힙이므로 음수로 저장
    heap = [(-len(dq), f) for f, dq in pools.items() if len(dq) > 0]
    heapq.heapify(heap)

    pairs: List[Dict[str, str]] = []
    rec_id = 0

    # 3) 항상 남은 개수가 가장 큰 두 함수를 뽑아 서로 다른 함수 페어 생성
    while len(heap) >= 2:
        n1, f1 = heapq.heappop(heap)  # 가장 큼
        n2, f2 = heapq.heappop(heap)  # 두 번째로 큼

        # 각 함수에서 하나씩 사용
        ridx1, q1, fn1 = pools[f1].pop()
        ridx2, q2, fn2 = pools[f2].pop()

        # 순서 랜덤
        if random.random() < 0.5:
            fn_combined = f"{fn1};{fn2}<end>"
            q_combined = f"{q1} 다음으로 {q2}"
        else:
            fn_combined = f"{fn2};{fn1}<end>"
            q_combined = f"{q2} 다음으로 {q1}"

        pairs.append({
            "index": f"P{rec_id:06d}",
            "query": q_combined,
            "function": fn_combined,
        })
        rec_id += 1

        # 남은 개수 갱신 후 다시 힙에 넣기
        if len(pools[f1]) > 0:
            heapq.heappush(heap, (-(len(pools[f1])), f1))
        if len(pools[f2]) > 0:
            heapq.heappush(heap, (-(len(pools[f2])), f2))

    # 4) 남은 것은 단일로
    leftovers_single: List[Tuple[str, str]] = []
    for f, dq in pools.items():
        while dq:
            _, q, fn = dq.pop()
            leftovers_single.append((q, fn))

    return pairs, leftovers_single

def main():
    parser = argparse.ArgumentParser(description="CSV 합치기 & 함수 2개 랜덤 페어링 + 단일 생성기")
    parser.add_argument("--input-dir", required=True, help="CSV들이 있는 폴더 경로")
    parser.add_argument("--output", default="paired_output.csv", help="결과 CSV 경로")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    args = parser.parse_args()

    df_all = load_all_csv(args.input_dir)

    # 절반은 페어링 풀, 절반은 단일 초기 후보
    pair_pool, singles_initial = build_pools(df_all, seed=args.seed)

    # 페어 만들기 + 페어링 풀 잔여분 수거(단일)
    pairs, leftovers_single = make_pairs_and_collect_singles(pair_pool, seed=args.seed)

    print(f"{len(pairs)} {len(leftovers_single)}");

    # 최종 단일 목록 = (초기 단일 후보) + (페어링 실패 잔여)
    singles_all: List[Dict[str, str]] = []
    start_idx = len(pairs)
    idx_counter = start_idx

    for q, fn in singles_initial + leftovers_single:
        singles_all.append({
            "index": f"P{idx_counter:06d}",
            "query": q,
            "function": ensure_end_suffix(fn),
        })
        idx_counter += 1

    # 최종 결과: 페어 + 단일
    final_rows = pairs + singles_all
    out_df = pd.DataFrame(final_rows, columns=["index", "query", "function"])  # 디버그 컬럼 제거
    out_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    # 리포트
    print(f"[DONE] Wrote {len(out_df)} rows -> {args.output}")
    print(f" - Pairs:  {len(pairs)}")
    print(f" - Singles:{len(singles_all)}")

if __name__ == "__main__":
    main()
