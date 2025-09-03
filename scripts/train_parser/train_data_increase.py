# -*- coding: utf-8 -*-
"""
train_data_increase.py
----------------------
Generate all replacement combinations for dataset rows by applying JSONL word-mapping rules
to specific text columns (default: "Query(한글)", "LLM Output").

Mapping JSONL format (one object per line):
    {"ori": ["월요일", "화요일"], "sr": ["수요일", "목요일"]}

Meaning:
  - For any occurrence of *any* token in `ori`, replace (globally) with *one* choice from `sr`.
  - When multiple mapping lines exist, the script composes them, producing the cartesian set
    of possibilities across rules ("모든 경우의 수").
  - The original (unmodified) row is kept as well, unless --drop-original is passed.

Notes:
  - Replacements are done as plain substring replacements (case-sensitive) to support Korean text.
    If you need word-boundary aware behavior, adjust `apply_one_rule_to_text()` accordingly.
  - Deduplication is applied at the row level using the values of target columns.

Usage:
    python train_data_increase.py \
      --input-csv inputs/data.csv \
      --mapping-jsonl inputs/mapping.jsonl \
      --output-csv outputs/data_expanded.csv \
      --cols "Query(한글)" "LLM Output"

    Optional flags:
      --drop-original    (do not include the original unmodified row)
      --max-expansions N (cap total variants per original row; useful to prevent explosion)
      --id-suffix        (append -Rk to the Index for generated variants)

Example
-------
Input CSV:
    Index,Query(한글),LLM Output
    0ABOVW,월요일 자카르타 대기질을 화면에 보여줘 다음으로 자격증 준비하고 있니,"&lt;function_IO&gt;(timeframe=3, location=자카르타);&lt;function_MR&gt;()&lt;end&gt;"
    0AHOUT,음성인식 켜져있어,&lt;function_QT&gt;(get=True)&lt;end&gt;

Mapping JSONL:
    {"ori":["월요일","화요일"], "sr":["수요일","목요일"]}

Output: rows including all combinations where 월요일/화요일 are swapped to 수요일/목요일 in the specified columns.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Iterable, Tuple, Set

import pandas as pd


@dataclass(frozen=True)
class Rule:
  ori: Tuple[str, ...]
  sr: Tuple[str, ...]


def read_rules(jsonl_path: Path) -> List[Rule]:
  rules: List[Rule] = []
  with jsonl_path.open("r", encoding="utf-8") as f:
    for ln, line in enumerate(f, start=1):
      line = line.strip()
      if not line:
        continue
      try:
        obj = json.loads(line)
      except json.JSONDecodeError as e:
        raise ValueError(f"[mapping.jsonl] Line {ln} JSON error: {e}") from e
      if "ori" not in obj or "sr" not in obj:
        raise ValueError(f"[mapping.jsonl] Line {ln} must contain 'ori' and 'sr' keys")
      ori_list = tuple(map(str, obj["ori"]))
      sr_list = tuple(map(str, obj["sr"]))
      if not ori_list or not sr_list:
        raise ValueError(f"[mapping.jsonl] Line {ln} 'ori' and 'sr' must be non-empty lists")
      rules.append(Rule(ori=ori_list, sr=sr_list))
  return rules


def apply_one_rule_to_text(text: str, rule: Rule, replacement: str) -> str:
  """Replace all occurrences of any token in rule.ori with `replacement` (global, substring)."""
  out = text
  for token in rule.ori:
    if token:
      out = out.replace(token, replacement)
  return out


def extract_location_value(output_text: str) -> str | None:
  """
  Extract the value next to 'location=' in LLM Output.
  Handles optional spaces and optional single/double quotes.
  Examples it should capture:
    location=자카르타
    location="자카르타"
    location = '서울'
  It will stop before delimiters like , ) ; ] } or whitespace.
  """
  if not output_text:
    return None
  pattern = r"location\s*=\s*(?P<q>['\"])?(?P<val>[^'\"\s,);\]}]+)(?P=q)?"
  m = re.search(pattern, output_text)
  if m:
    return m.group("val")
  return None


def conditional_loc_replace(row: pd.Series) -> pd.Series:
  """
  If LLM Output contains 'location=XXX', and the exact same string XXX
  appears in Query(한글), replace those occurrences in Query(한글) with '<loc>'.
  """
  qcol = "Query(한글)"
  ocol = "LLM Output"
  if qcol not in row or ocol not in row:
    return row
  loc_val = extract_location_value(str(row.get(ocol, "")))
  if loc_val and loc_val in str(row.get(qcol, "")):
    row[qcol] = str(row[qcol]).replace(loc_val, "<loc>")
    row[ocol] = str(row[ocol]).replace(loc_val, "<loc>")
  return row

def conditional_pos_replace(row: pd.Series) -> pd.Series:
  """
  If LLM Output contains 'position=XXX', and the exact same string XXX
  appears in Query(한글), replace those occurrences in Query(한글) with '<pos>'.
  """
  qcol = "Query(한글)"
  ocol = "LLM Output"
  if qcol not in row or ocol not in row:
    return row
  pattern = r"position\s*=\s*(?P<q>['\"])?(?P<val>[^'\"\s,);\]}]+)(?P=q)?"
  m = re.search(pattern, str(row.get(ocol, "")))
  if m:
    pos_val = m.group("val")
    if pos_val and pos_val in str(row.get(qcol, "")):
      row[qcol] = str(row[qcol]).replace(pos_val, "<pos>")
      row[ocol] = str(row[ocol]).replace(pos_val, "<pos>")
  return row


def expand_row_by_rules(
    row: pd.Series,
    target_cols: List[str],
    rules: List[Rule],
    keep_original: bool = True,
    max_expansions: int | None = None,
) -> List[pd.Series]:
  """
  For a single row, generate all combinations by applying each rule with each replacement option.
  We perform composition across rules: for each rule, either
    - keep as-is, OR
    - choose one sr value and replace all ori tokens with that sr (for all target columns).
  This yields the cartesian product of choices across rules.

  Deduplication is performed to avoid identical variants.
  """
  # Start with the original variant
  base_variant = row.copy()
  variants = [base_variant] if keep_original else []

  # Each element in the working set is a pd.Series
  working = [row.copy()]

  for rule in rules:
    next_working: List[pd.Series] = []
    for var in working:
      # Choice 1: do not apply this rule
      next_working.append(var.copy())

      # Determine if the rule is applicable (any ori token exists in any target col)
      applicable = any(any(tok in str(var[col]) for tok in rule.ori) for col in target_cols)
      if applicable:
        for rep in rule.sr:
          new_var = var.copy()
          for col in target_cols:
            new_var[col] = apply_one_rule_to_text(str(new_var[col]), rule, rep)
          next_working.append(new_var)
      # If not applicable, nothing to do (only the non-apply branch)

    # Cap explosion if requested
    if max_expansions is not None and len(next_working) > max_expansions:
      next_working = next_working[:max_expansions]

    working = next_working

  # Deduplicate by the tuple of target column values (preserve first occurrence order)
  seen: Set[Tuple[str, ...]] = set()
  deduped: List[pd.Series] = []
  for var in working if keep_original else working[1:]:
    key = tuple(str(var[col]) for col in target_cols)
    if key not in seen:
      seen.add(key)
      deduped.append(var)

  return deduped if deduped else [row.copy()] if keep_original else []


def assign_variant_indices(
    rows: List[pd.Series], original_index: str | int | None, use_suffix: bool
) -> None:
  """Mutate rows to set a unique Index if present/desired."""
  if not rows:
    return
  if "Index" not in rows[0].index:
    # No Index column; nothing to assign
    return

  if not use_suffix:
    # Leave indices as-is (may collide)
    return

  # Ensure first row keeps original Index, others get suffixed
  if original_index is None:
    base = "IDX"
  else:
    base = str(original_index)

  for i, r in enumerate(rows):
    if i == 0:
      r["Index"] = base
    else:
      r["Index"] = f"{base}-R{i}"


def main(argv: List[str]) -> int:
  ap = argparse.ArgumentParser(description="Expand dataset by applying mapping JSONL replacements.")
  ap.add_argument("--input-csv", type=Path, required=True, help="Path to input CSV file")
  ap.add_argument("--mapping-jsonl", type=Path, required=True, help="Path to mapping JSONL file")
  ap.add_argument("--output-csv", type=Path, required=True, help="Path to write expanded CSV")
  ap.add_argument(
    "--cols",
    nargs="+",
    default=["Query(한글)", "LLM Output"],
    help="Target text columns to apply replacements (space-separated)",
  )
  ap.add_argument("--drop-original", action="store_true", help="Exclude the original unmodified row")
  ap.add_argument("--max-expansions", type=int, default=None, help="Cap variants per original row")
  ap.add_argument("--id-suffix", action="store_true", help="Append -Rk to 'Index' for generated variants")
  args = ap.parse_args(argv)

  if not args.input_csv.exists():
    raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
  if not args.mapping_jsonl.exists():
    raise FileNotFoundError(f"Mapping JSONL not found: {args.mapping_jsonl}")

  df = pd.read_csv(args.input_csv, dtype=str).fillna("")
  rules = read_rules(args.mapping_jsonl)

  # Validate columns
  for c in args.cols:
    if c not in df.columns:
      raise ValueError(f"Column '{c}' not found in CSV. Available: {list(df.columns)}")

  expanded_rows: List[pd.Series] = []

  for _, row in df.iterrows():
    # Apply conditional location-based replacement on Query(한글) first
    row = conditional_loc_replace(row)
    row = conditional_pos_replace(row)

    variants = expand_row_by_rules(
        row=row,
        target_cols=args.cols,
        rules=rules,
        keep_original=not args.drop_original,
        max_expansions=args.max_expansions,
    )

    # Drop any variants that still contain the placeholder "<loc>" in any target column
    filtered_variants = []
    for v in variants:
        has_loc = any("<loc>" in str(v[c]) for c in args.cols)
        if not has_loc:
            filtered_variants.append(v)

    # If nothing remains after filtering, skip appending for this row
    if not filtered_variants:
        continue

    assign_variant_indices(filtered_variants, row.get("Index", None), use_suffix=args.id_suffix)
    expanded_rows.extend(filtered_variants)

  out_df = pd.DataFrame(expanded_rows, columns=df.columns)
  args.output_csv.parent.mkdir(parents=True, exist_ok=True)
  out_df.to_csv(args.output_csv, index=False, encoding="utf-8")

  print(f"[OK] Expanded {len(df)} rows -> {len(out_df)} rows")
  return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
