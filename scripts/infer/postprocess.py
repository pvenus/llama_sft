import re
from time import perf_counter
from typing import List, Tuple

def postprocess_fn_calls(s: str) -> str:
    """
    입력 문자열을 표준 포맷으로 보정:
      '<function_xx>(a=1, b=2);<function_yy>()<end>'
    규칙:
      - 모든 " 제거
      - <fuction_> 는 <function_> 으로 교정
      - 첫 함수 태그 이전 프리픽스 제거
      - 등장 순서 유지, 함수 최대 2개
      - 태그 직후 괄호 있으면 인자 파싱(최대 2개 k=v 정규화)
      - 인자 없더라도 항상 () 출력
      - 항상 <end> 로 마무리
    """
    if not isinstance(s, str):
        s = str(s)

    s = s.replace('"', '')

    n = len(s)
    i = 0

    def match_fn_tag(idx: int):
        # '<function_' or '<fuction_'
        if idx >= n or s[idx] != '<':
            return None
        if s.startswith('function_', idx + 1):
            start = idx
            name_start = idx + 1 + len('function_')
            prefix = 'function_'
        elif s.startswith('fuction_', idx + 1):
            start = idx
            name_start = idx + 1 + len('fuction_')
            prefix = 'function_'  # 교정
        else:
            return None

        # 이름은 '>' 전까지 [a-zA-Z0-9_]+ 만 취합
        j = name_start
        name_chars = []
        while j < n and s[j] != '>':
            ch = s[j]
            if ch.isalnum() or ch == '_':
                name_chars.append(ch)
            j += 1
        if j >= n or s[j] != '>' or not name_chars:
            return None  # 불완전 태그
        name = ''.join(name_chars)
        end = j + 1
        return (start, end, prefix + name)

    # 첫 태그 위치 찾아 프리픽스 제거
    first = None
    j = 0
    while j < n:
        mt = match_fn_tag(j)
        if mt:
            first = mt[0]
            break
        j += 1
    if first is not None and first > 0:
        s = s[first:]
        n = len(s)

    # 다시 스캔하여 순서대로 최대 2개 수집
    i = 0
    calls = []
    while i < n and len(calls) < 2:
        mt = match_fn_tag(i)
        if not mt:
            i += 1
            continue
        tag_start, tag_end, fname = mt
        i = tag_end  # 태그 뒤로 이동

        # 공백 스킵
        while i < n and s[i].isspace():
            i += 1

        # 괄호가 있으면 인자 파싱
        args = []
        if i < n and s[i] == '(':
            i += 1
            arg_buf = []
            paren_open = 1
            # 간단한 괄호 닫힘까지 스캔 (중첩/인용문은 요구사항 범위 밖이라 단순히 처리)
            while i < n and paren_open > 0:
                ch = s[i]
                if ch == '(':
                    paren_open += 1
                elif ch == ')':
                    paren_open -= 1
                    if paren_open == 0:
                        i += 1
                        break
                if paren_open > 0:
                    arg_buf.append(ch)
                i += 1

            arg_text = ''.join(arg_buf).strip()
            if arg_text:
                # k=v, k=v,... → 최대 2개
                for piece in arg_text.split(','):
                    tok = piece.strip()
                    if '=' not in tok or not tok:
                        continue
                    k, v = tok.split('=', 1)
                    k, v = k.strip(), v.strip()
                    if not k:
                        continue
                    args.append((k, v))
                    if len(args) == 2:
                        break

        calls.append((fname, args))

    if not calls:
        return "<end>"

    out = []
    for fname, args in calls:
        if args:
            kv = ", ".join(f"{k}={v}" for k, v in args)
            out.append(f"<{fname}>({kv})")
        else:
            out.append(f"<{fname}>()")

    return ";".join(out) + "<end>"

def run_and_time_ms(s: str) -> Tuple[str, float]:
    t0 = perf_counter()
    out = postprocess_fn_calls(s)
    t1 = perf_counter()
    return out, (t1 - t0) * 1000.0

# --- 예시 ---
if __name__ == "__main__":
    tests = [
        'junk"zzz<fuction_aa>(x = 테스트)<function_ss> (a=abc)other <function_bb>(k=v)',
        'junk"zzz<fuction_aa>(x = y) other <function_bb>(k=v)',
        'junk"zzz<fuction_aa>(x = 테스트)<function_ss> (a=abc)other <function_bb>(k=v)',
        'noise <function_xx> foo <function_yy>(a=1, b=2, c=3)',
        '<function_one>() and <fuction_two>',
    ]
    for tx in tests:
        out, ms = run_and_time_ms(tx)
        print(out)
        print(f"{ms:.3f} ms")
