import json
import csv
import itertools

# 입력 jsonl 파일 (한 줄에 하나씩 JSON)
jsonl_file = "input.jsonl"
csv_file = "output.csv"

with open(jsonl_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

rows = []
idx = 0

for line in lines:
    data = json.loads(line.strip())
    func_name = data["name"]
    datas = data["datas"]        # ["<부사어구> <목적어구> <서술어>", ...]
    tags = {t["tag_name"]: t["word"] for t in data["tags"]}

    # datas의 각 패턴에 대해 가능한 조합 생성
    for pattern in datas:
        # 태그 순서 추출
        tag_seq = [seg.strip("<>") for seg in pattern.split()]
        # 각 태그에 해당하는 단어 리스트 꺼내기
        word_lists = [tags[tag] for tag in tag_seq]

        # 모든 조합(product)
        for combo in itertools.product(*word_lists):
            query = " ".join(combo)
            rows.append([idx, query, func_name])
            idx += 1

# CSV로 저장
with open(csv_file, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["index", "query", "function"])
    writer.writerows(rows)

print(f"CSV 저장 완료: {csv_file}")
