import csv
import json


def csv_to_jsonl(csv_path: str, jsonl_path: str):
    """
    CSV 파일을 JSONL 파일로 변환합니다.
    LLM Output에 포함된 큰따옴표(")는 제거합니다.

    Args:
        csv_path: 입력 CSV 경로
        jsonl_path: 출력 JSONL 경로
    """
    with open(csv_path, "r", encoding="utf-8") as csv_file, \
            open(jsonl_path, "w", encoding="utf-8") as jsonl_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            query = row["Query(한글)"].strip()
            output = row["LLM Output"].strip().replace('"', "")

            record = {
                "message": query,
                "expected": output
            }
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    csv_to_jsonl("assets/train_data.csv", "assets/prompt/test_user_msg.jsonl")
    print("변환 완료: output.jsonl")
