# 주석은 이렇게 작성

.PHONY: venv train merge infer  # 실제 파일이 아닌 '가상 타겟'임을 표시

# 가상환경 생성
venv:
	python3 -m venv .venv && . .venv/bin/activate && pip install -U pip

# 학습 실행
sft:
	. .venv/bin/activate && python scripts/main.py