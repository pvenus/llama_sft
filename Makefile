# 주석은 이렇게 작성

.PHONY: venv sft  # 실제 파일이 아닌 '가상 타겟'임을 표시

# 가상환경 생성
venv:
	python3 -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install --upgrade pip && pip install -r requirements.txt

# 학습 실행
sft:
	. .venv/bin/activate && python -m scripts.main
# Makefile — cross-platform (macOS/Linux/Windows) venv + run
# NOTE: We avoid `source`/`activate` and call the venv binaries directly.

.PHONY: venv sft help

# Detect OS and set venv executables
ifeq ($(OS),Windows_NT)
  PY := .venv\\Scripts\\python.exe
  PIP := .venv\\Scripts\\pip.exe
else
  PY := .venv/bin/python
  PIP := .venv/bin/pip
endif

help:
	@echo "Targets:"
	@echo "  venv   - Create venv and install requirements"
	@echo "  sft    - Run training/eval pipeline (scripts.main)"

# Create venv and install deps
venv:
	python -m venv .venv
	$(PY) -m pip install -U pip
	$(PIP) install -r requirements.txt

# Run SFT pipeline (dataset prepare -> train -> eval)
sft: | venv
	$(PY) -m scripts.main