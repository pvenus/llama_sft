# Makefile — cross-platform (macOS/Linux/Windows) venv + lock + run
.PHONY: venv lock install sft clean help

# Detect OS and set venv executables & requirements file
ifeq ($(OS),Windows_NT)
  SHELL := cmd
  .SHELLFLAGS := /C
  PY := .venv\Scripts\python.exe
  PIP := .venv\Scripts\pip.exe
  REQ := requirements-win.txt
  PYBOOT := py -3.12
  CHECK_REQ := if not exist requirements.in ( echo requirements.in not found & exit /b 1 )
else
  PY := .venv/bin/python
  PIP := .venv/bin/pip
  UNAME_S := $(shell uname -s)
  ifeq ($(UNAME_S),Darwin)
    REQ := requirements-mac.txt
  else
    REQ := requirements-linux.txt
  endif
  PYBOOT := $(shell command -v /opt/homebrew/bin/python3 || command -v python3 || echo /usr/bin/python3)
  CHECK_REQ := test -f requirements.in || { echo "requirements.in not found"; exit 1; }
endif

help:
	@echo "Targets:"
	@echo "  venv    - Create venv, install pip-tools"
	@echo "  lock    - Compile requirements.in -> $(REQ) (per-OS lockfile)"
	@echo "  install - Install deps from $(REQ)"
	@echo "  sft     - Run pipeline (scripts.main)"
	@echo "  clean   - Remove venv and lockfiles"

# 1) Create venv and ensure pip-tools is present
venv:
	$(PYBOOT) -m venv .venv
	"$(PY)" -m pip install -U pip pip-tools

# 2) Compile per-OS lockfile from requirements.in
#    - If you want hashes, add --generate-hashes
lock: | venv
	@$(CHECK_REQ)
	"$(PY)" -m piptools compile requirements.in -o "$(REQ)" --upgrade

install: | lock
	"$(PIP)" install -r "$(REQ)"
ifeq ($(OS),Windows_NT)
	"$(PIP)" uninstall torch torchvision torchaudio -y
	"$(PIP)" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
endif

# 4) Run your SFT pipeline
sft:
	"$(PY)" -m scripts.main

infer:
	"$(PY)" -m scripts.infer.infer

# 5) Cleanup
clean:
	-@rm -rf .venv 2>/dev/null || rmdir /S /Q .venv 2>nul
	-@rm -f requirements-linux.txt requirements-mac.txt requirements-win.txt 2>/dev/null || del /Q requirements-*.txt 2>nul

# === 변환 & JSONL 생성 ===
# === convert → test JSONL 경로 변수 (필요시 override 가능) ===
IN_CSV    ?= assets/convert/func_samples_10.csv
OUT_CSV   ?= assets/convert/func_samples_10.converted.csv
RULES     ?= assets/convert/function_rules.yaml
OUT_JSONL ?= assets/prompt/test_convert_msg.jsonl

convert_raw:
	"$(PY)" -m scripts.convert.function_converter --in "$(IN_CSV)" --out "$(OUT_CSV)" --config "$(RULES)" --verbose

convert_msg:
	"$(PY)" -m scripts.convert.function_converter --from-converted "$(OUT_CSV)" --emit-test-jsonl "$(OUT_JSONL)" --verbose

convert_raw_msg:
	"$(PY)" -m scripts.convert.function_converter --in "$(IN_CSV)" --out "$(OUT_CSV)" --config "$(RULES)" --emit-test-jsonl "$(OUT_JSONL)" --verbose

all:
	$(MAKE) venv
	$(MAKE) lock
	$(MAKE) install