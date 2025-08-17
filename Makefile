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
else
  PY := .venv/bin/python
  PIP := .venv/bin/pip
  UNAME_S := $(shell uname -s)
  ifeq ($(UNAME_S),Darwin)
    REQ := requirements-mac.txt
  else
    REQ := requirements-linux.txt
  endif
  PYBOOT := python3
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
	@if not exist requirements.in (echo requirements.in not found && exit 1)
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

all:
	$(MAKE) venv
	$(MAKE) lock
	$(MAKE) install