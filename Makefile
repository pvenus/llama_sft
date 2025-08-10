# Makefile — cross-platform (macOS/Linux/Windows) venv + run

.PHONY: venv sft help

# Detect OS and set venv executables & requirements file
ifeq ($(OS),Windows_NT)
  PY := .venv/Scripts/python.exe
  PIP := .venv/Scripts/pip.exe
  REQ := requirements-win.txt
else
  UNAME_S := $(shell uname -s)
  ifeq ($(UNAME_S),Darwin)
    PY := .venv/bin/python
    PIP := .venv/bin/pip
    REQ := requirements-mac.txt
  else
    PY := .venv/bin/python
    PIP := .venv/bin/pip
    REQ := requirements-linux.txt
  endif
endif

help:
	@echo "Targets:"
	@echo "  venv   - Create venv and install requirements ($(REQ))"
	@echo "  sft    - Run training/eval pipeline (scripts.main)"

# Create venv and install deps
venv:
	python -m venv .venv
	$(PY) -m pip install -U pip
	$(PIP) install -r $(REQ)

# Run SFT pipeline (dataset prepare -> train -> eval)
sft: | venv
	$(PY) -m scripts.main
