PYTHON_BIN ?= /opt/homebrew/bin/python3.15
UV ?= uv

.PHONY: all run test dev clean

all: test

run:
	$(UV) run --python $(PYTHON_BIN) uvicorn app.main:app --host 127.0.0.1 --port 8000

dev:
	$(UV) run --python $(PYTHON_BIN) uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

test:
	$(UV) run --extra dev --python $(PYTHON_BIN) pytest tests/ -v

clean:
	rm -rf .pytest_cache __pycache__ app/__pycache__ tests/__pycache__ .coverage
