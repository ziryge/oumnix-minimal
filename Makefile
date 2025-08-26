.PHONY: help install lint format test audit ci precommit

help:
	@echo "Targets: install lint format test audit ci precommit"

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install pytest ruff black pip-audit pre-commit

lint:
	ruff check --config .github/linters/.ruff.toml .

format:
	black --line-length 120 .

test:
	pytest -q

audit:
	python -m pip install --upgrade pip
	pip install pip-audit
	pip-audit -r requirements.txt || true

ci:
	$(MAKE) lint
	$(MAKE) test
	$(MAKE) audit

precommit:
	pre-commit run -a

release:
	python -m pip install --upgrade pip build
	python -m build
