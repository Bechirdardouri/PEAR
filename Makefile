# --- PEAR Makefile -----------------------------------------------------
#
# Common targets:
#   make install        editable install + dev deps
#   make test           pytest tests/ -q
#   make figures        regenerate all 9 figures from committed parquets
#   make smoke          run the package smoke test (synthetic, CPU, ~1s)
#   make clean          remove caches, build artifacts, generated figures
#   make all            install + test + figures
# ----------------------------------------------------------------------

PY        ?= python3
PIP       ?= $(PY) -m pip
PYTEST    ?= $(PY) -m pytest

.DEFAULT_GOAL := help
.PHONY: help install dev test smoke figures audit-figures proposal-figures \
        clean clean-figures distclean all

help:
	@printf "PEAR -- Perceptual Edge Audit for RL in vision-language models\n\n"
	@printf "  \033[1mmake install\033[0m           editable install\n"
	@printf "  \033[1mmake dev\033[0m               editable install + dev/test extras\n"
	@printf "  \033[1mmake test\033[0m              run the full test suite (50 tests, ~2s)\n"
	@printf "  \033[1mmake smoke\033[0m             run the synthetic VEST smoke test\n"
	@printf "  \033[1mmake figures\033[0m           regenerate all 9 figures\n"
	@printf "  \033[1mmake audit-figures\033[0m     regenerate the 5 audit figures only\n"
	@printf "  \033[1mmake proposal-figures\033[0m  regenerate the 4 proposal schematics only\n"
	@printf "  \033[1mmake clean\033[0m             remove caches and build artifacts\n"
	@printf "  \033[1mmake distclean\033[0m         clean + remove generated figures\n"
	@printf "  \033[1mmake all\033[0m               install + test + figures\n"

install:
	$(PIP) install -e .

dev:
	$(PIP) install -e ".[dev]"

test:
	PYTHONPATH=. $(PYTEST) tests/ -q

smoke:
	$(PY) -m pear smoke

audit-figures:
	$(PY) scripts/make_figures.py

proposal-figures:
	$(PY) scripts/make_proposal_figures.py

figures: audit-figures proposal-figures

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} \;
	find . -type f -name "*.pyc" -delete

clean-figures:
	rm -f results/figures/*.png results/figures/*.pdf
	rm -f proposal/figures/*.png proposal/figures/*.pdf

distclean: clean clean-figures

all: install test figures
