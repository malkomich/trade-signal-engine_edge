SHELL := /bin/bash

.PHONY: bootstrap test run

bootstrap:
	uv sync

test:
	uv run pytest

run:
	uv run python -m trade_signal_edge

