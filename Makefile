.PHONY: install run test lint

install:
	uv sync --extra dev --python 3.12

run:
	uv run setmap-audio

test:
	uv run pytest -q

lint:
	uv run ruff check .
