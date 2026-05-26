.PHONY: install run debug clean lint index

SOURCE_ROOT ?= data/raw/vllm-0.10.1
OUTPUT_DIR ?= data/processed
MAX_CHUNK_SIZE ?= 2000

install:
	uv sync

run:
	uv run python -m student

debug:
	uv run python -m pdb -m student

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .mypy_cache

lint:
	uv run flake8 src
	MYPYPATH=src uv run mypy --explicit-package-bases src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

index:
	uv run python -m student index --source_root $(SOURCE_ROOT) --output_dir $(OUTPUT_DIR) --max_chunk_size $(MAX_CHUNK_SIZE)
