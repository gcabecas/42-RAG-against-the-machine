.PHONY: install run debug clean lint index search search_dataset evaluate

SOURCE_ROOT ?= data/raw/vllm-0.10.1
OUTPUT_DIR ?= data/processed
MAX_CHUNK_SIZE ?= 2000
QUERY ?=
K ?= 10
PROCESSED_DIR ?= $(OUTPUT_DIR)
DATASET_PATH ?= data/datasets_public/public/AnsweredQuestions/dataset_code_public.json
SAVE_DIRECTORY ?= data/output/search_results
STUDENT_RESULTS_PATH ?= $(SAVE_DIRECTORY)/$(notdir $(DATASET_PATH))

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

search:
	uv run python -m student search --query "$(QUERY)" --k $(K) --processed_dir $(PROCESSED_DIR)

search_dataset:
	uv run python -m student search_dataset --dataset_path $(DATASET_PATH) --k $(K) --processed_dir $(PROCESSED_DIR) --save_directory $(SAVE_DIRECTORY)

evaluate:
	uv run python -m student evaluate --student_results_path $(STUDENT_RESULTS_PATH) --dataset_path $(DATASET_PATH) --k $(K)
