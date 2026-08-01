*This project has been created as part of the 42 curriculum by gcabecas.*

# RAG Against the Machine

## Description

This project implements a small Retrieval-Augmented Generation system for the
vLLM codebase. It indexes the repository, retrieves relevant code or
documentation chunks with BM25, and uses `Qwen/Qwen3-0.6B` to generate answers
grounded in the retrieved context.

The goal is to answer questions about vLLM while also measuring retrieval
quality with recall@k.

## Instructions

Install the dependencies:

```bash
uv sync
```

Extract the supplied v2 resources directly into the following layout:

```text
data/
├── raw/
│   └── vllm-0.10.1/
└── datasets/
    ├── AnsweredQuestions/
    │   ├── dataset_code_public.json
    │   └── dataset_docs_public.json
    └── UnansweredQuestions/
        ├── dataset_code_public.json
        └── dataset_docs_public.json
```

The supplied corpus, datasets, generated index, model weights, and outputs must
remain outside Git. The repository already ignores `data/`.

Build the index:

```bash
uv run python -m src index --max_chunk_size 2000
```

Run a single search:

```bash
uv run python -m src search "How to configure OpenAI server?" --k 10
```

Answer a single question:

```bash
uv run python -m src answer "How to configure OpenAI server?" --k 10
```

Search a dataset:

```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
```

Evaluate search results:

```bash
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10
```

Generate answers from search results:

```bash
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

The Makefile exposes the five mandatory project rules:

```bash
make install
make run
make debug
make clean
make lint
```

## System Architecture

The pipeline is split into six main parts:

- `src/student/index/`: reads files, chunks them, and builds the BM25 index.
- `src/student/search/`: loads the saved index and retrieves top-k chunks.
- `src/student/evaluate/`: computes recall@k against annotated datasets.
- `src/student/answer/`: builds the prompt context and calls Qwen.
- `src/student/app/`: exposes the command-line interface with Python Fire.
- `src/__main__.py`: exposes the required `python -m src` entry point and
  delegates command parsing to the Python Fire CLI.

Data flow:

```text
data/raw/vllm-0.10.1
    -> index
    -> data/processed/chunks + data/processed/bm25_index
    -> search
    -> retrieved_sources
    -> answer
    -> grounded natural language answer
```

## Chunking Strategy

The maximum chunk size is configurable with `--max_chunk_size` and is capped at
2000 characters.

Implemented strategies:

- Python files: parsed with `ast`; top-level symbol boundaries guide grouping
  while imports, constants, decorators, and module-level code remain covered.
- Markdown files: split by headings so documentation sections stay coherent.
- Other text files: split into fixed-size text chunks.

Small chunks improve precision but can lose context. Large chunks preserve more
context but make ranking less precise and increase the prompt size for the LLM.

## Retrieval Method

The retrieval system uses BM25 through the `bm25s` library. During indexing,
each chunk is tokenized and stored in a persistent BM25 index under
`data/processed/bm25_index`.

The tokenizer:

- lowercases tokens;
- splits `snake_case`;
- splits `CamelCase`;
- keeps useful identifiers and numbers.

The indexed text includes the parent directory and file name in addition to the
chunk content. This improves retrieval for questions that mention modules,
classes, endpoints, configuration files, or command names.

## Answer Generation

The default model is:

```text
Qwen/Qwen3-0.6B
```

The answer command first retrieves chunks, then sends their content to the model
with instructions to use only the provided sources. The generated answer is kept
short and source-grounded.

Supported model behavior is intentionally simple: the project relies on the
default Qwen model required by the subject and does not require a GPU to run,
although CUDA is used automatically when available.

A CPU smoke test with the locked dependencies successfully loaded the model and
generated an answer grounded in a retrieved vLLM source.

## Evaluation And Performance

The evaluator computes recall@k by comparing retrieved source ranges with the
expected source ranges from the answered datasets.

Local public dataset results observed with the v2 resources:

```text
Indexed files: 2818
Generated chunks: 17169
Indexing time: about 10 seconds
Docs recall@5: 0.82
Code recall@5: 0.828
Search throughput: 199 questions in about 5 seconds
```

Subject thresholds:

```text
Docs recall@5 >= 0.80
Code recall@5 >= 0.50
Indexing time <= 5 minutes
Warm retrieval <= 90 seconds for 200 questions
```

## Design Decisions

BM25 was chosen instead of embeddings because it is fast, deterministic, easy to
debug, and does not require model downloads for retrieval. This makes indexing
and batch search reliable during evaluation.

The main trade-off is lexical dependency: BM25 works best when question terms
appear in the retrieved files. The custom tokenizer and file-name weighting help
with code identifiers, module names, and API endpoints.

The index is stored on disk so search commands do not need to rebuild it each
time. This keeps cold start simple and warm retrieval fast.

## Challenges Faced

- Code questions often refer to symbols, file names, or implementation details
  instead of natural language documentation. The tokenizer was adjusted to split
  identifiers and improve matching.
- Documentation can contain long sections. Markdown chunking keeps sections
  readable while respecting the maximum chunk size.
- Answer generation must stay grounded. The prompt asks the model to use only
  retrieved context and to avoid unsupported claims.

## Resources

- vLLM documentation: https://docs.vllm.ai/
- BM25 overview: https://en.wikipedia.org/wiki/Okapi_BM25
- Pydantic documentation: https://docs.pydantic.dev/
- Python Fire documentation: https://google.github.io/python-fire/
- Hugging Face Transformers documentation: https://huggingface.co/docs/transformers
- Qwen model page: https://huggingface.co/Qwen/Qwen3-0.6B

## AI

  for readme and docstring
