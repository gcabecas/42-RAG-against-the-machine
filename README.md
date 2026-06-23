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

Prepare the vLLM repository if it is not already extracted:

```bash
mkdir -p data/raw
unzip -q -n info/given/vllm-0.10.1.zip -d data/raw
```

Build the index:

```bash
uv run python -m student index --max_chunk_size 2000
```

Run a single search:

```bash
uv run python -m student search "How to configure OpenAI server?" --k 10
```

Answer a single question:

```bash
uv run python -m student answer "How to configure OpenAI server?" --k 10
```

Search a dataset:

```bash
uv run python -m student search_dataset \
  --dataset_path data/datasets_public/public/AnsweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results
```

Evaluate search results:

```bash
uv run python -m student evaluate \
  --student_results_path data/output/search_results/dataset_docs_public.json \
  --dataset_path data/datasets_public/public/AnsweredQuestions/dataset_docs_public.json \
  --k 10
```

Generate answers from search results:

```bash
uv run python -m student answer_dataset \
  --student_search_results_path data/output/search_results/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer
```

The Makefile also exposes shortcuts:

```bash
make install
make index
make search QUERY="How to configure OpenAI server?" K=10
make search_dataset DATASET_PATH=data/datasets_public/public/AnsweredQuestions/dataset_docs_public.json K=10
make evaluate STUDENT_RESULTS_PATH=data/output/search_results/dataset_docs_public.json DATASET_PATH=data/datasets_public/public/AnsweredQuestions/dataset_docs_public.json K=10
make answer QUERY="What is vLLM?" ANSWER_K=10
make lint
```

## System Architecture

The pipeline is split into five main parts:

- `src/student/index/`: reads files, chunks them, and builds the BM25 index.
- `src/student/search/`: loads the saved index and retrieves top-k chunks.
- `src/student/evaluate/`: computes recall@k against annotated datasets.
- `src/student/answer/`: builds the prompt context and calls Qwen.
- `src/student/app/`: exposes the command-line interface with Python Fire.

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

- Python files: parsed with `ast`; top-level classes and functions are kept as
  independent chunks when they fit in the size limit.
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

## Evaluation And Performance

The evaluator computes recall@k by comparing retrieved source ranges with the
expected source ranges from the answered datasets.

Local public dataset results observed on this repository:

```text
Indexing time: about 6 seconds
Docs recall@5: 0.86
Code recall@5: 0.76
Search throughput: about 100 questions in less than 1 second
```

Subject thresholds:

```text
Docs recall@5 >= 0.80
Code recall@5 >= 0.50
Indexing time <= 5 minutes
Warm retrieval <= 90 seconds for 1000 questions
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

# AI

AI assistance was used to create the readme