import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import bm25s

from student.common.models import MinimalSearchResults, MinimalSource
from student.common.tokenizer import tokenize


@dataclass(frozen=True)
class SearchIndex:
    chunks: list[dict[str, object]]
    retriever: Any


def load_chunks(processed_dir: Path) -> list[dict[str, object]]:
    chunks_path = processed_dir / "chunks" / "chunks.json"
    try:
        data = json.loads(chunks_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(
            f"unable to read chunks from {chunks_path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid chunks JSON in {chunks_path}: {error}") \
            from error

    if not isinstance(data, list):
        raise ValueError(f"{chunks_path}: expected a list")
    return data


def load_bm25(processed_dir: Path) -> Any:
    bm25_path = processed_dir / "bm25_index"
    if not bm25_path.is_dir():
        raise ValueError(f"BM25 index not found in {bm25_path}")
    return bm25s.BM25.load(bm25_path)


def load_search_index(processed_dir: Path) -> SearchIndex:
    return SearchIndex(
        chunks=load_chunks(processed_dir),
        retriever=load_bm25(processed_dir),
    )


def source_from_chunk(chunk: dict[str, object]) -> MinimalSource:
    file_path = chunk["file_path"]
    first = chunk["first_character_index"]
    last = chunk["last_character_index"]
    if not isinstance(file_path, str):
        raise ValueError("chunk file_path must be a string")
    if not isinstance(first, int) or not isinstance(last, int):
        raise ValueError("chunk offsets must be integers")
    return MinimalSource(
        file_path=file_path,
        first_character_index=first,
        last_character_index=last,
    )


def process(
    query: str,
    k: int,
    search_index: SearchIndex,
    question_id: str = "manual",
) -> MinimalSearchResults:

    query_tokens = tokenize(query)
    retrieved = search_index.retriever.retrieve(
        [query_tokens],
        corpus=search_index.chunks,
        k=min(k, len(search_index.chunks)),
        show_progress=False,
    )
    row = retrieved.documents[0]
    if hasattr(row, "tolist"):
        row = row.tolist()
    retrieved_chunks = cast(list[dict[str, object]], row)

    return MinimalSearchResults(
        question_id=question_id,
        question_str=query,
        retrieved_sources=[
            source_from_chunk(chunk)
            for chunk in retrieved_chunks
        ],
    )
