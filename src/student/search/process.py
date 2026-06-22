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
    data = json.loads(chunks_path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"{chunks_path}: expected a list")
    return cast(list[dict[str, object]], data)


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
            MinimalSource.model_validate(chunk)
            for chunk in retrieved_chunks
        ],
    )
