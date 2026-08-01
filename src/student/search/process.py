import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import bm25s

from student.common.models import MinimalSearchResults, MinimalSource
from student.common.tokenizer import tokenize


@dataclass(frozen=True)
class SearchIndex:
    """Hold persisted chunks and their loaded BM25 retriever.

    Args:
        chunks: Serialized chunks used as the retrieval corpus.
        retriever: Loaded BM25 retriever.
    """

    chunks: list[dict[str, object]]
    retriever: Any


def load_chunks(processed_dir: Path) -> list[dict[str, object]]:
    """Load serialized chunks from an index directory.

    Args:
        processed_dir: Directory containing persisted index artifacts.

    Returns:
        Serialized chunks used as the retrieval corpus.

    Raises:
        ValueError: If the serialized value is not a list.
    """
    chunks_path = processed_dir / "chunks" / "chunks.json"
    data = json.loads(chunks_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{chunks_path}: expected a list")
    return cast(list[dict[str, object]], data)


def load_search_index(processed_dir: Path) -> SearchIndex:
    """Load all artifacts required for retrieval.

    Args:
        processed_dir: Directory containing persisted index artifacts.

    Returns:
        Loaded chunks and their BM25 retriever.
    """
    bm25_path = processed_dir / "bm25_index"
    if not bm25_path.is_dir():
        raise ValueError(f"BM25 index not found in {bm25_path}")
    return SearchIndex(
        chunks=load_chunks(processed_dir),
        retriever=bm25s.BM25.load(bm25_path),
    )


def process(
    query: str,
    k: int,
    search_index: SearchIndex,
    question_id: str = "manual",
) -> MinimalSearchResults:
    """Retrieve ranked source locations for one query.

    Args:
        query: Search query.
        k: Maximum number of sources to retrieve.
        search_index: Loaded chunks and BM25 retriever.
        question_id: Identifier associated with the query.

    Returns:
        Ranked source locations for the query.

    Raises:
        ValueError: If the query has no tokens or matching source.
    """
    query_tokens = tokenize(query)
    if not query_tokens:
        raise ValueError("query must contain searchable characters")
    retrieved = search_index.retriever.retrieve(
        [query_tokens],
        corpus=search_index.chunks,
        k=min(k, len(search_index.chunks)),
        show_progress=False,
    )
    row = retrieved.documents[0]
    if hasattr(row, "tolist"):
        row = row.tolist()
    chunks = cast(list[dict[str, object]], row)

    score_row = retrieved.scores[0]
    if hasattr(score_row, "tolist"):
        score_row = score_row.tolist()
    scores = cast(list[float], score_row)
    retrieved_chunks = [
        chunk
        for chunk, score in zip(chunks, scores)
        if float(score) > 0
    ]
    if not retrieved_chunks:
        raise ValueError("query did not match the indexed corpus")

    return MinimalSearchResults(
        question_id=question_id,
        question=query,
        retrieved_sources=[
            MinimalSource.model_validate(chunk)
            for chunk in retrieved_chunks
        ],
    )
