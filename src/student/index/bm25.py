from pathlib import Path

import bm25s
from tqdm import tqdm

from student.common.tokenizer import tokenize
from student.index.models import Chunk


def chunk_search_text(chunk: Chunk) -> str:
    """Combine path hints and content for lexical indexing.

    Args:
        chunk: Source chunk to represent in the index.

    Returns:
        Searchable text containing path hints and source content.
    """
    path = Path(chunk.file_path)
    return f"{path.parent.name} {path.name} {path.name} {chunk.text}"


def gen_bm25(chunk_list: list[Chunk], output_dir: Path) -> None:
    """Build and persist a BM25 index for the supplied chunks.

    Args:
        chunk_list: Source chunks to index.
        output_dir: Directory in which the index is saved.

    Raises:
        ValueError: If no chunks are supplied or the index cannot be saved.
    """
    if not chunk_list:
        raise ValueError("cannot build a BM25 index without chunks")

    tokenized_chunks = [
        tokenize(chunk_search_text(chunk))
        for chunk in tqdm(chunk_list, desc="Tokenizing chunks", unit="chunk")
    ]
    retriever = bm25s.BM25()
    retriever.index(tokenized_chunks, show_progress=True)
    try:
        retriever.save(output_dir / "bm25_index", show_progress=True)
    except OSError as error:
        raise ValueError(f"unable to save BM25 index: {error}") from error
