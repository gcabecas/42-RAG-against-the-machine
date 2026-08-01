import json
from pathlib import Path

from tqdm import tqdm

from student.index.models import Chunk, File
from student.index.parser.md import chunk_markdown
from student.index.parser.py import chunk_python
from student.index.parser.text import chunk_text


def chunk_file(file: File, chunk_size: int) -> list[Chunk]:
    """Dispatch one file to its dedicated chunking strategy.

    Args:
        file: Corpus file to split.
        chunk_size: Maximum number of characters per chunk.

    Returns:
        Chunks generated from the file.
    """
    suffix = file.path.suffix.lower()
    if suffix == ".py":
        return chunk_python(file, chunk_size)
    if suffix == ".md":
        return chunk_markdown(file, chunk_size)
    return chunk_text(file, chunk_size)


def save_chunks(output_dir: Path, chunk_list: list[Chunk]) -> None:
    """Persist chunks as JSON beside the BM25 index.

    Args:
        output_dir: Directory in which chunks are saved.
        chunk_list: Chunks to serialize.

    Raises:
        ValueError: If the chunks cannot be written.
    """
    chunks_dir = output_dir / "chunks"
    chunks_path = chunks_dir / "chunks.json"

    try:
        chunks_dir.mkdir(parents=True, exist_ok=True)
        with chunks_path.open("w", encoding="utf-8") as chunks_file:
            json.dump(
                [chunk.model_dump() for chunk in chunk_list],
                chunks_file,
            )
    except OSError as error:
        raise ValueError(
            f"unable to save chunks to {chunks_path}: {error}"
        ) from error


def select_method(
    file_list: list[File],
    chunk_size: int,
    output_dir: Path,
) -> list[Chunk]:
    """Chunk all files, validate the size, and save the result.

    Args:
        file_list: Corpus files to chunk.
        chunk_size: Maximum number of characters per chunk.
        output_dir: Directory in which chunks are saved.

    Returns:
        All chunks generated from the corpus.

    """
    chunk_list: list[Chunk] = []
    for file in tqdm(file_list, desc="Chunking files", unit="file"):
        chunk_list.extend(chunk_file(file, chunk_size))

    save_chunks(output_dir, chunk_list)
    return chunk_list
