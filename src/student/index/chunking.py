import json
from pathlib import Path

from tqdm import tqdm

from student.index.models import Chunk, File, FileType
from student.index.parser.md import chunk_markdown
from student.index.parser.py import chunk_python
from student.index.parser.text import chunk_text


def chunk_file(file: File, chunk_size: int) -> list[Chunk]:
    if file.type == FileType.PYTHON:
        return chunk_python(file, chunk_size)
    if file.type == FileType.MARKDOWN:
        return chunk_markdown(file, chunk_size)
    return chunk_text(file, chunk_size)


def save_chunks(output_dir: Path, chunk_list: list[Chunk]) -> None:
    chunks_dir = output_dir / "chunks"
    chunks_path = chunks_dir / "chunks.json"

    try:
        chunks_dir.mkdir(parents=True, exist_ok=True)
        with chunks_path.open("w") as chunks_file:
            chunks = []
            for chunk in tqdm(chunk_list, desc="Saving chunks", unit="chunk"):
                chunks.append(chunk.model_dump())
            json.dump(chunks, chunks_file)
    except OSError as error:
        raise ValueError(
            f"unable to save chunks to {chunks_path}: {error}"
        ) from error


def select_method(
    file_list: list[File],
    chunk_size: int,
    output_dir: Path,
) -> list[Chunk]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")

    chunk_list: list[Chunk] = []
    for file in tqdm(file_list, desc="Chunking files", unit="file"):
        chunk_list.extend(chunk_file(file, chunk_size))

    save_chunks(output_dir, chunk_list)
    return chunk_list
