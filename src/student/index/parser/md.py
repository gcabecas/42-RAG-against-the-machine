import re

from student.index.models import Chunk, File
from student.index.parser.text import chunk_text


def chunk_markdown(file: File, max_chunk_size: int) -> list[Chunk]:
    """Split Markdown by headings before applying the size limit.

    Args:
        file: Markdown file to split.
        max_chunk_size: Maximum number of characters per chunk.

    Returns:
        Chunks generated from the Markdown file.
    """
    headings = [
        match.start()
        for match in re.finditer(r"(?m)^#{1,4}\s+.+$", file.text)
    ]
    if not headings:
        return chunk_text(file, max_chunk_size)

    starts = ([0] if headings[0] != 0 else []) + headings + [len(file.text)]
    chunks: list[Chunk] = []
    for start, end in zip(starts, starts[1:]):
        text = file.text[start:end]
        if text.strip():
            chunks.extend(
                chunk_text(
                    File(path=file.path, text=text),
                    max_chunk_size,
                    base_offset=start,
                )
            )
    return chunks
