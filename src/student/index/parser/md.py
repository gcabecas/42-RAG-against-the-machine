import re

from student.index.models import Chunk, File
from student.index.parser.text import chunk_text


def section_starts(file: File) -> list[int]:
    heading_pattern = re.compile(r"(?m)^#{1,4}\s+.+$")
    headings = [match.start() for match in heading_pattern.finditer(file.text)]
    if not headings:
        return []

    starts = [0] if headings[0] != 0 else []
    starts.extend(headings)
    starts.append(len(file.text))
    return starts


def make_chunk(file: File, start: int, end: int) -> Chunk:
    file_path = file.path.as_posix()
    return Chunk(
        chunk_id=f"{file_path}:{start}:{end}",
        file_path=file_path,
        text=file.text[start:end],
        first_character_index=start,
        last_character_index=end,
    )


def chunk_section(file: File, start: int, end: int,
                  max_chunk_size: int) -> list[Chunk]:
    text = file.text[start:end]
    if not text.strip():
        return []
    if len(text) <= max_chunk_size:
        return [make_chunk(file, start, end)]
    return [
        chunk
        for chunk in chunk_text(
            File(path=file.path, type=file.type, text=text),
            max_chunk_size,
            base_offset=start,
        )
    ]


def chunk_markdown(file: File, max_chunk_size: int) -> list[Chunk]:
    starts = section_starts(file)
    if not starts:
        return chunk_text(file, max_chunk_size)

    chunks: list[Chunk] = []
    for index in range(len(starts) - 1):
        chunks.extend(chunk_section(
            file, starts[index], starts[index + 1], max_chunk_size))
    return chunks
