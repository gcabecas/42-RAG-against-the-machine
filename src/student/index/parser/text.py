from student.index.models import Chunk, File


def chunk_text(
    file: File,
    max_chunk_size: int,
    base_offset: int = 0,
) -> list[Chunk]:
    """Split plain text into consecutive fixed-size chunks.

    Args:
        file: File or subsection to split.
        max_chunk_size: Maximum number of characters per chunk.
        base_offset: Absolute offset of the supplied text in its source file.

    Returns:
        Consecutive chunks covering the supplied text.
    """
    chunks: list[Chunk] = []
    for start in range(0, len(file.text), max_chunk_size):
        end = min(start + max_chunk_size, len(file.text))
        chunks.append(
            Chunk(
                file_path=file.path.as_posix(),
                text=file.text[start:end],
                first_character_index=base_offset + start,
                last_character_index=base_offset + end,
            )
        )
    return chunks
