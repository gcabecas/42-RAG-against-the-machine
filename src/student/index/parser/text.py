from student.index.models import Chunk, File


def chunk_text(
        file: File,
        max_chunk_size: int,
        base_offset: int = 0) -> list[Chunk]:
    chunks: list[Chunk] = []

    for start in range(0, len(file.text), max_chunk_size):
        end = min(start + max_chunk_size, len(file.text))
        text = file.text[start:end]
        absolute_start = base_offset + start
        absolute_end = base_offset + end
        if not text:
            continue
        file_path = file.path.as_posix()
        chunks.append(
            Chunk(
                chunk_id=f"{file_path}:{absolute_start}:{absolute_end}",
                file_path=file_path,
                text=text,
                first_character_index=absolute_start,
                last_character_index=absolute_end,
            )
        )

    return chunks
