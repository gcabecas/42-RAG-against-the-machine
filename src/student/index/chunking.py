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


def select_method(file_list: list[File], chunk_size: int) -> list[Chunk]:
    chunk_list: list[Chunk] = []
    for file in file_list:
        chunk_list.extend(chunk_file(file, chunk_size))
    return chunk_list
