import ast

from student.index.models import Chunk, File
from student.index.parser.text import chunk_text


def line_starts(text: str) -> list[int]:
    starts = [0]
    starts.extend(index + 1 for index, char in enumerate(text) if char == "\n")
    starts.append(len(text))
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


def symbol_span(starts: list[int], node: ast.AST) -> tuple[int, int] | None:
    if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
        return None
    end_lineno = node.end_lineno
    if end_lineno is None:
        return None
    start_line = node.lineno - 1
    if start_line < 0 or end_lineno >= len(starts):
        return None
    return starts[start_line], starts[end_lineno]


def chunk_python(file: File, max_chunk_size: int) -> list[Chunk]:
    try:
        tree = ast.parse(file.text, filename=file.path.as_posix())
    except (SyntaxError, ValueError):
        return chunk_text(file, max_chunk_size)

    starts = line_starts(file.text)
    chunks: list[Chunk] = []
    for node in tree.body:
        if not isinstance(
            node,
            (ast.ClassDef,
             ast.FunctionDef,
             ast.AsyncFunctionDef)):
            continue
        span = symbol_span(starts, node)
        if span is None:
            continue
        start, end = span
        text = file.text[start:end]
        if len(text) <= max_chunk_size:
            chunks.append(make_chunk(file, start, end))
        else:
            chunks.extend(
                chunk_text(
                    File(path=file.path, type=file.type, text=text),
                    max_chunk_size,
                    base_offset=start,
                )
            )

    return chunks or chunk_text(file, max_chunk_size)
