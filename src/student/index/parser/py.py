import ast

from student.index.models import Chunk, File
from student.index.parser.text import chunk_text


def line_starts(text: str) -> list[int]:
    """Return the character index at the start of every source line.

    Args:
        text: Python source text.

    Returns:
        Character offsets for each line boundary.
    """
    starts = [0]
    starts.extend(index + 1 for index, char in enumerate(text) if char == "\n")
    starts.append(len(text))
    return starts


def symbol_span(
    starts: list[int],
    node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[int, int] | None:
    """Return a top-level symbol span, including its decorators.

    Args:
        starts: Character offsets for each line boundary.
        node: Abstract syntax tree node to locate.

    Returns:
        Start and end offsets, or ``None`` when unavailable.
    """
    end_lineno = node.end_lineno
    if end_lineno is None:
        return None
    first_lineno = node.lineno
    if node.decorator_list:
        first_lineno = min(first_lineno, node.decorator_list[0].lineno)
    if end_lineno >= len(starts):
        return None
    return starts[first_lineno - 1], starts[end_lineno]


def chunk_python(file: File, max_chunk_size: int) -> list[Chunk]:
    """Partition Python around top-level symbols without omitting code.

    Args:
        file: Python file to split.
        max_chunk_size: Maximum number of characters per chunk.

    Returns:
        Chunks covering the complete Python source.
    """
    try:
        tree = ast.parse(file.text, filename=file.path.as_posix())
    except (SyntaxError, ValueError):
        return chunk_text(file, max_chunk_size)

    starts = line_starts(file.text)
    boundaries = {0, len(file.text)}
    symbol_types = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in tree.body:
        if isinstance(node, symbol_types):
            span = symbol_span(starts, node)
            if span is not None:
                boundaries.update(span)

    ordered_boundaries = sorted(boundaries)
    groups: list[tuple[int, int]] = []
    group_start = ordered_boundaries[0]
    group_end = group_start
    for start, end in zip(ordered_boundaries, ordered_boundaries[1:]):
        if group_end == group_start or end - group_start <= max_chunk_size:
            group_end = end
        else:
            groups.append((group_start, group_end))
            group_start, group_end = start, end
    if group_end > group_start:
        groups.append((group_start, group_end))

    chunks: list[Chunk] = []
    for start, end in groups:
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
