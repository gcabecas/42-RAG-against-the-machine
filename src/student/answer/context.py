from dataclasses import dataclass
from typing import cast

from student.common.models import MinimalSource


ContextBySource = dict[tuple[str, int, int], str]


@dataclass(frozen=True)
class SourceContext:
    source: MinimalSource
    text: str


def source_key(source: MinimalSource) -> tuple[str, int, int]:
    return (
        source.file_path,
        source.first_character_index,
        source.last_character_index,
    )


def make_contexts(chunks: list[dict[str, object]]) -> ContextBySource:
    return {
        (
            cast(str, chunk["file_path"]),
            cast(int, chunk["first_character_index"]),
            cast(int, chunk["last_character_index"]),
        ): cast(str, chunk["text"])
        for chunk in chunks
    }


def load_context(
    source: MinimalSource,
    context_by_source: ContextBySource,
) -> SourceContext:
    key = source_key(source)
    if key not in context_by_source:
        raise ValueError(
            "context not found for "
            f"{source.file_path}:"
            f"{source.first_character_index}-"
            f"{source.last_character_index}"
        )
    return SourceContext(source=source, text=context_by_source[key])
