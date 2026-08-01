from typing import cast

from student.common.models import MinimalSource


def make_contexts(
    chunks: list[dict[str, object]],
) -> dict[tuple[str, int, int], str]:
    """Map indexed source locations to their original text.

    Args:
        chunks: Serialized chunks loaded from the persisted index.

    Returns:
        A mapping from source coordinates to chunk text.
    """
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
    context_by_source: dict[tuple[str, int, int], str],
) -> str:
    """Return the indexed text matching one retrieved source.

    Args:
        source: Source location whose text is required.
        context_by_source: Indexed text keyed by source coordinates.

    Returns:
        The text stored for the source.

    Raises:
        ValueError: If the source is absent from the index.
    """
    key = (
        source.file_path,
        source.first_character_index,
        source.last_character_index,
    )
    if key not in context_by_source:
        raise ValueError(
            "context not found for "
            f"{source.file_path}:"
            f"{source.first_character_index}-"
            f"{source.last_character_index}"
        )
    return context_by_source[key]
