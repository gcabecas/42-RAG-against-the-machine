from pathlib import Path

from pydantic import BaseModel

from student.common.models import MinimalSource


class File(BaseModel):
    """Represent one readable corpus file.

    Args:
        path: Path of the file relative to the current project.
        text: Decoded file contents.
    """

    path: Path
    text: str


class Chunk(MinimalSource):
    """Extend a source location with its indexed text.

    Args:
        file_path: Corpus-relative path to the source file.
        first_character_index: Start offset of the source range.
        last_character_index: End offset of the source range.
        text: Exact source text covered by the inherited location.
    """

    text: str
