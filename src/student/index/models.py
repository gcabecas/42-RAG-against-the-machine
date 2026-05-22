import enum
from pathlib import Path

from pydantic import BaseModel


class FileType(str, enum.Enum):
    PYTHON = "python"
    MARKDOWN = "markdown"
    TEXT = "text"


class File(BaseModel):
    path: Path
    type: FileType
    text: str


class Chunk(BaseModel):
    chunk_id: str
    file_path: str
    text: str
    first_character_index: int
    last_character_index: int
