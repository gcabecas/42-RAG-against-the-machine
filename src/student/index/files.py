from pathlib import Path

from student.index.models import File


def read_text(path: Path) -> str | None:
    """Read UTF-8 text without changing character offsets.

    Args:
        path: File to read.

    Returns:
        Decoded text, or ``None`` when the file cannot be read.
    """
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            return stream.read()
    except (OSError, UnicodeDecodeError):
        return None


def list_file(root: Path) -> list[File]:
    """Load every readable UTF-8 file below a corpus root.

    Args:
        root: Root directory of the corpus.

    Returns:
        Readable corpus files sorted by path.

    Raises:
        ValueError: If the root cannot be scanned.
    """
    if not root.is_dir():
        raise ValueError(f"repository root does not exist: {root}")

    file_list: list[File] = []

    try:
        paths = sorted(root.rglob("*"))
    except OSError as error:
        raise ValueError(
            f"unable to scan repository root {root}: {error}"
        ) from error

    for path in paths:
        if not path.is_file():
            continue
        text = read_text(path)
        if text is None:
            continue
        file_list.append(File(path=path, text=text))
    return file_list
