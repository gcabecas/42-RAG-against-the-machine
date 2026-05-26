from pathlib import Path

from student.index.models import File, FileType


def file_type(path: Path) -> FileType:
    suffix = path.suffix.lower()
    if suffix == ".py":
        return FileType.PYTHON
    if suffix == ".md":
        return FileType.MARKDOWN
    return FileType.TEXT


def read_text(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    except UnicodeDecodeError:
        return None

    return text.replace("\r\n", "\n").replace("\r", "\n")


def list_file(root: Path) -> list[File]:
    if not root.is_dir():
        raise ValueError(f"repository root does not exist: {root}")

    file_list = []

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
        file_list.append(File(path=path, type=file_type(path), text=text))
    return file_list
