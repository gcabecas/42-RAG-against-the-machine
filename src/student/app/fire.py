from pathlib import Path
import fire as fire_lib

from student.index.files import list_file
from student.index.chunking import select_method


class StudentCli:
    DEFAULT_SOURCE_ROOT = "data/raw/vllm-0.10.1"
    DEFAULT_OUTPUT_DIR = "data/processed"
    DEFAULT_MAX_CHUNK_SIZE = 2000

    def index(
        self,
        source_root: str = DEFAULT_SOURCE_ROOT,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    ) -> dict[str, str | int]:

        if max_chunk_size < 1 or max_chunk_size > self.DEFAULT_MAX_CHUNK_SIZE:
            raise ValueError("max_chunk_size must be between 1 and 2000")

        file_list = list_file(Path(source_root))
        chunk_list = select_method(file_list, max_chunk_size)

        return {
            "command": "index",
            "source_root": Path(source_root).as_posix(),
            "output_dir": Path(output_dir).as_posix(),
            "max_chunk_size": max_chunk_size,
            "files": len(file_list),
            "chunks": len(chunk_list),
        }


def main() -> None:
    fire_lib.Fire(StudentCli)


if __name__ == "__main__":
    main()
