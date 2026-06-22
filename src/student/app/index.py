from pathlib import Path

from student.app.base import BaseCli
from student.index.bm25 import gen_bm25
from student.index.chunking import select_method
from student.index.files import list_file


class IndexCli(BaseCli):
    def index(
        self,
        source_root: str = BaseCli.DEFAULT_SOURCE_ROOT,
        output_dir: str = BaseCli.DEFAULT_OUTPUT_DIR,
        max_chunk_size: int | str = BaseCli.DEFAULT_MAX_CHUNK_SIZE,
    ) -> dict[str, str | int]:
        try:
            chunk_size = int(max_chunk_size)
        except (TypeError, ValueError):
            return self._error("index", "max_chunk_size must be an integer")

        if chunk_size < 1 or chunk_size > self.DEFAULT_MAX_CHUNK_SIZE:
            return self._error(
                "index",
                "max_chunk_size must be between 1 and 2000",
            )

        source_path = Path(source_root)
        output_path = Path(output_dir)
        if not source_path.is_dir():
            return self._error(
                "index",
                (
                    "source_root does not exist or is not a directory: "
                    f"{source_path}"
                ),
            )
        if output_path.exists() and not output_path.is_dir():
            return self._error(
                "index",
                f"output_dir exists but is not a directory: "
                f"{output_path}",
            )

        try:
            file_list = list_file(source_path)
            if not file_list:
                return self._error(
                    "index",
                    f"no text files found in: {source_path}",
                )

            chunk_list = select_method(
                file_list,
                chunk_size,
                output_path,
            )
            if not chunk_list:
                return self._error("index", "no chunks were generated")

            gen_bm25(chunk_list, output_path)
        except (OSError, ValueError) as error:
            return self._error("index", str(error))
        except Exception as error:
            return self._error(
                "index",
                f"unexpected indexing error: {error}",
            )

        return {
            "command": "index",
            "status": "ok",
            "source_root": source_path.as_posix(),
            "output_dir": output_path.as_posix(),
            "max_chunk_size": chunk_size,
            "files": len(file_list),
            "chunks": len(chunk_list),
            "bm25_documents": len(chunk_list),
        }
