from student.app.base import BaseCli
from student.index.bm25 import gen_bm25
from student.index.chunking import select_method
from student.index.files import list_file


class IndexCli(BaseCli):
    """Expose corpus indexing through Python Fire."""

    def index(
        self,
        source_root: str = "data/raw",
        output_dir: str = "data/processed",
        max_chunk_size: int = 2000,
    ) -> dict[str, str | int]:
        """Chunk a corpus and persist its BM25 index.

        Args:
            source_root: Directory containing the corpus to index.
            output_dir: Directory in which index artifacts are saved.
            max_chunk_size: Maximum number of characters in each chunk.

        Returns:
            A summary of the completed indexing operation.
        """
        try:
            chunk_size = self._positive_int(
                max_chunk_size,
                "max_chunk_size",
                2000,
            )
            source_path = self._path(source_root, "source_root")
            output_path = self._path(output_dir, "output_dir")
            if output_path.exists() and not output_path.is_dir():
                raise ValueError(
                    "output_dir exists but is not a directory: "
                    f"{output_path}"
                )

            file_list = list_file(source_path)
            if not file_list:
                raise ValueError(f"no text files found in: {source_path}")
            chunk_list = select_method(file_list, chunk_size, output_path)
            gen_bm25(chunk_list, output_path)
        except Exception as error:
            return self._error("index", str(error))

        return {
            "command": "index",
            "status": "ok",
            "source_root": source_path.as_posix(),
            "output_dir": output_path.as_posix(),
            "max_chunk_size": chunk_size,
            "files": len(file_list),
            "chunks": len(chunk_list),
        }
