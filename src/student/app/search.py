from tqdm import tqdm

from student.app.base import BaseCli
from student.common.models import RagDataset, StudentSearchResults
from student.search.process import load_search_index, process


class SearchCli(BaseCli):
    """Expose single and batch BM25 retrieval through Python Fire."""

    def search(
        self,
        query: str,
        k: int = 10,
        processed_dir: str = "data/processed",
    ) -> dict[str, object]:
        """Return the top-k sources for one query.

        Args:
            query: Search query.
            k: Maximum number of sources to retrieve.
            processed_dir: Directory containing persisted index artifacts.

        Returns:
            A command result containing the ranked source locations.
        """
        try:
            clean_query = self._query(query)
            top_k = self._positive_int(k, "k")
            processed_path = self._path(processed_dir, "processed_dir")
            search_index = load_search_index(processed_path)
            result = process(clean_query, top_k, search_index)
        except Exception as error:
            return self._error("search", str(error))
        return {
            "command": "search",
            "status": "ok",
            "result": result.model_dump(),
        }

    def search_dataset(
        self,
        dataset_path: str,
        save_directory: str,
        k: int = 10,
        processed_dir: str = "data/processed",
    ) -> dict[str, object]:
        """Search every question in a dataset and save validated JSON.

        Args:
            dataset_path: Path to a RAG question dataset.
            save_directory: Directory in which results are written.
            k: Maximum number of sources retrieved per question.
            processed_dir: Directory containing persisted index artifacts.

        Returns:
            A summary containing the output path and question count.
        """
        try:
            top_k = self._positive_int(k, "k")
            dataset = self._path(dataset_path, "dataset_path")
            save_dir = self._path(save_directory, "save_directory")
            processed_path = self._path(processed_dir, "processed_dir")
            if not dataset.is_file():
                raise ValueError(
                    "dataset_path does not exist or is not a file: "
                    f"{dataset}"
                )
            rag_dataset = RagDataset.model_validate_json(
                dataset.read_text(encoding="utf-8")
            )
            search_index = load_search_index(processed_path)
            results = StudentSearchResults(
                search_results=[
                    process(
                        question.question,
                        top_k,
                        search_index,
                        question.question_id,
                    )
                    for question in tqdm(
                        rag_dataset.rag_questions,
                        desc="Searching questions",
                        unit="question",
                    )
                ],
                k=top_k,
            )
            save_path = save_dir / dataset.name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(
                results.model_dump_json(indent=4),
                encoding="utf-8",
            )
        except Exception as error:
            return self._error("search_dataset", str(error))

        return {
            "command": "search_dataset",
            "status": "ok",
            "dataset_path": dataset.as_posix(),
            "save_path": save_path.as_posix(),
            "k": top_k,
            "questions": len(rag_dataset.rag_questions),
        }
