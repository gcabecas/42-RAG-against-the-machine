import json
from pathlib import Path

from tqdm import tqdm

from student.app.base import BaseCli
from student.common.models import RagDataset, StudentSearchResults
from student.search.process import load_search_index, process


class SearchCli(BaseCli):
    DEFAULT_SEARCH_SAVE_DIRECTORY = "data/output/search_results"

    def search(
        self,
        query: str,
        k: int = 10,
        processed_dir: str = BaseCli.DEFAULT_OUTPUT_DIR,
    ) -> dict[str, object]:
        if not query.strip():
            return self._error("search", "query must not be empty")
        try:
            top_k = self._parse_k(k)
        except ValueError as error:
            return self._error("search", str(error))

        try:
            search_index = load_search_index(Path(processed_dir))
            result = process(query, top_k, search_index)
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
        k: int = 10,
        processed_dir: str = BaseCli.DEFAULT_OUTPUT_DIR,
        save_directory: str = DEFAULT_SEARCH_SAVE_DIRECTORY,
    ) -> dict[str, object]:
        try:
            top_k = self._parse_k(k)
        except ValueError as error:
            return self._error("search_dataset", str(error))
        dataset = Path(dataset_path)
        if not dataset.is_file():
            return self._error(
                "search_dataset",
                f"dataset_path does not exist or is not a file: {dataset}",
            )

        try:
            rag_dataset = RagDataset.model_validate_json(dataset.read_text())
            search_index = load_search_index(Path(processed_dir))
            search_results = []
            for question in tqdm(
                rag_dataset.rag_questions,
                desc="Searching questions",
                unit="question",
            ):
                search_results.append(
                    process(
                        question.question,
                        top_k,
                        search_index,
                        question.question_id,
                    )
                )
            output = StudentSearchResults(
                search_results=search_results,
                k=top_k,
            ).model_dump()
            save_path = Path(save_directory) / dataset.name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(
                json.dumps(output, ensure_ascii=False, indent=4)
            )
        except OSError as error:
            return self._error("search_dataset", str(error))
        except ValueError as error:
            return self._error("search_dataset", str(error))
        except Exception as error:
            return self._error(
                "search_dataset",
                f"unexpected error: {error}",
            )

        return {
            "command": "search_dataset",
            "status": "ok",
            "dataset_path": dataset.as_posix(),
            "save_path": save_path.as_posix(),
            "k": top_k,
            "questions": len(rag_dataset.rag_questions),
        }
