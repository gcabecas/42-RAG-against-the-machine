import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from student.app.base import BaseCli
from student.common.models import StudentSearchResults
from student.search.process import load_search_index, process


class SearchCli(BaseCli):
    DEFAULT_SEARCH_SAVE_DIRECTORY = "data/output/search_results"

    def parse_k(self, command: str, k: int | str) -> int:
        try:
            top_k = int(k)
        except (TypeError, ValueError):
            return self._error(command, "k must be an integer")
        if top_k < 1:
            return self._error(command, "k must be at least 1")
        return top_k

    def search(
        self,
        query: str,
        k: int | str = 10,
        processed_dir: str = BaseCli.DEFAULT_OUTPUT_DIR,
    ) -> dict[str, object]:
        if not query.strip():
            return self._error("search", "query must not be empty")
        top_k = self.parse_k("search", k)

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

    def dataset_questions(
        self,
        data: Any,
        dataset_path: Path,
    ) -> list[dict[str, str]]:
        if not isinstance(data, dict):
            raise ValueError(f"{dataset_path}: expected a JSON object")
        raw_questions = data.get("rag_questions")
        if not isinstance(raw_questions, list):
            raise ValueError(f"{dataset_path}: missing rag_questions list")

        questions: list[dict[str, str]] = []
        for index, item in enumerate(raw_questions):
            if not isinstance(item, dict):
                raise ValueError(
                    f"{dataset_path}: rag_questions[{index}] must be an object"
                )
            question_id = item.get("question_id")
            question = item.get("question")
            if not isinstance(question_id, str):
                raise ValueError(
                    f"{dataset_path}: question_id must be a string"
                )
            if not isinstance(question, str):
                raise ValueError(f"{dataset_path}: question must be a string")
            questions.append({
                "question_id": question_id,
                "question": question,
            })
        return questions

    def search_dataset(
        self,
        dataset_path: str,
        k: int | str = 10,
        processed_dir: str = BaseCli.DEFAULT_OUTPUT_DIR,
        save_directory: str = DEFAULT_SEARCH_SAVE_DIRECTORY,
    ) -> dict[str, object]:
        top_k = self.parse_k("search_dataset", k)
        dataset = Path(dataset_path)
        if not dataset.is_file():
            return self._error(
                "search_dataset",
                f"dataset_path does not exist or is not a file: {dataset}",
            )

        try:
            data = json.loads(dataset.read_text(encoding="utf-8"))
            questions = self.dataset_questions(data, dataset)
            search_index = load_search_index(Path(processed_dir))
            search_results = []
            for question in tqdm(
                questions,
                desc="Searching questions",
                unit="question",
            ):
                search_results.append(
                    process(
                        question["question"],
                        top_k,
                        search_index,
                        question["question_id"],
                    )
                )
            output = StudentSearchResults(
                search_results=search_results,
                k=top_k,
            ).model_dump()
            save_path = Path(save_directory) / dataset.name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(
                json.dumps(output, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as error:
            return self._error("search_dataset", str(error))
        except json.JSONDecodeError as error:
            return self._error("search_dataset", f"invalid JSON: {error}")
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
            "questions": len(questions),
        }
