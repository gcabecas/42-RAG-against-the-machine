import json
from pathlib import Path

from tqdm import tqdm

from student.answer.context import make_contexts
from student.answer.generator import load_answer_model
from student.answer.process import process as answer_process
from student.app.base import BaseCli
from student.common.models import (
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from student.search.process import load_chunks, load_search_index
from student.search.process import process as search_process


class AnswerCli(BaseCli):
    DEFAULT_ANSWER_SAVE_DIRECTORY = "data/output/search_results_and_answer"

    def answer(
        self,
        query: str,
        k: int = 3,
        processed_dir: str = BaseCli.DEFAULT_OUTPUT_DIR,
        max_new_tokens: int = 128,
    ) -> dict[str, object]:
        if not query.strip():
            return self._error("answer", "query must not be empty")
        try:
            top_k = self._parse_k(k)
        except ValueError as error:
            return self._error("answer", str(error))

        try:
            processed_path = Path(processed_dir)
            search_index = load_search_index(processed_path)
            generator = load_answer_model(max_new_tokens)
            search_result = search_process(query, top_k, search_index)
            answer = answer_process(
                search_result,
                generator,
                make_contexts(search_index.chunks),
            )
        except Exception as error:
            return self._error("answer", str(error))

        return {
            "command": "answer",
            "status": "ok",
            "processed_dir": processed_path.as_posix(),
            "result": answer.model_dump(),
        }

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = DEFAULT_ANSWER_SAVE_DIRECTORY,
        processed_dir: str = BaseCli.DEFAULT_OUTPUT_DIR,
        max_new_tokens: int = 128,
    ) -> dict[str, object]:
        search_results_path = Path(student_search_results_path)
        if not search_results_path.is_file():
            return self._error(
                "answer_dataset",
                (
                    "student_search_results_path does not exist or is not "
                    f"a file: {search_results_path}"
                ),
            )

        try:
            student_results = StudentSearchResults.model_validate_json(
                search_results_path.read_text()
            )
            context_by_source = make_contexts(
                load_chunks(Path(processed_dir))
            )
            generator = load_answer_model(max_new_tokens)
            answers = [
                answer_process(
                    search_result,
                    generator,
                    context_by_source,
                )
                for search_result in tqdm(
                    student_results.search_results,
                    desc="Answering questions",
                    unit="question",
                )
            ]
            output = StudentSearchResultsAndAnswer(
                search_results=answers,
                k=student_results.k,
            ).model_dump()
            save_path = Path(save_directory) / search_results_path.name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(
                json.dumps(output, ensure_ascii=False, indent=4)
            )
        except Exception as error:
            return self._error("answer_dataset", str(error))

        return {
            "command": "answer_dataset",
            "status": "ok",
            "student_search_results_path": search_results_path.as_posix(),
            "save_path": save_path.as_posix(),
            "processed_dir": Path(processed_dir).as_posix(),
            "k": student_results.k,
            "questions": len(student_results.search_results),
        }
