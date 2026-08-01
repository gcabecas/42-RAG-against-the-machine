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
    """Expose single and batch grounded generation through Python Fire."""

    def answer(
        self,
        query: str,
        k: int = 3,
        processed_dir: str = "data/processed",
        max_new_tokens: int = 128,
    ) -> dict[str, object]:
        """Retrieve context and answer one query with Qwen.

        Args:
            query: Question to answer.
            k: Maximum number of sources to retrieve.
            processed_dir: Directory containing persisted index artifacts.
            max_new_tokens: Maximum number of tokens in the generated answer.

        Returns:
            A command result containing the grounded answer.
        """
        try:
            clean_query = self._query(query)
            top_k = self._positive_int(k, "k")
            token_limit = self._positive_int(
                max_new_tokens,
                "max_new_tokens",
            )
            processed_path = self._path(processed_dir, "processed_dir")
            search_index = load_search_index(processed_path)
            search_result = search_process(
                clean_query,
                top_k,
                search_index,
            )
            generator = load_answer_model(token_limit)
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
        save_directory: str,
        processed_dir: str = "data/processed",
        max_new_tokens: int = 128,
    ) -> dict[str, object]:
        """Generate answers for saved retrieval results.

        Args:
            student_search_results_path: Path to saved retrieval results.
            save_directory: Directory in which answers are written.
            processed_dir: Directory containing persisted index artifacts.
            max_new_tokens: Maximum number of tokens generated per answer.

        Returns:
            A summary containing the output path and question count.
        """
        try:
            token_limit = self._positive_int(
                max_new_tokens,
                "max_new_tokens",
            )
            search_results_path = self._path(
                student_search_results_path,
                "student_search_results_path",
            )
            save_dir = self._path(save_directory, "save_directory")
            processed_path = self._path(processed_dir, "processed_dir")
            if not search_results_path.is_file():
                raise ValueError(
                    "student_search_results_path does not exist or is not "
                    f"a file: {search_results_path}"
                )
            student_results = StudentSearchResults.model_validate_json(
                search_results_path.read_text(encoding="utf-8")
            )
            self._positive_int(student_results.k, "k")
            context_by_source = make_contexts(
                load_chunks(processed_path)
            )
            generator = load_answer_model(token_limit)
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
            )
            save_path = save_dir / search_results_path.name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(
                output.model_dump_json(indent=4),
                encoding="utf-8",
            )
        except Exception as error:
            return self._error("answer_dataset", str(error))

        return {
            "command": "answer_dataset",
            "status": "ok",
            "student_search_results_path": search_results_path.as_posix(),
            "save_path": save_path.as_posix(),
            "processed_dir": processed_path.as_posix(),
            "k": student_results.k,
            "questions": len(student_results.search_results),
        }
