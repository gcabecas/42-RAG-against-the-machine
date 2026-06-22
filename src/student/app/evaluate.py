from pathlib import Path

from student.app.base import BaseCli
from student.evaluate.process import process


class EvaluateCli(BaseCli):
    def evaluate(
        self,
        student_results_path: str,
        dataset_path: str,
        k: int = 10,
    ) -> dict[str, object]:
        try:
            top_k = self._parse_k(k)
        except ValueError as error:
            return self._error("evaluate", str(error))

        student_results = Path(student_results_path)
        dataset = Path(dataset_path)
        if not student_results.is_file():
            return self._error(
                "evaluate",
                (
                    "student_results_path does not exist or is not a file: "
                    f"{student_results}"
                ),
            )
        if not dataset.is_file():
            return self._error(
                "evaluate",
                f"dataset_path does not exist or is not a file: {dataset}",
            )

        try:
            result = process(student_results, dataset, top_k)
        except OSError as error:
            return self._error("evaluate", str(error))
        except ValueError as error:
            return self._error("evaluate", str(error))

        return {
            "command": "evaluate",
            "status": "ok",
            "result": result,
        }
