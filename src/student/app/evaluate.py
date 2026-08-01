from student.app.base import BaseCli
from student.evaluate.process import process


class EvaluateCli(BaseCli):
    """Expose local recall evaluation through Python Fire."""

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
        k: int = 10,
    ) -> dict[str, object]:
        """Compare saved retrieval results with an answered dataset.

        Args:
            student_search_results_path: Path to saved retrieval results.
            dataset_path: Path to the answered reference dataset.
            k: Number of retrieved sources considered per question.

        Returns:
            A command result containing recall metrics.
        """
        try:
            top_k = self._positive_int(k, "k")
            student_results = self._path(
                student_search_results_path,
                "student_search_results_path",
            )
            dataset = self._path(dataset_path, "dataset_path")
            if not student_results.is_file():
                raise ValueError(
                    "student_search_results_path does not exist or is not "
                    f"a file: {student_results}"
                )
            if not dataset.is_file():
                raise ValueError(
                    "dataset_path does not exist or is not a file: "
                    f"{dataset}"
                )
            result = process(student_results, dataset, top_k)
        except Exception as error:
            return self._error("evaluate", str(error))

        return {
            "command": "evaluate",
            "status": "ok",
            "result": result,
        }
