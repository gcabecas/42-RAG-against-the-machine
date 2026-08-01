from pathlib import Path

from student.common.models import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)


def source_iou(expected: MinimalSource, predicted: MinimalSource) -> float:
    """Compute source overlap when both paths are identical.

    Args:
        expected: Reference source range.
        predicted: Retrieved source range.

    Returns:
        Intersection over union, or zero for different files.
    """
    if expected.file_path != predicted.file_path:
        return 0.0
    intersection = max(
        0,
        min(expected.last_character_index, predicted.last_character_index)
        - max(expected.first_character_index, predicted.first_character_index),
    )
    expected_len = (
        expected.last_character_index - expected.first_character_index
    )
    predicted_len = (
        predicted.last_character_index - predicted.first_character_index
    )
    union = expected_len + predicted_len - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def question_recall(
    expected_sources: list[MinimalSource],
    predicted_sources: list[MinimalSource],
) -> float:
    """Compute recall for one question at a chosen result limit.

    Args:
        expected_sources: Reference source ranges.
        predicted_sources: Retrieved source ranges to evaluate.

    Returns:
        Fraction of reference sources matched by a prediction.
    """
    if not expected_sources:
        return 1.0
    found = sum(
        1
        for expected in expected_sources
        if any(
            source_iou(expected, predicted) >= 0.05
            for predicted in predicted_sources
        )
    )
    return found / len(expected_sources)


def process(
    student_search_results_path: Path,
    dataset_path: Path,
    k: int,
) -> dict[str, object]:
    """Compute local recall metrics for saved retrieval results.

    Args:
        student_search_results_path: Path to student retrieval results.
        dataset_path: Path to the answered reference dataset.
        k: Maximum number of predictions considered per question.

    Returns:
        Evaluation metadata and recall metrics.

    Raises:
        ValueError: If the reference dataset contains unanswered questions.
    """
    student_results = StudentSearchResults.model_validate_json(
        student_search_results_path.read_text(encoding="utf-8")
    )
    dataset = RagDataset.model_validate_json(
        dataset_path.read_text(encoding="utf-8")
    )
    predicted_by_id = {
        result.question_id: result.retrieved_sources
        for result in student_results.search_results
    }

    recall_limits = [limit for limit in (1, 3, 5, 10) if limit <= k]
    if k not in recall_limits:
        recall_limits.append(k)
    recalls: dict[int, list[float]] = {
        limit: []
        for limit in recall_limits
    }

    for question in dataset.rag_questions:
        if not isinstance(question, AnsweredQuestion):
            raise ValueError(f"{dataset_path}: expected answered questions")
        predicted_sources = predicted_by_id.get(question.question_id, [])
        for limit in recall_limits:
            recalls[limit].append(
                question_recall(question.sources, predicted_sources[:limit])
            )

    metrics = {
        f"recall@{limit}": (
            sum(values) / len(values)
            if values
            else 0.0
        )
        for limit, values in recalls.items()
    }
    return {
        "student_search_results_path": (
            student_search_results_path.as_posix()
        ),
        "dataset_path": dataset_path.as_posix(),
        "k": k,
        "questions": len(dataset.rag_questions),
        "student_results": len(student_results.search_results),
        **metrics,
    }
