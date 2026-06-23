from pathlib import Path

from student.common.models import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)

MINIMAL_IOU_THRESHOLD = 0.05


def source_iou(expected: MinimalSource, predicted: MinimalSource) -> float:
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
    if not expected_sources:
        return 1.0
    found = sum(
        1
        for expected in expected_sources
        if any(
            source_iou(expected, predicted) >= MINIMAL_IOU_THRESHOLD
            for predicted in predicted_sources
        )
    )
    return found / len(expected_sources)


def process(
    student_results_path: Path,
    dataset_path: Path,
    k: int,
) -> dict[str, object]:
    student_results = StudentSearchResults.model_validate_json(
        student_results_path.read_text()
    )
    dataset = RagDataset.model_validate_json(
        dataset_path.read_text()
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
        "student_results_path": student_results_path.as_posix(),
        "dataset_path": dataset_path.as_posix(),
        "k": k,
        "questions": len(dataset.rag_questions),
        "student_results": len(student_results.search_results),
        **metrics,
    }
