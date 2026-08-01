from student.answer.context import load_context
from student.answer.generator import AnswerGenerator
from student.common.models import MinimalAnswer, MinimalSearchResults


def process(
    search_result: MinimalSearchResults,
    generator: AnswerGenerator,
    context_by_source: dict[tuple[str, int, int], str],
) -> MinimalAnswer:
    """Generate an answer while preserving every retrieved source.

    Args:
        search_result: Retrieval result to answer.
        generator: Loaded model wrapper used for generation.
        context_by_source: Indexed text keyed by source coordinates.

    Returns:
        The retrieval result extended with a generated answer.
    """
    contexts = [
        f"{source.file_path}\n{load_context(source, context_by_source)}"
        for source in search_result.retrieved_sources[:10]
    ]
    return MinimalAnswer(
        question_id=search_result.question_id,
        question=search_result.question,
        retrieved_sources=search_result.retrieved_sources,
        answer=generator.generate(search_result.question, contexts),
    )
