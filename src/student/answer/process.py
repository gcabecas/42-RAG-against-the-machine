from student.answer.context import ContextBySource, load_context
from student.answer.generator import AnswerGenerator
from student.common.models import MinimalAnswer, MinimalSearchResults

MAX_ANSWER_CONTEXTS = 10


def process(
    search_result: MinimalSearchResults,
    generator: AnswerGenerator,
    context_by_source: ContextBySource,
) -> MinimalAnswer:
    selected_sources = search_result.retrieved_sources[:MAX_ANSWER_CONTEXTS]
    contexts = [
        load_context(source, context_by_source)
        for source in selected_sources
    ]
    return MinimalAnswer(
        question_id=search_result.question_id,
        question_str=search_result.question_str,
        retrieved_sources=selected_sources,
        answer=generator.generate(search_result.question_str, contexts),
    )
