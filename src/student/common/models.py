import uuid

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """Describe one exact character range in a corpus file.

    Args:
        file_path: Corpus-relative path to the source file.
        first_character_index: Start offset of the source range.
        last_character_index: End offset of the source range.
    """

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Represent a question that has no reference answer.

    Args:
        question_id: Stable identifier for the question.
        question: Natural-language question text.
    """

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Extend a question with its reference sources and answer.

    Args:
        question_id: Stable identifier for the question.
        question: Natural-language question text.
        sources: Reference source locations that support the answer.
        answer: Reference answer text.
    """

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Contain the questions exchanged by dataset commands.

    Args:
        rag_questions: Answered or unanswered questions in the dataset.
    """

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Contain the ranked source locations for one question.

    Args:
        question_id: Identifier copied from the input question.
        question: Original question text.
        retrieved_sources: Source locations ranked by retrieval.
    """

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Extend retrieval results with a generated answer.

    Args:
        question_id: Identifier copied from the input question.
        question: Original question text.
        retrieved_sources: Source locations ranked by retrieval.
        answer: Generated answer grounded in the retrieved sources.
    """

    answer: str


class StudentSearchResults(BaseModel):
    """Contain batch retrieval results and the requested k.

    Args:
        search_results: Retrieval results for all input questions.
        k: Number of sources requested for each question.
    """

    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Contain batch answers and the requested retrieval k.

    Args:
        search_results: Generated answers for all input questions.
        k: Number of sources requested for each question.
    """

    search_results: list[MinimalAnswer]
    k: int
