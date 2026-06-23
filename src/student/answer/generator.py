import re
from dataclasses import dataclass
from typing import Any, cast

from student.answer.context import SourceContext
from student.answer.prompt import build_messages
from student.common.tokenizer import tokenize

IGNORED_SOURCE_TOKENS = {
    "the",
    "and",
    "for",
    "with",
    "what",
    "how",
    "which",
    "does",
    "are",
    "you",
    "can",
    "used",
    "using",
    "that",
    "this",
    "from",
    "when",
    "where",
    "vllm",
}
SOURCE_CITATION_PATTERN = re.compile(r"\[source\s+([0-9]+)\]", re.IGNORECASE)


@dataclass
class AnswerGenerator:
    tokenizer: Any
    model: Any
    device: str
    max_new_tokens: int

    def generate(self, question: str, contexts: list[SourceContext]) -> str:
        messages = build_messages(question, contexts, self.max_new_tokens)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        answer = str(self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ))
        answer = clean_answer(answer)
        return ensure_single_source_citation(question, answer, contexts)


def ensure_single_source_citation(
    question: str,
    answer: str,
    contexts: list[SourceContext],
) -> str:
    source_index = first_valid_source_citation(answer, len(contexts))
    answer_without_citations = SOURCE_CITATION_PATTERN.sub("", answer).strip()

    if not contexts:
        return answer_without_citations

    if source_index is None:
        source_index = best_source_index(
            question,
            answer_without_citations,
            contexts,
        )

    if not answer_without_citations:
        return f"[source {source_index}]"
    return f"{answer_without_citations} [source {source_index}]"


def first_valid_source_citation(answer: str, context_count: int) -> int | None:
    for match in SOURCE_CITATION_PATTERN.finditer(answer):
        source_index = int(match.group(1))
        if 1 <= source_index <= context_count:
            return source_index
    return None


def best_source_index(
    question: str,
    answer: str,
    contexts: list[SourceContext],
) -> int:
    question_tokens = important_tokens(question)
    answer_tokens = important_tokens(answer)
    best_index = 1
    best_score = -1

    for index, context in enumerate(contexts, start=1):
        context_tokens = set(tokenize(context.text))
        score = (
            len(answer_tokens & context_tokens) * 2
            + len(question_tokens & context_tokens)
        )
        if score > best_score:
            best_score = score
            best_index = index

    return best_index


def important_tokens(text: str) -> set[str]:
    return {
        token
        for token in tokenize(text)
        if len(token) > 2 and token not in IGNORED_SOURCE_TOKENS
    }


def clean_answer(answer: str) -> str:
    if "</think>" in answer:
        answer = answer.split("</think>", 1)[1]
    answer = answer.replace("**", "")
    answer = re.sub(
        r"\[([^\]]+)\]\(\s*\[source ([0-9]+)\]\s*\)",
        r"\1 [source \2]",
        answer,
    )
    answer = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1 \2", answer)
    answer = re.sub(r"\[([^\]]+)\]\(#[^)]+\)", r"\1", answer)
    answer = re.sub(r"\[([^\]]+)\]\[[^\]]+\]", r"\1", answer)
    answer = re.sub(r"\s*\(?data/raw/[^)\s]+(?:\))?", "", answer)
    answer = re.sub(r"\s+and\s+mention this", "", answer)
    answer = answer.replace("```console", "").replace("```bash", "")
    answer = answer.replace("```python", "").replace("```", "")
    answer = re.sub(
        r"\bvLLM(?=(metric|model|server|service|framework|deployment)\b)",
        "vLLM ",
        answer,
    )
    answer = re.sub(
        r"\b(framework|command|script|metric|method|class|endpoint|"
        r"parameter|version)(?=(that|used|is|can|should)\b)",
        r"\1 ",
        answer,
    )
    return answer.strip()


def load_answer_model(
    max_new_tokens: int,
) -> AnswerGenerator:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = cast(Any, AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B"))
    model = cast(Any, AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        dtype=dtype,
    ))
    model.to(device)
    model.eval()
    return AnswerGenerator(
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_new_tokens=max_new_tokens,
    )
