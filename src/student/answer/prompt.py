from student.answer.context import SourceContext
from student.common.tokenizer import tokenize


SYSTEM_PROMPT = (
    "You answer questions about the vLLM codebase. "
    "Use only the provided sources. "
    "If the sources do not contain the answer, say so. "
    "Do not use Markdown links or bold text. "
    "Do not mention source file paths in the answer. "
    "Cite source labels exactly as [source N]. "
    "Include URLs only when the question asks for a URL, endpoint, or link."
)


def question_tokens(question: str) -> set[str]:
    ignored = {
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
        "vllm",
    }
    return {
        token
        for token in tokenize(question)
        if len(token) > 2 and token not in ignored
    }


def code_block_bounds(lines: list[str], line_index: int) -> tuple[int, int]:
    start = line_index
    while start > 0 and not lines[start].lstrip().startswith("```"):
        start -= 1

    end = line_index
    while end + 1 < len(lines) and not lines[end].lstrip().startswith("```"):
        end += 1

    if (
        lines[start].lstrip().startswith("```")
        and lines[end].lstrip().startswith("```")
    ):
        return start, end
    return max(0, line_index - 1), min(len(lines) - 1, line_index + 2)


def table_bounds(lines: list[str], line_index: int) -> tuple[int, int]:
    start = line_index
    while start > 0 and "|" in lines[start - 1]:
        start -= 1

    end = line_index
    while end + 1 < len(lines) and "|" in lines[end + 1]:
        end += 1

    return start, end


def add_window(
    selected: set[int],
    lines: list[str],
    line_index: int,
) -> None:
    line = lines[line_index]
    if "|" in line:
        start, end = table_bounds(lines, line_index)
    elif line.lstrip().startswith("```"):
        start, end = code_block_bounds(lines, line_index)
    else:
        start = max(0, line_index - 2)
        end = min(len(lines) - 1, line_index + 3)

    selected.update(range(start, end + 1))


def compact_text(question: str, text: str, max_chars: int = 1200) -> str:
    clean_text = text.strip()
    if len(clean_text) <= max_chars:
        return clean_text

    tokens = question_tokens(question)
    if not tokens:
        return clean_text

    lines = text.splitlines()
    selected: set[int] = set()
    for line_index, line in enumerate(lines):
        line_tokens = set(tokenize(line))
        if tokens & line_tokens:
            add_window(selected, lines, line_index)

    if not selected:
        return clean_text

    output = []
    previous = -2
    for line_index in sorted(selected):
        if line_index > previous + 1:
            output.append("[...]")
        output.append(lines[line_index])
        previous = line_index
        if len("\n".join(output)) >= max_chars:
            break

    return "\n".join(output).strip()


def build_sources(question: str, contexts: list[SourceContext]) -> str:
    blocks = []
    for index, context in enumerate(contexts, start=1):
        source = context.source
        blocks.append(
            f"[source {index}] {source.file_path}:"
            f"{source.first_character_index}-"
            f"{source.last_character_index}\n"
            f"{compact_text(question, context.text)}"
        )
    return "\n\n".join(blocks)


def build_user_prompt(
    question: str,
    contexts: list[SourceContext],
    max_new_tokens: int,
) -> str:
    return (
        f"Sources:\n{build_sources(question, contexts)}\n\n"
        f"Question:\n{question}\n\n"
        "Rules:\n"
        f"- Keep the answer under {max_new_tokens} tokens.\n"
        "- Start with the direct answer, not with background.\n"
        "- Do not mention source file paths.\n"
        "- End with exactly one citation in the format [source N].\n"
        "- Do not use fenced code blocks; write commands inline.\n"
        "- Copy exact names, commands, flags, values, endpoints, classes, "
        "methods, parameters, versions, and environment variables.\n"
        "- If the question asks for several items, include all relevant "
        "items found in the sources.\n"
        "- If the question asks how to run, install, generate, configure, "
        "enable, or fix something, start with the exact command or setting "
        "when it appears in the sources.\n"
        "- If the question asks yes/no, start with Yes or No.\n"
        "- Preserve negatives and limits such as not, none, only, partial, "
        "unsupported, deprecated, and default.\n"
        "- For errors, causes, requirements, and support status, give the "
        "exact error, cause, requirement, or status from the sources.\n"
        "- Prefer earlier sources when multiple sources look relevant."
    )


def build_messages(
    question: str,
    contexts: list[SourceContext],
    max_new_tokens: int,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(
                question,
                contexts,
                max_new_tokens,
            ),
        },
    ]
