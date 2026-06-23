from student.answer.context import SourceContext


SYSTEM_PROMPT = (
    "You answer questions about the vLLM codebase. "
    "Use only the provided sources. "
    "If the sources do not contain the answer, say so. "
    "Do not use Markdown links or bold text. "
    "Include URLs only when the question asks for a URL, endpoint, or link."
)


def build_sources(contexts: list[SourceContext]) -> str:
    blocks = []
    for index, context in enumerate(contexts, start=1):
        source = context.source
        blocks.append(
            f"[source {index}] {source.file_path}:"
            f"{source.first_character_index}-"
            f"{source.last_character_index}\n"
            f"{context.text.strip()}"
        )
    return "\n\n".join(blocks)


def build_user_prompt(
    question: str,
    contexts: list[SourceContext],
    max_new_tokens: int,
) -> str:
    return (
        f"Sources:\n{build_sources(contexts)}\n\n"
        f"Question:\n{question}\n\n"
        "Rules:\n"
        f"- Keep the answer under {max_new_tokens} tokens.\n"
        "- Start with the direct answer, not with background.\n"
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
        "- Prefer earlier sources when multiple sources look relevant.\n"
        "- End with exactly one citation in the format [source N]."
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
