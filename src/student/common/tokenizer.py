import re


def tokenize(text: str) -> list[str]:
    """Tokenize natural language and code identifiers for BM25.

    Args:
        text: Natural-language or source-code text to tokenize.

    Returns:
        Normalized lexical tokens.
    """
    tokens: list[str] = []
    for token in re.findall(r"[A-Za-z0-9_]+", text):
        candidates = [token]
        for snake_part in token.split("_"):
            candidates.append(snake_part)
            candidates.extend(
                re.split(
                    (
                        r"(?<=[a-z0-9])(?=[A-Z])"
                        r"|(?<=[A-Z])(?=[A-Z][a-z])"
                    ),
                    snake_part,
                )
            )

        seen: set[str] = set()
        for candidate in candidates:
            normalized = candidate.strip("_").casefold()
            if len(normalized) > 1 and normalized not in seen:
                seen.add(normalized)
                tokens.append(normalized)
    return tokens
