from functools import lru_cache
import re


def split_camel_case(token: str) -> list[str]:
    pattern = r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"
    return re.split(pattern, token)


@lru_cache(maxsize=100000)
def token_parts(token: str) -> tuple[str, ...]:
    parts: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        value = value.strip("_").casefold()
        if len(value) > 1 and value not in seen:
            seen.add(value)
            parts.append(value)

    add(token)
    for snake_part in token.split("_"):
        add(snake_part)
        for camel_part in split_camel_case(snake_part):
            add(camel_part)
    return tuple(parts)


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for token in re.findall(r"[A-Za-z0-9_]+", text):
        tokens.extend(token_parts(token))
    return tokens
