import sys
from typing import NoReturn


class BaseCli:
    DEFAULT_SOURCE_ROOT = "data/raw/vllm-0.10.1"
    DEFAULT_OUTPUT_DIR = "data/processed"
    DEFAULT_MAX_CHUNK_SIZE = 2000

    def _error(self, command: str, message: str) -> NoReturn:
        print(f"{command}: error: {message}", file=sys.stderr)
        raise SystemExit(1)

    def _parse_k(self, k: int) -> int:
        if not isinstance(k, int) or isinstance(k, bool):
            raise ValueError("k must be an integer")
        if k < 1:
            raise ValueError("k must be at least 1")
        return k
