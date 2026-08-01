import sys
from pathlib import Path
from typing import NoReturn


class BaseCli:
    """Provide validation and error handling shared by CLI commands."""

    def _error(self, command: str, message: str) -> NoReturn:
        """Print a controlled CLI error and stop without a traceback.

        Args:
            command: Name of the command that failed.
            message: Error details shown to the user.

        Raises:
            SystemExit: Always, with a nonzero exit status.
        """
        print(f"{command}: error: {message}", file=sys.stderr)
        raise SystemExit(1)

    def _positive_int(
        self,
        value: object,
        name: str,
        maximum: int | None = None,
    ) -> int:
        """Validate a positive integer CLI argument.

        Args:
            value: Value received from Python Fire.
            name: Argument name used in validation errors.
            maximum: Optional inclusive upper bound.

        Returns:
            The validated integer.

        Raises:
            ValueError: If the value is not within the accepted range.
        """
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
        if value < 1:
            raise ValueError(f"{name} must be at least 1")
        if maximum is not None and value > maximum:
            raise ValueError(f"{name} must be at most {maximum}")
        return value

    def _query(self, value: object) -> str:
        """Validate and normalize a searchable query.

        Args:
            value: Query value received from Python Fire.

        Returns:
            The stripped, nonempty query.

        Raises:
            ValueError: If the query is not a nonempty string.
        """
        if not isinstance(value, str):
            raise ValueError("query must be a string")
        query = value.strip()
        if not query:
            raise ValueError("query must not be empty")
        return query

    def _path(self, value: object, name: str) -> Path:
        """Validate a path received through Python Fire.

        Args:
            value: Path value received from Python Fire.
            name: Argument name used in validation errors.

        Returns:
            The validated path.

        Raises:
            ValueError: If the value is not a nonempty string.
        """
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty path")
        return Path(value)
