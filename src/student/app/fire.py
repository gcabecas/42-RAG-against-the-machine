import fire as fire_lib

from student.app.answer import AnswerCli
from student.app.evaluate import EvaluateCli
from student.app.index import IndexCli
from student.app.search import SearchCli


class StudentCli(IndexCli, SearchCli, EvaluateCli, AnswerCli):
    """Combine every mandatory command in one Python Fire CLI."""


def main() -> None:
    """Run the project command-line interface."""
    fire_lib.Fire(StudentCli())
