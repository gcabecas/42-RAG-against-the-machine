import fire as fire_lib

from student.app.answer import AnswerCli
from student.app.evaluate import EvaluateCli
from student.app.index import IndexCli
from student.app.search import SearchCli


class StudentCli(IndexCli, SearchCli, EvaluateCli, AnswerCli):
    pass


def main() -> None:
    fire_lib.Fire(StudentCli)


if __name__ == "__main__":
    main()
