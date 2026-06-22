import fire as fire_lib

from student.app.evaluate import EvaluateCli
from student.app.index import IndexCli
from student.app.search import SearchCli


class StudentCli(IndexCli, SearchCli, EvaluateCli):
    pass


def main() -> None:
    fire_lib.Fire(StudentCli)


if __name__ == "__main__":
    main()
