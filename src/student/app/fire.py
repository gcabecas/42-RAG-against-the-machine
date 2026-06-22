import fire as fire_lib

from student.app.index import IndexCli
from student.app.search import SearchCli


class StudentCli(IndexCli, SearchCli):
    pass


def main() -> None:
    fire_lib.Fire(StudentCli)


if __name__ == "__main__":
    main()
