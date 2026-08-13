import os
import shutil
import subprocess
from pathlib import Path

import pytest

git_path = shutil.which("git")
if git_path is None:
    raise RuntimeError("git is required for these tests")
GIT: str = git_path


@pytest.fixture(autouse=True)
def isolated_git_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep inherited Git state out of the repositories built by tests.

    A `pre-commit` hook runs with `GIT_INDEX_FILE` and friends exported, which
    the `git` subprocesses started by tests would otherwise pick up.
    """
    for name in list(os.environ):
        if name.startswith("GIT_"):
            monkeypatch.delenv(name, raising=False)


def run_git(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(  # noqa: S603
        [GIT, "-C", str(repository), *arguments],
        text=True,
    ).strip()


def commit_file(repository: Path, number: int) -> str:
    _ = (repository / "value.txt").write_text(str(number))
    _ = subprocess.run(  # noqa: S603
        [GIT, "-C", str(repository), "add", "value.txt"],
        check=True,
    )
    _ = subprocess.run(  # noqa: S603
        [
            GIT,
            "-C",
            repository,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            str(number),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return run_git(repository, "rev-parse", "HEAD")


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    checkout = tmp_path / "repository"
    _ = subprocess.run(  # noqa: S603
        [GIT, "init", "-q", str(checkout)], check=True
    )
    return checkout
