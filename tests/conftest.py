import os

import pytest


@pytest.fixture(autouse=True)
def isolated_git_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep inherited Git state out of the repositories built by tests.

    A `pre-commit` hook runs with `GIT_INDEX_FILE` and friends exported, which
    the `git` subprocesses started by tests would otherwise pick up.
    """
    for name in list(os.environ):
        if name.startswith("GIT_"):
            monkeypatch.delenv(name, raising=False)
