import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

from fetcher_counter import counting
from fetcher_counter.counting import GrepError, count_fetcher, count_fetchers

git_path = shutil.which("git")
if git_path is None:
    raise RuntimeError("git is required for these tests")
GIT: str = git_path


@pytest.fixture
def repository(tmp_path: Path) -> tuple[Path, str]:
    _ = subprocess.run(  # noqa: S603
        [GIT, "init", "-q", str(tmp_path)], check=True
    )
    _ = (tmp_path / "package.nix").write_text(
        "fetchurl fetchurl\nfetchFromGitHub\n"
    )
    _ = (tmp_path / "ignored.txt").write_text("fetchurl\n")
    _ = subprocess.run(  # noqa: S603
        [GIT, "-C", str(tmp_path), "add", "."], check=True
    )
    _ = subprocess.run(  # noqa: S603
        [
            GIT,
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "fixture",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    commit = subprocess.check_output(  # noqa: S603
        [GIT, "-C", str(tmp_path), "rev-parse", "HEAD"], text=True
    ).strip()
    return tmp_path, commit


@pytest.mark.asyncio
async def test_count_fetcher_counts_occurrences_not_lines(
    repository: tuple[Path, str],
) -> None:
    path, commit = repository

    assert await count_fetcher(path, commit, "fetchurl") == 2
    assert await count_fetcher(path, commit, "missingFetcher") == 0


@pytest.mark.asyncio
async def test_count_fetcher_reports_git_errors(tmp_path: Path) -> None:
    with pytest.raises(GrepError, match=r"fetchurl.*missing-commit"):
        _ = await count_fetcher(tmp_path, "missing-commit", "fetchurl")


@pytest.mark.asyncio
async def test_count_fetchers_starts_all_counts_concurrently(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    started: set[str] = set()
    all_started = asyncio.Event()

    async def fake_count(_repository: Path, _commit: str, fetcher: str) -> int:
        started.add(fetcher)
        if len(started) == 3:
            _ = all_started.set()
        _ = await asyncio.wait_for(all_started.wait(), timeout=1)
        return len(fetcher)

    monkeypatch.setattr(counting, "count_fetcher", fake_count)

    counts = await count_fetchers(tmp_path, "abc", ["three", "one", "two"])

    assert counts == {"one": 3, "three": 5, "two": 3}
