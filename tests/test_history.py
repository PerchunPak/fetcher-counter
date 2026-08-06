import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

from fetcher_counter.history import GitCommandError, checkout, sampled_commits

git_path = shutil.which("git")
if git_path is None:
    raise RuntimeError("git is required for these tests")
GIT: str = git_path


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
    _ = subprocess.run(  # noqa: S603
        [GIT, "init", "-q", str(tmp_path)], check=True
    )
    return tmp_path


@pytest.mark.asyncio
async def test_sampled_commits_are_oldest_anchored(repository: Path) -> None:
    hashes = [commit_file(repository, number) for number in range(5)]

    samples = await sampled_commits(repository, interval=2)

    assert [sample.commit for sample in samples] == hashes[::2]
    assert all(sample.date for sample in samples)


@pytest.mark.asyncio
async def test_sampled_commits_rejects_invalid_interval(repository: Path) -> None:
    with pytest.raises(ValueError, match="positive"):
        _ = await sampled_commits(repository, interval=0)


@pytest.mark.asyncio
async def test_checkout_detaches_at_requested_commit(repository: Path) -> None:
    first = commit_file(repository, 1)
    _ = commit_file(repository, 2)

    await checkout(repository, first)

    head = await asyncio.to_thread(run_git, repository, "rev-parse", "HEAD")
    symbolic_ref = await asyncio.create_subprocess_exec(
        GIT,
        "-C",
        str(repository),
        "symbolic-ref",
        "-q",
        "HEAD",
    )
    assert head == first
    assert await symbolic_ref.wait() == 1


@pytest.mark.asyncio
async def test_sampling_keeps_tip_after_historical_checkout(
    repository: Path,
) -> None:
    hashes = [commit_file(repository, number) for number in range(3)]
    before = await sampled_commits(repository, interval=1)
    await checkout(repository, hashes[0])

    after = await sampled_commits(repository, interval=1)

    assert after == before


@pytest.mark.asyncio
async def test_checkout_reports_git_failure(repository: Path) -> None:
    with pytest.raises(GitCommandError, match="checkout"):
        await checkout(repository, "not-a-commit")
