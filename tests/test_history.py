import asyncio
from pathlib import Path

import pytest

from fetcher_counter import history
from fetcher_counter.history import GitCommandError, checkout, sampled_commits
from tests.conftest import GIT, commit_file, run_git


class Process:
    def __init__(
        self,
        returncode: int,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
    ) -> None:
        self.returncode: int = returncode
        self.stdout: bytes = stdout
        self.stderr: bytes = stderr
        self.input: bytes | None = None

    async def communicate(
        self,
        input: bytes | None = None,
    ) -> tuple[bytes, bytes]:
        self.input = input
        return self.stdout, self.stderr


def index_lock_error(repository: Path) -> GitCommandError:
    template = "git checkout failed: fatal: Unable to create '{}': File exists."
    return GitCommandError(template.format(repository / ".git/index.lock"))


@pytest.mark.asyncio
async def test_samples_are_oldest_anchored_and_traversed_newest_first(
    repository: Path,
) -> None:
    hashes = [commit_file(repository, number) for number in range(6)]

    samples = await sampled_commits(repository, interval=2)

    assert [sample.commit for sample in samples] == [
        hashes[4],
        hashes[2],
        hashes[0],
    ]
    assert all(sample.date for sample in samples)

    newest = commit_file(repository, 6)
    samples_after_pull = await sampled_commits(repository, interval=2)

    assert [sample.commit for sample in samples_after_pull] == [
        newest,
        hashes[4],
        hashes[2],
        hashes[0],
    ]


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


@pytest.mark.asyncio
async def test_checkout_retries_index_lock_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    attempts = 0
    delays: list[float] = []

    async def run_git(_repository: Path, *_arguments: str) -> bytes:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise index_lock_error(tmp_path)
        return b""

    async def sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(history, "_run_git", run_git)
    monkeypatch.setattr(asyncio, "sleep", sleep)

    await checkout(tmp_path, "commit")

    assert attempts == 3
    assert delays == [1.0, 2.0]


@pytest.mark.asyncio
async def test_checkout_stops_retrying_index_lock_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    attempts = 0

    async def run_git(_repository: Path, *_arguments: str) -> bytes:
        nonlocal attempts
        attempts += 1
        raise index_lock_error(tmp_path)

    async def sleep(_delay: float) -> None:
        return

    monkeypatch.setattr(history, "_run_git", run_git)
    monkeypatch.setattr(asyncio, "sleep", sleep)

    with pytest.raises(GitCommandError, match=r"index\.lock"):
        await checkout(tmp_path, "commit")

    assert attempts == history.CHECKOUT_INDEX_LOCK_RETRIES + 1


@pytest.mark.asyncio
async def test_checkout_does_not_retry_other_git_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    attempts = 0

    async def run_git(_repository: Path, *_arguments: str) -> bytes:
        nonlocal attempts
        attempts += 1
        raise GitCommandError("git checkout failed: unknown revision")

    monkeypatch.setattr(history, "_run_git", run_git)

    with pytest.raises(GitCommandError, match="unknown revision"):
        await checkout(tmp_path, "commit")

    assert attempts == 1


def commit_record(commit: str, timestamp: int, offset: str = "+0000") -> bytes:
    raw_commit = (
        "tree 0000000000000000000000000000000000000000\n"
        + f"committer Test <test@example.com> {timestamp} {offset}\n"
        + "\nmessage\n"
    ).encode()
    return f"{commit} commit {len(raw_commit)}\n".encode() + raw_commit + b"\n"


@pytest.mark.parametrize(
    ("first_parent", "history_arguments"),
    [(True, ("--first-parent",)), (False, ())],
)
@pytest.mark.asyncio
async def test_sampled_commits_reads_dates_only_for_selected_revisions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    first_parent: bool,
    history_arguments: tuple[str, ...],
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object], Process]] = []
    commits = ["newest", "fourth", "third", "second", "oldest"]
    selected_chunks = [
        [("newest", 978652800)],
        [("oldest", 978307200)],
    ]

    async def fake_tip(_repository: Path) -> str:
        return "tip"

    async def create_process(
        *arguments: object,
        **options: object,
    ) -> Process:
        if len(calls) == 0:
            process = Process(0, stdout=("\n".join(commits) + "\n").encode())
        else:
            process = Process(
                0,
                stdout=b"".join(
                    commit_record(commit, timestamp)
                    for commit, timestamp in selected_chunks[len(calls) - 1]
                ),
            )
        calls.append((arguments, options, process))
        return process

    monkeypatch.setattr(history, "history_tip", fake_tip)
    monkeypatch.setattr(history, "COMMIT_DATE_WORKERS", 2)
    monkeypatch.setattr("asyncio.create_subprocess_exec", create_process)

    samples = await sampled_commits(
        tmp_path,
        interval=2,
        first_parent=first_parent,
        completed={"third"},
    )

    assert samples == [
        history.SampledCommit("newest", "2001-01-05T00:00:00Z"),
        history.SampledCommit("third", ""),
        history.SampledCommit("oldest", "2001-01-01T00:00:00Z"),
    ]
    assert len(calls) == 3
    arguments, options, process = calls[0]
    assert arguments == (
        "git",
        "-C",
        str(tmp_path),
        "rev-list",
        *history_arguments,
        "tip",
    )
    assert options["stdin"] is None
    assert process.input is None

    for call, expected_input in zip(
        calls[1:],
        (b"newest\n", b"oldest\n"),
        strict=True,
    ):
        arguments, options, process = call
        assert arguments == (
            "git",
            "-C",
            str(tmp_path),
            "cat-file",
            "--batch",
        )
        assert options["stdin"] == asyncio.subprocess.PIPE
        assert process.input == expected_input


@pytest.mark.asyncio
async def test_sampled_commits_preserves_committer_timezone(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fake_tip(_repository: Path) -> str:
        return "tip"

    calls = 0

    async def create_process(
        *_arguments: object,
        **_options: object,
    ) -> Process:
        nonlocal calls
        calls += 1
        if calls == 1:
            return Process(0, stdout=b"commit\n")
        return Process(0, stdout=commit_record("commit", 978307200, "+0530"))

    monkeypatch.setattr(history, "history_tip", fake_tip)
    monkeypatch.setattr("asyncio.create_subprocess_exec", create_process)

    assert await sampled_commits(tmp_path, interval=1) == [
        history.SampledCommit("commit", "2001-01-01T05:30:00+05:30")
    ]


@pytest.mark.asyncio
async def test_sampled_commits_reports_rev_list_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fake_tip(_repository: Path) -> str:
        return "tip"

    async def create_process(
        *_arguments: object,
        **_options: object,
    ) -> Process:
        return Process(2, stderr=b"revision walk failed")

    monkeypatch.setattr(history, "history_tip", fake_tip)
    monkeypatch.setattr("asyncio.create_subprocess_exec", create_process)

    with pytest.raises(
        GitCommandError, match=r"git rev-list.*revision walk failed"
    ):
        _ = await sampled_commits(tmp_path, interval=2)


@pytest.mark.asyncio
async def test_sampled_commits_rejects_malformed_commit_batch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fake_tip(_repository: Path) -> str:
        return "tip"

    calls = 0

    async def create_process(
        *_arguments: object,
        **_options: object,
    ) -> Process:
        nonlocal calls
        calls += 1
        if calls == 1:
            return Process(0, stdout=b"commit\n")
        return Process(0, stdout=b"commit commit 4\nbad\n\n")

    monkeypatch.setattr(history, "history_tip", fake_tip)
    monkeypatch.setattr("asyncio.create_subprocess_exec", create_process)

    with pytest.raises(GitCommandError, match="invalid committer date"):
        _ = await sampled_commits(tmp_path, interval=1)
