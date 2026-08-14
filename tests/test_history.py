import asyncio
import gc
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

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
        self.stdout: asyncio.StreamReader = asyncio.StreamReader()
        self.stdout.feed_data(stdout)
        self.stdout.feed_eof()
        self.stderr: asyncio.StreamReader = asyncio.StreamReader()
        self.stderr.feed_data(stderr)
        self.stderr.feed_eof()

    async def wait(self) -> int:
        return self.returncode


class StreamingProcess:
    """A `git log` process that only ends once it is asked to."""

    def __init__(self, *, stdout: bytes = b"", stderr: bytes = b"") -> None:
        self.stdout: asyncio.StreamReader = asyncio.StreamReader()
        self.stdout.feed_data(stdout)
        self.stderr: asyncio.StreamReader = asyncio.StreamReader()
        self.stderr.feed_data(stderr)
        self.terminated: int = 0
        self.killed: int = 0
        self._exited: asyncio.Event = asyncio.Event()

    async def wait(self) -> int:
        _ = await self._exited.wait()
        return 0

    def terminate(self) -> None:
        self.terminated += 1
        self._exit()

    def kill(self) -> None:
        self.killed += 1
        self._exit()

    def _exit(self) -> None:
        if not self._exited.is_set():
            self.stdout.feed_eof()
            self.stderr.feed_eof()
            self._exited.set()


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


@pytest.mark.asyncio
async def test_sampled_commits_streams_oldest_first_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def fake_tip(_repository: Path) -> str:
        return "tip"

    async def create_process(
        *arguments: object,
        **options: object,
    ) -> Process:
        calls.append((arguments, options))
        return Process(
            0,
            stdout=b"oldest\x002001\nsecond\x002002\nthird\x002003\nfourth\x002004\nnewest\x002005\n",
        )

    monkeypatch.setattr(history, "history_tip", fake_tip)
    monkeypatch.setattr("asyncio.create_subprocess_exec", create_process)

    samples = await sampled_commits(tmp_path, interval=2)

    assert samples == [
        history.SampledCommit("newest", "2005"),
        history.SampledCommit("third", "2003"),
        history.SampledCommit("oldest", "2001"),
    ]
    assert len(calls) == 1
    arguments, options = calls[0]
    assert arguments == (
        "git",
        "-C",
        str(tmp_path),
        "log",
        "--first-parent",
        "--reverse",
        "--format=%H%x00%cI",
        "tip",
    )
    assert options["stdout"] == asyncio.subprocess.PIPE
    assert options["stderr"] == asyncio.subprocess.PIPE
    assert options["start_new_session"] is True


@pytest.mark.asyncio
async def test_sampled_commits_reports_streamed_log_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fake_tip(_repository: Path) -> str:
        return "tip"

    async def create_process(
        *_arguments: object,
        **_options: object,
    ) -> Process:
        return Process(2, stderr=b"log failed")

    monkeypatch.setattr(history, "history_tip", fake_tip)
    monkeypatch.setattr("asyncio.create_subprocess_exec", create_process)

    with pytest.raises(GitCommandError, match=r"git log.*log failed"):
        _ = await sampled_commits(tmp_path, interval=2)


def record_created_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> list[asyncio.Task[Any]]:
    created: list[asyncio.Task[Any]] = []
    original = asyncio.create_task

    def create_task(
        coroutine: Coroutine[Any, Any, Any],
        **options: Any,
    ) -> asyncio.Task[Any]:
        task = original(coroutine, **options)
        created.append(task)
        return task

    monkeypatch.setattr(asyncio, "create_task", create_task)
    return created


def record_loop_exceptions() -> list[dict[str, Any]]:
    reported: list[dict[str, Any]] = []
    asyncio.get_running_loop().set_exception_handler(
        lambda _loop, context: reported.append(context)
    )
    return reported


async def assert_tasks_reconciled(
    created: list[asyncio.Task[Any]],
    reported: list[dict[str, Any]],
) -> None:
    assert created
    assert all(task.done() for task in created)
    assert not any(task.cancelled() for task in created)
    _ = gc.collect()
    await asyncio.sleep(0)
    assert reported == []


@pytest.mark.asyncio
async def test_streamed_log_reconciles_tasks_after_parse_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    process = StreamingProcess(stdout=b"missing-separator\n")

    async def fake_tip(_repository: Path) -> str:
        return "tip"

    async def create_process(
        *_arguments: object,
        **_options: object,
    ) -> StreamingProcess:
        return process

    monkeypatch.setattr(history, "history_tip", fake_tip)
    monkeypatch.setattr("asyncio.create_subprocess_exec", create_process)
    reported = record_loop_exceptions()
    created = record_created_tasks(monkeypatch)

    with pytest.raises(GitCommandError, match="failed to read git history") as (
        error
    ):
        _ = await sampled_commits(tmp_path, interval=1)

    assert isinstance(error.value.__cause__, ValueError)
    assert process.terminated == 1
    assert process.killed == 0
    await assert_tasks_reconciled(created, reported)


@pytest.mark.asyncio
async def test_streamed_log_reconciles_tasks_after_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    process = StreamingProcess(stdout=b"oldest\x002001\n")

    async def fake_tip(_repository: Path) -> str:
        return "tip"

    async def create_process(
        *_arguments: object,
        **_options: object,
    ) -> StreamingProcess:
        return process

    monkeypatch.setattr(history, "history_tip", fake_tip)
    monkeypatch.setattr("asyncio.create_subprocess_exec", create_process)
    reported = record_loop_exceptions()
    sampling = asyncio.create_task(sampled_commits(tmp_path, interval=1))
    created = record_created_tasks(monkeypatch)
    await asyncio.sleep(0.01)

    _ = sampling.cancel()
    await asyncio.sleep(0)

    assert not sampling.done()
    assert process.terminated == 0
    assert process.killed == 0

    process._exit()
    with pytest.raises(asyncio.CancelledError):
        _ = await sampling

    assert process.terminated == 0
    assert process.killed == 0
    await assert_tasks_reconciled(created, reported)
