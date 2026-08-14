import asyncio
import contextlib
import fcntl
from collections.abc import Generator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from fetcher_counter.processes import (
    TERMINATE_TIMEOUT,
    communicate_cancellable,
    create_subprocess_exec,
)


@dataclass(frozen=True, slots=True)
class SampledCommit:
    commit: str
    date: str


@dataclass(frozen=True, slots=True)
class WorktreeRequest:
    index: int
    initial_commit: str


@dataclass(frozen=True, slots=True)
class WorktreeRecord:
    path: Path
    attributes: dict[str, str] = field(default_factory=dict[str, str])


class GitCommandError(RuntimeError):
    pass


class WorktreeError(RuntimeError):
    pass


class WorktreePoolLockedError(RuntimeError):
    pass


HISTORY_REF = "refs/fetcher-counter/history-tip"
POOL_LOCK_NAME = "coordinator.lock"
STATUS_DETAIL_LINES = 5
CHECKOUT_INDEX_LOCK_RETRIES = 3
CHECKOUT_INDEX_LOCK_RETRY_DELAY = 1.0
# Historical checkouts of Nixpkgs are dominated by index work rather than by
# writing the few files that differ: the index holds tens of thousands of
# entries and is rewritten on every checkout. Version 4 compresses path names,
# skipping the trailing hash avoids checksumming the whole index, and parallel
# workers speed up the revisions that do rewrite many files.
CHECKOUT_OPTIONS = (
    "-c",
    "index.version=4",
    "-c",
    "index.skipHash=true",
    "-c",
    "checkout.workers=0",
)


async def _run_git(repository: Path, *arguments: str) -> bytes:
    logger.debug("Running git in {}: {}", repository, " ".join(arguments))
    process = await create_subprocess_exec(
        "git",
        "-C",
        str(repository),
        *arguments,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await communicate_cancellable(process)
    if process.returncode != 0:
        message = stderr.decode(errors="replace").strip()
        logger.debug(
            "Git command failed with exit code {}: {}",
            process.returncode,
            message,
        )
        raise GitCommandError(f"git {' '.join(arguments)} failed: {message}")
    logger.debug("Git command returned {} bytes", len(stdout))
    return stdout


async def history_tip(repository: Path) -> str:
    head = (await _run_git(repository, "rev-parse", "HEAD")).decode().strip()
    try:
        saved = (
            (await _run_git(repository, "rev-parse", "--verify", HISTORY_REF))
            .decode()
            .strip()
        )
    except GitCommandError:
        logger.debug("Saving initial history tip {} at {}", head, HISTORY_REF)
        _ = await _run_git(repository, "update-ref", HISTORY_REF, head)
        return head

    try:
        _ = await _run_git(repository, "merge-base", "--is-ancestor", saved, head)
    except GitCommandError:
        logger.debug("Using saved history tip {} instead of HEAD {}", saved, head)
        return saved
    logger.debug("Advancing saved history tip from {} to {}", saved, head)
    _ = await _run_git(repository, "update-ref", HISTORY_REF, head)
    return head


async def _read_samples(
    stdout: asyncio.StreamReader,
    *,
    interval: int,
) -> tuple[list[SampledCommit], int]:
    samples: list[SampledCommit] = []
    commit_count = 0
    async for raw_line in stdout:
        if commit_count % interval == 0:
            commit, date = (
                raw_line.rstrip(b"\r\n").decode().split("\0", maxsplit=1)
            )
            samples.append(SampledCommit(commit=commit, date=date))
        commit_count += 1
    return samples, commit_count


async def sampled_commits(
    repository: Path,
    *,
    interval: int = 50,
) -> list[SampledCommit]:
    if interval < 1:
        raise ValueError("interval must be positive")

    tip = await history_tip(repository)
    arguments = (
        "log",
        "--first-parent",
        "--reverse",
        "--format=%H%x00%cI",
        tip,
    )
    logger.debug("Running git in {}: {}", repository, " ".join(arguments))
    process = await create_subprocess_exec(
        "git",
        "-C",
        str(repository),
        *arguments,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    stdout_task = asyncio.create_task(
        _read_samples(process.stdout, interval=interval)
    )
    stderr_task = asyncio.create_task(process.stderr.read())
    wait_task = asyncio.create_task(process.wait())
    completion = asyncio.gather(stdout_task, stderr_task, wait_task)
    cleanup_task: asyncio.Task[None] | None = None

    async def reconcile_tasks() -> None:
        """Wait out every owned task, whatever already failed.

        `completion` fails as soon as one of its children does, so awaiting it
        again would return immediately instead of waiting for the other two.
        A separate gather over the same tasks is what actually reconciles
        them; `completion` itself is consumed afterwards so asyncio does not
        report its exception as never retrieved.
        """
        results = await asyncio.gather(
            stdout_task,
            stderr_task,
            wait_task,
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                logger.debug("Git log task ended during cleanup: {}", result)
        if completion.done():
            with contextlib.suppress(BaseException):
                _ = completion.result()

    async def reconcile_git_log_process() -> None:
        nonlocal cleanup_task
        if cleanup_task is None:
            cleanup_task = asyncio.create_task(reconcile_tasks())
        await asyncio.shield(cleanup_task)

    async def stop_git_log_process() -> None:
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
        try:
            await asyncio.wait_for(
                reconcile_git_log_process(),
                timeout=TERMINATE_TIMEOUT,
            )
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await reconcile_git_log_process()
        except Exception as error:  # noqa: BLE001
            logger.warning("Git log cleanup failed: {}", error)

    try:
        (samples, commit_count), stderr, returncode = await asyncio.shield(
            completion
        )
    except asyncio.CancelledError:
        await reconcile_git_log_process()
        raise
    except Exception as error:
        await stop_git_log_process()
        raise GitCommandError(f"failed to read git history: {error}") from error
    if returncode != 0:
        message = stderr.decode(errors="replace").strip()
        logger.debug(
            "Git command failed with exit code {}: {}",
            returncode,
            message,
        )
        raise GitCommandError(f"git {' '.join(arguments)} failed: {message}")

    samples.reverse()
    logger.debug(
        "Selected {} of {} commits with oldest-anchored interval {}",
        len(samples),
        commit_count,
        interval,
    )
    logger.debug("Traversing selected commits from newest to oldest")
    return samples


def _is_index_lock_error(error: GitCommandError) -> bool:
    message = str(error)
    return (
        "fatal: Unable to create '" in message
        and "index.lock': File exists." in message
    )


async def checkout(repository: Path, commit: str) -> None:
    logger.debug("Checking out Nixpkgs commit {}", commit)
    for retry in range(CHECKOUT_INDEX_LOCK_RETRIES + 1):
        try:
            _ = await _run_git(
                repository,
                *CHECKOUT_OPTIONS,
                "checkout",
                "--detach",
                "--force",
                "--quiet",
                "--no-recurse-submodules",
                commit,
            )
        except GitCommandError as error:
            if retry == CHECKOUT_INDEX_LOCK_RETRIES or not _is_index_lock_error(
                error
            ):
                raise
            delay = CHECKOUT_INDEX_LOCK_RETRY_DELAY * 2**retry
            logger.warning(
                "Git checkout blocked by index lock; retrying in {} seconds",
                delay,
            )
            await asyncio.sleep(delay)
        else:
            return


def _canonical(path: Path) -> Path:
    """Canonicalize a path that may not exist yet.

    A registered but missing worktree still has to compare against the
    listing, so resolution must not require the path to exist.
    """
    return path.resolve(strict=False)


def worker_name(index: int) -> str:
    return f"worker-{index}"


@contextlib.contextmanager
def worktree_pool_lock(pool_dir: Path) -> Generator[None]:
    """Own the managed worker worktree pool at `pool_dir` for this run.

    The pool directory and its lock file are created here, because the lock
    has to be held before anything else touches shared state. Contention
    fails immediately instead of blocking the event loop, and the lock file
    is deliberately left on disk: the advisory lock, not the file, marks
    ownership, so a crashed run releases it by closing its descriptor.
    """
    if pool_dir.exists() and not pool_dir.is_dir():
        raise WorktreeError(f"worktree pool path is not a directory: {pool_dir}")
    pool_dir.mkdir(parents=True, exist_ok=True)
    lock_path = pool_dir / POOL_LOCK_NAME
    if lock_path.is_dir():
        raise WorktreeError(f"worktree pool lock path is a directory: {lock_path}")
    logger.debug("Locking worktree pool {}", pool_dir)
    lock_file = lock_path.open("a+b")
    try:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise WorktreePoolLockedError(
                f"another fetcher-counter invocation is using {pool_dir}"
            ) from error
        yield
    finally:
        lock_file.close()
        logger.debug("Released worktree pool lock for {}", pool_dir)


def parse_worktree_list(output: bytes) -> list[WorktreeRecord]:
    """Parse `git worktree list --porcelain -z` into records.

    Attributes are NUL terminated and records are separated by an empty
    entry, so paths containing whitespace survive. Attributes beyond
    `worktree` (`bare`, `detached`, `branch`, `locked`, `prunable`) may
    appear in any order and may carry no value.
    """
    records: list[WorktreeRecord] = []
    attributes: dict[str, str] = {}

    def flush() -> None:
        nonlocal attributes
        if not attributes:
            return
        path = attributes.get("worktree")
        if path is None:
            raise WorktreeError(
                "git worktree list returned a record without a path"
            )
        records.append(WorktreeRecord(path=Path(path), attributes=attributes))
        attributes = {}

    for raw_entry in output.split(b"\0"):
        entry = raw_entry.decode(errors="replace")
        if not entry:
            flush()
            continue
        key, _, value = entry.partition(" ")
        attributes[key] = value
    flush()
    return records


async def _worktree_records(repository: Path) -> dict[Path, WorktreeRecord]:
    listing = await _run_git(repository, "worktree", "list", "--porcelain", "-z")
    return {
        _canonical(record.path): record for record in parse_worktree_list(listing)
    }


def _validate_registered_worktree(record: WorktreeRecord, path: Path) -> None:
    if "locked" in record.attributes:
        reason = record.attributes["locked"] or "no reason given"
        raise WorktreeError(
            f"worker worktree {path} is locked ({reason}); fetcher-counter"
            + " refuses to reuse locked worktrees even though Git could"
        )
    if "prunable" in record.attributes:
        reason = record.attributes["prunable"] or "no reason given"
        raise WorktreeError(
            f"worker worktree {path} is prunable ({reason}); fetcher-counter"
            + " refuses to reuse prunable worktrees even though Git could"
        )
    if not path.is_dir():
        raise WorktreeError(
            f"worker worktree {path} is registered but missing; run"
            + " 'git worktree prune' and try again"
        )


async def _require_pristine_worktree(path: Path) -> None:
    """Refuse a worker worktree holding any state at all.

    Ignored files matter as much as untracked ones: a later historical
    checkout can un-ignore a stray `.nix` file, which `count_fetchers()`
    would then count, and any stray file can obstruct that checkout.
    """
    status = await _run_git(
        path,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignored=matching",
    )
    if not status:
        return
    detail = status.decode(errors="replace").strip().splitlines()
    raise WorktreeError(
        f"worker worktree {path} is not pristine: "
        + f"{'; '.join(detail[:STATUS_DETAIL_LINES])}. Inspect and clean it"
        + " manually; fetcher-counter never resets a worktree"
    )


async def _create_worktree(repository: Path, path: Path, commit: str) -> None:
    logger.info("Creating worker worktree {} at {}", path, commit)
    _ = await _run_git(
        repository,
        *CHECKOUT_OPTIONS,
        "worktree",
        "add",
        "--detach",
        str(path),
        commit,
    )


async def provision_worktrees(
    *,
    repository: Path,
    pool_dir: Path,
    requests: Sequence[WorktreeRequest],
) -> dict[int, Path]:
    """Create or safely reuse one persistent worktree per request.

    `pool_dir` is expected to exist already, because `worktree_pool_lock()`
    creates it. Nothing is ever cleaned, reset, or removed here: every
    unexpected state is reported instead, and worktrees created before a
    later failure stay registered for reuse.
    """
    records = await _worktree_records(repository)
    source = _canonical(repository)
    assigned: dict[Path, int] = {}
    worktrees: dict[int, Path] = {}

    for request in requests:
        path = pool_dir / worker_name(request.index)
        target = _canonical(path)
        if target == source:
            raise WorktreeError(
                f"worker worktree {path} is the supplied Nixpkgs checkout"
            )
        if target in assigned:
            raise WorktreeError(
                f"worker worktree {path} is already assigned to shard"
                + f" {assigned[target]}"
            )
        record = records.get(target)
        if record is None:
            if path.exists():
                raise WorktreeError(
                    f"{path} exists but is not a worktree of {repository};"
                    + " it belongs to another repository or is not a worktree"
                    + " at all, so fetcher-counter leaves it untouched"
                )
            await _create_worktree(repository, path, request.initial_commit)
            records[target] = WorktreeRecord(path=path)
        else:
            _validate_registered_worktree(record, path)
            await _require_pristine_worktree(path)
            logger.debug("Reusing pristine worker worktree {}", path)
        assigned[target] = request.index
        worktrees[request.index] = path

    return worktrees
