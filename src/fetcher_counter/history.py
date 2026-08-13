import asyncio
import contextlib
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from fetcher_counter.processes import TERMINATE_TIMEOUT, communicate_cancellable


@dataclass(frozen=True, slots=True)
class SampledCommit:
    commit: str
    date: str


class GitCommandError(RuntimeError):
    pass


HISTORY_REF = "refs/fetcher-counter/history-tip"
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
    process = await asyncio.create_subprocess_exec(
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
    process = await asyncio.create_subprocess_exec(
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

    async def stop_git_log_process() -> None:
        nonlocal cleanup_task
        if cleanup_task is None:
            cleanup_task = asyncio.create_task(reconcile_tasks())
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
        try:
            await asyncio.wait_for(
                asyncio.shield(cleanup_task),
                timeout=TERMINATE_TIMEOUT,
            )
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await asyncio.shield(cleanup_task)
        except Exception as error:  # noqa: BLE001
            logger.warning("Git log cleanup failed: {}", error)

    try:
        (samples, commit_count), stderr, returncode = await asyncio.shield(
            completion
        )
    except asyncio.CancelledError:
        await stop_git_log_process()
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
