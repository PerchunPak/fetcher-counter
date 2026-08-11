import asyncio
from dataclasses import dataclass
from pathlib import Path

from loguru import logger


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
    stdout, stderr = await process.communicate()
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
    stderr_task = asyncio.create_task(process.stderr.read())

    samples: list[SampledCommit] = []
    commit_count = 0
    async for raw_line in process.stdout:
        if commit_count % interval == 0:
            commit, date = (
                raw_line.rstrip(b"\r\n").decode().split("\0", maxsplit=1)
            )
            samples.append(SampledCommit(commit=commit, date=date))
        commit_count += 1

    returncode = await process.wait()
    stderr = await stderr_task
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
