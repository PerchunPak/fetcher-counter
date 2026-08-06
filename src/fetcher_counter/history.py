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
    output = await _run_git(
        repository,
        "log",
        "--first-parent",
        "--format=%H%x00%cI",
        tip,
    )
    commits: list[SampledCommit] = []
    for line in output.decode().splitlines():
        commit, date = line.split("\0", maxsplit=1)
        commits.append(SampledCommit(commit=commit, date=date))
    offset = (len(commits) - 1) % interval
    samples = commits[offset::interval]
    logger.debug(
        "Selected {} of {} commits with oldest-anchored interval {}",
        len(samples),
        len(commits),
        interval,
    )
    logger.debug("Traversing selected commits from newest to oldest")
    return samples


async def checkout(repository: Path, commit: str) -> None:
    logger.debug("Checking out Nixpkgs commit {}", commit)
    _ = await _run_git(repository, "checkout", "--detach", "--force", commit)
