import asyncio
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SampledCommit:
    commit: str
    date: str


class GitCommandError(RuntimeError):
    pass


async def _run_git(repository: Path, *arguments: str) -> bytes:
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
        raise GitCommandError(f"git {' '.join(arguments)} failed: {message}")
    return stdout


async def sampled_commits(
    repository: Path,
    *,
    interval: int = 50,
) -> list[SampledCommit]:
    if interval < 1:
        raise ValueError("interval must be positive")

    output = await _run_git(
        repository,
        "log",
        "--first-parent",
        "--reverse",
        "--format=%H%x00%cI",
        "HEAD",
    )
    commits: list[SampledCommit] = []
    for line in output.decode().splitlines():
        commit, date = line.split("\0", maxsplit=1)
        commits.append(SampledCommit(commit=commit, date=date))
    return commits[::interval]


async def checkout(repository: Path, commit: str) -> None:
    _ = await _run_git(repository, "checkout", "--detach", "--force", commit)
