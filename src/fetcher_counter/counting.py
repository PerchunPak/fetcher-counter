import asyncio
from collections.abc import Iterable
from pathlib import Path


class GrepError(RuntimeError):
    pass


async def count_fetcher(repository: Path, commit: str, fetcher: str) -> int:
    process = await asyncio.create_subprocess_exec(
        "git",
        "-C",
        str(repository),
        "grep",
        "-F",
        "-w",
        "-o",
        "--no-color",
        "-e",
        fetcher,
        commit,
        "--",
        "*.nix",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode == 1:
        return 0
    if process.returncode != 0:
        message = stderr.decode(errors="replace").strip()
        raise GrepError(
            f"failed to count {fetcher!r} at commit {commit}: {message}"
        )
    return len(stdout.splitlines())


async def count_fetchers(
    repository: Path,
    commit: str,
    fetchers: Iterable[str],
) -> dict[str, int]:
    unique_fetchers = sorted(set(fetchers))
    counts = await asyncio.gather(
        *(
            count_fetcher(repository, commit, fetcher)
            for fetcher in unique_fetchers
        )
    )
    return dict(zip(unique_fetchers, counts, strict=True))
