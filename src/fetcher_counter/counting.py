import asyncio
from collections.abc import Iterable
from pathlib import Path

from loguru import logger


class GrepError(RuntimeError):
    pass


async def count_fetcher(repository: Path, commit: str, fetcher: str) -> int:
    logger.debug("Counting {} at {}", fetcher, commit)
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
        logger.debug("Found no occurrences of {} at {}", fetcher, commit)
        return 0
    if process.returncode != 0:
        message = stderr.decode(errors="replace").strip()
        logger.debug(
            "Grep for {} at {} failed with exit code {}: {}",
            fetcher,
            commit,
            process.returncode,
            message,
        )
        raise GrepError(
            f"failed to count {fetcher!r} at commit {commit}: {message}"
        )
    count = len(stdout.splitlines())
    logger.debug("Counted {} occurrences of {} at {}", count, fetcher, commit)
    return count


async def count_fetchers(
    repository: Path,
    commit: str,
    fetchers: Iterable[str],
) -> dict[str, int]:
    unique_fetchers = sorted(set(fetchers))
    logger.debug(
        "Starting {} concurrent grep operations at {}",
        len(unique_fetchers),
        commit,
    )
    counts = await asyncio.gather(
        *(
            count_fetcher(repository, commit, fetcher)
            for fetcher in unique_fetchers
        )
    )
    result = dict(zip(unique_fetchers, counts, strict=True))
    logger.debug("Finished all fetcher counts at {}: {}", commit, result)
    return result
