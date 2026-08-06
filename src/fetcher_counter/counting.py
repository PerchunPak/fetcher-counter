import asyncio
import shlex
from collections.abc import Iterable
from pathlib import Path

from loguru import logger


class GrepError(RuntimeError):
    pass


async def count_fetcher(repository: Path, commit: str, fetcher: str) -> int:
    logger.debug("Counting {} at {}", fetcher, commit)
    command = f"fd -e nix | xargs rg -w {shlex.quote(fetcher)} | wc -l"
    process = await asyncio.create_subprocess_shell(
        command,
        cwd=repository,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    message = stderr.decode(errors="replace").strip()
    if process.returncode != 0 or message:
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
    try:
        count = int(stdout)
    except ValueError as error:
        output = stdout.decode(errors="replace").strip()
        raise GrepError(
            f"failed to parse count for {fetcher!r} at commit {commit}: {output!r}"
        ) from error
    logger.debug("Counted {} matching lines for {} at {}", count, fetcher, commit)
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
