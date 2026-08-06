import asyncio
import os
import shlex
from collections.abc import Iterable
from pathlib import Path

from loguru import logger


class GrepError(RuntimeError):
    pass


CPU_CORES = os.process_cpu_count() or 1
RIPGREP_WORKERS = max(CPU_CORES - 1, 1)


async def find_nix_files(repository: Path, commit: str) -> bytes:
    logger.debug("Finding Nix files once at {}", commit)
    process = await asyncio.create_subprocess_exec(
        "fd",
        "-0",
        "-e",
        "nix",
        cwd=repository,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        message = stderr.decode(errors="replace").strip()
        raise GrepError(f"failed to find Nix files at commit {commit}: {message}")
    logger.debug("Cached {} Nix file paths at {}", stdout.count(b"\0"), commit)
    return stdout


async def count_fetcher(
    repository: Path,
    commit: str,
    fetcher: str,
    *,
    nix_files: bytes | None = None,
) -> int:
    if nix_files is None:
        nix_files = await find_nix_files(repository, commit)

    logger.debug("Counting {} at {}", fetcher, commit)
    command = f"xargs -0 -r rg -w {shlex.quote(fetcher)} | wc -l"
    process = await asyncio.create_subprocess_shell(
        command,
        cwd=repository,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate(nix_files)

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
    if not unique_fetchers:
        logger.debug("No active fetchers to count at {}", commit)
        return {}
    nix_files = await find_nix_files(repository, commit)
    semaphore = asyncio.Semaphore(RIPGREP_WORKERS)
    logger.debug(
        "Starting {} ripgrep operations with at most {} workers at {}",
        len(unique_fetchers),
        RIPGREP_WORKERS,
        commit,
    )

    async def worker(fetcher: str) -> int:
        logger.debug("Waiting for a ripgrep worker for {} at {}", fetcher, commit)
        async with semaphore:
            return await count_fetcher(
                repository,
                commit,
                fetcher,
                nix_files=nix_files,
            )

    counts = await asyncio.gather(
        *(worker(fetcher) for fetcher in unique_fetchers)
    )
    result = dict(zip(unique_fetchers, counts, strict=True))
    logger.debug("Finished all fetcher counts at {}: {}", commit, result)
    return result
