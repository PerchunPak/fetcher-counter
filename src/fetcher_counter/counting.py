import asyncio
import re
from collections.abc import Iterable
from pathlib import Path

from loguru import logger


class GrepError(RuntimeError):
    pass


_WORD_CHARACTER = re.compile(r"\w")


def _matching_prefixes(
    fetcher: str,
    fetchers: list[str],
) -> tuple[str, ...]:
    return tuple(
        candidate
        for candidate in fetchers
        if fetcher.startswith(candidate)
        and (
            len(candidate) == len(fetcher)
            or _WORD_CHARACTER.fullmatch(fetcher[len(candidate)]) is None
        )
    )


def _line_matcher(
    fetchers: list[str],
) -> tuple[re.Pattern[str], dict[str, tuple[str, ...]]]:
    longest_first = sorted(fetchers, key=lambda fetcher: (-len(fetcher), fetcher))
    alternatives = "|".join(re.escape(fetcher) for fetcher in longest_first)
    matcher = re.compile(rf"(?=(?<!\w)({alternatives})(?!\w))")
    prefixes = {
        fetcher: _matching_prefixes(fetcher, fetchers) for fetcher in fetchers
    }
    return matcher, prefixes


async def count_fetchers(
    repository: Path,
    commit: str,
    fetchers: Iterable[str],
) -> dict[str, int]:
    unique_fetchers = sorted(set(fetchers))
    if not unique_fetchers:
        logger.debug("No active fetchers to count at {}", commit)
        return {}

    command = [
        "rg",
        "--no-filename",
        "--no-line-number",
        "--color=never",
        "--fixed-strings",
        "--word-regexp",
    ]
    for fetcher in unique_fetchers:
        command.extend(("-e", fetcher))
    command.extend(("--glob", "*.nix", "."))

    logger.debug(
        "Counting {} fetchers with one ripgrep scan at {}",
        len(unique_fetchers),
        commit,
    )
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=repository,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    message = stderr.decode(errors="replace").strip()
    if process.returncode not in {0, 1} or message:
        logger.debug(
            "Grep at {} failed with exit code {}: {}",
            commit,
            process.returncode,
            message,
        )
        raise GrepError(f"failed to count fetchers at commit {commit}: {message}")

    matcher, prefixes = _line_matcher(unique_fetchers)
    counts = dict.fromkeys(unique_fetchers, 0)
    for raw_line in stdout.split(b"\n"):
        line = raw_line.decode(errors="replace")
        matching_fetchers: set[str] = set()
        for match in matcher.finditer(line):
            matching_fetchers.update(prefixes[match.group(1)])
        for fetcher in matching_fetchers:
            counts[fetcher] += 1

    logger.debug("Finished all fetcher counts at {}: {}", commit, counts)
    return counts
