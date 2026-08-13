import asyncio
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

from loguru import logger

from fetcher_counter.processes import communicate_cancellable


class GrepError(RuntimeError):
    pass


class IncrementalCountError(RuntimeError):
    pass


_WORD_CHARACTER = re.compile(r"\w")
_HUNK_HEADER = re.compile(rb"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")


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


def _matching_fetchers(
    raw_line: bytes,
    matcher: re.Pattern[str],
    prefixes: Mapping[str, tuple[str, ...]],
) -> set[str]:
    line = raw_line.decode(errors="replace")
    matching_fetchers: set[str] = set()
    for match in matcher.finditer(line):
        matching_fetchers.update(prefixes[match.group(1)])
    return matching_fetchers


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
    stdout, stderr = await communicate_cancellable(process)
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
        for fetcher in _matching_fetchers(raw_line, matcher, prefixes):
            counts[fetcher] += 1

    logger.debug("Finished all fetcher counts at {}: {}", commit, counts)
    return counts


def _hunk_length(value: bytes | None) -> int:
    return 1 if value is None else int(value)


def _apply_diff(
    output: bytes,
    counts: dict[str, int],
    matcher: re.Pattern[str],
    prefixes: Mapping[str, tuple[str, ...]],
) -> None:
    old_remaining = 0
    new_remaining = 0

    for raw_line in output.split(b"\n"):
        header = _HUNK_HEADER.match(raw_line)
        if header is not None:
            if old_remaining or new_remaining:
                raise IncrementalCountError("incomplete Git diff hunk")
            old_remaining = _hunk_length(header.group(1))
            new_remaining = _hunk_length(header.group(2))
            continue

        if not old_remaining and not new_remaining:
            continue
        if not raw_line:
            break

        marker = raw_line[:1]
        if marker == b"-" and old_remaining:
            old_remaining -= 1
            for fetcher in _matching_fetchers(raw_line[1:], matcher, prefixes):
                counts[fetcher] -= 1
        elif marker == b"+" and new_remaining:
            new_remaining -= 1
            for fetcher in _matching_fetchers(raw_line[1:], matcher, prefixes):
                counts[fetcher] += 1
        elif marker == b" " and old_remaining and new_remaining:
            old_remaining -= 1
            new_remaining -= 1
        elif marker == b"\\":
            continue
        else:
            raise IncrementalCountError("malformed Git diff hunk")

    if old_remaining or new_remaining:
        raise IncrementalCountError("incomplete Git diff hunk")


async def update_fetcher_counts(
    repository: Path,
    base_commit: str,
    target_commit: str,
    base_counts: Mapping[str, int],
) -> dict[str, int]:
    counts = dict(sorted(base_counts.items()))
    if not counts:
        return {}
    if any(count < 0 for count in counts.values()):
        raise IncrementalCountError("base fetcher count is negative")

    logger.debug(
        "Updating {} fetcher counts from {} to {} with Git diff",
        len(counts),
        base_commit,
        target_commit,
    )
    command = (
        "git",
        "-C",
        str(repository),
        "diff",
        "--unified=0",
        "--no-color",
        "--no-ext-diff",
        "--no-renames",
        "--text",
        base_commit,
        target_commit,
        "--",
        "*.nix",
    )
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await communicate_cancellable(process)
    if process.returncode != 0:
        message = stderr.decode(errors="replace").strip()
        logger.debug(
            "Git diff from {} to {} failed with exit code {}: {}",
            base_commit,
            target_commit,
            process.returncode,
            message,
        )
        detail = f"failed to update counts from {base_commit} to " + (
            f"{target_commit}: {message}"
        )
        raise IncrementalCountError(detail)

    matcher, prefixes = _line_matcher(list(counts))
    _apply_diff(stdout, counts, matcher, prefixes)
    negative = sorted(fetcher for fetcher, count in counts.items() if count < 0)
    if negative:
        raise IncrementalCountError(
            f"incremental counts became negative for {', '.join(negative)}"
        )

    logger.debug("Finished incremental counts at {}: {}", target_commit, counts)
    return counts
