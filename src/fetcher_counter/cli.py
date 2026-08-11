import argparse
import asyncio
import sys
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from loguru import logger

from fetcher_counter.counting import (
    IncrementalCountError,
    count_fetchers,
    update_fetcher_counts,
)
from fetcher_counter.database import FetcherDatabase
from fetcher_counter.discovery import FetcherDiscoveryError, discover_fetchers
from fetcher_counter.history import checkout, sampled_commits

DEFAULT_INTERVAL = 50
DEFAULT_FULL_SCAN_INTERVAL = 25
DEFAULT_LOG_LEVEL = "INFO"
LOG_LEVELS = ("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL")


@dataclass(frozen=True, slots=True)
class Config:
    nixpkgs: Path
    database: Path
    expression: Path
    interval: int = DEFAULT_INTERVAL
    full_scan_interval: int = DEFAULT_FULL_SCAN_INTERVAL
    log_level: str = DEFAULT_LOG_LEVEL


def default_expression() -> Path:
    packaged = Path(__file__).with_name("get-fetchers.nix")
    if packaged.is_file():
        return packaged
    return Path(__file__).resolve().parents[2] / "get-fetchers.nix"


def _positive_int(value: str) -> int:
    integer = int(value)
    if integer < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return integer


def parse_args(arguments: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(
        description="Count fetcher mentions across sampled Nixpkgs history."
    )
    _ = parser.add_argument(
        "--nixpkgs",
        type=Path,
        default=Path("nixpkgs"),
        help="Nixpkgs Git checkout (default: ./nixpkgs)",
    )
    _ = parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/fetchers.sqlite3"),
        help="SQLite output path (default: ./data/fetchers.sqlite3)",
    )
    _ = parser.add_argument(
        "--expression",
        type=Path,
        default=default_expression(),
        help="fetcher discovery Nix expression",
    )
    _ = parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL,
        help="first-parent commit sampling interval (default: 50)",
    )
    _ = parser.add_argument(
        "--full-scan-interval",
        type=_positive_int,
        default=DEFAULT_FULL_SCAN_INTERVAL,
        help="sampled iterations between full scans (default: 25)",
    )
    _ = parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=LOG_LEVELS,
        default=DEFAULT_LOG_LEVEL,
        help="minimum log level (default: INFO)",
    )
    namespace = parser.parse_args(arguments)
    return Config(
        nixpkgs=namespace.nixpkgs,
        database=namespace.database,
        expression=namespace.expression,
        interval=namespace.interval,
        full_scan_interval=namespace.full_scan_interval,
        log_level=namespace.log_level,
    )


def configure_logging(log_level: str) -> None:
    logger.remove()
    _ = logger.add(sys.stderr, level=log_level)


@contextmanager
def timed(stage: str, commit: str) -> Generator[None]:
    started = perf_counter()
    try:
        yield
    finally:
        logger.info(
            "{} at {} took {:.2f} seconds",
            stage,
            commit,
            perf_counter() - started,
        )


async def run(config: Config) -> None:
    if config.full_scan_interval < 1:
        raise ValueError("full scan interval must be positive")
    logger.debug("Starting fetcher counter with configuration {}", config)
    commits = await sampled_commits(config.nixpkgs, interval=config.interval)
    async with FetcherDatabase(config.database) as database:
        completed = await database.completed_commits()
        pending = [
            (index, sample, commits[index - 1] if index else None)
            for index, sample in enumerate(commits)
            if sample.commit not in completed
        ]
        logger.debug("Skipping completed commit hashes: {}", sorted(completed))
        logger.info(
            "Found {} sampled commits; {} remain",
            len(commits),
            len(pending),
        )

        for position, (sample_index, sample, newer_sample) in enumerate(
            pending,
            start=1,
        ):
            logger.info(
                "Processing {} ({}/{})",
                sample.commit,
                position,
                len(pending),
            )
            with timed("Checkout", sample.commit):
                await checkout(config.nixpkgs, sample.commit)
            try:
                with timed("Discovery", sample.commit):
                    fetchers = await discover_fetchers(
                        config.nixpkgs,
                        config.expression,
                        commit=sample.commit,
                    )
            except FetcherDiscoveryError as error:
                logger.warning("Skipping {}: {}", sample.commit, error)
                _ = await database.store(
                    sample.commit,
                    sample.date,
                    {},
                    is_skipped=True,
                )
                continue
            logger.debug("Active fetchers at {}: {}", sample.commit, fetchers)
            counts: dict[str, int] | None = None
            scheduled_full_scan = (
                sample_index + 1
            ) % config.full_scan_interval == 0
            if scheduled_full_scan:
                logger.debug(
                    "Using scheduled full scan at {} (sample iteration {})",
                    sample.commit,
                    sample_index + 1,
                )
            if newer_sample is not None and not scheduled_full_scan:
                newer_counts = await database.counts_for_commit(
                    newer_sample.commit,
                    fetchers,
                )
                if newer_counts is not None:
                    try:
                        with timed("Incremental count", sample.commit):
                            counts = await update_fetcher_counts(
                                config.nixpkgs,
                                newer_sample.commit,
                                sample.commit,
                                newer_counts,
                            )
                    except IncrementalCountError as error:
                        logger.warning(
                            "Incremental count at {} failed; using full scan: {}",
                            sample.commit,
                            error,
                        )
                    else:
                        logger.debug(
                            "Used counts from adjacent commit {} for {}",
                            newer_sample.commit,
                            sample.commit,
                        )

            if counts is None:
                logger.debug("Using full fetcher scan at {}", sample.commit)
                with timed("Full scan", sample.commit):
                    counts = await count_fetchers(
                        config.nixpkgs,
                        sample.commit,
                        fetchers,
                    )
            logger.debug("Persisting counts at {}: {}", sample.commit, counts)
            _ = await database.store(sample.commit, sample.date, counts)


def main() -> None:
    parsed = parse_args()
    config = Config(
        nixpkgs=parsed.nixpkgs.resolve(),
        database=parsed.database.resolve(),
        expression=parsed.expression.resolve(),
        interval=parsed.interval,
        full_scan_interval=parsed.full_scan_interval,
        log_level=parsed.log_level,
    )
    configure_logging(config.log_level)
    if not config.nixpkgs.is_dir():
        raise SystemExit(f"Nixpkgs checkout does not exist: {config.nixpkgs}")
    if not config.expression.is_file():
        raise SystemExit(
            f"fetcher discovery expression does not exist: {config.expression}"
        )
    config.database.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(config))
