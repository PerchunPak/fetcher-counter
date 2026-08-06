import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from fetcher_counter.counting import count_fetchers
from fetcher_counter.database import FetcherDatabase
from fetcher_counter.discovery import discover_fetchers
from fetcher_counter.history import checkout, sampled_commits

DEFAULT_INTERVAL = 50


@dataclass(frozen=True, slots=True)
class Config:
    nixpkgs: Path
    database: Path
    expression: Path
    interval: int = DEFAULT_INTERVAL


def default_expression() -> Path:
    packaged = Path(__file__).with_name("get-fetchers.nix")
    if packaged.is_file():
        return packaged
    return Path(__file__).resolve().parents[2] / "get-fetchers.nix"


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
    namespace = parser.parse_args(arguments)
    return Config(
        nixpkgs=namespace.nixpkgs,
        database=namespace.database,
        expression=namespace.expression,
        interval=namespace.interval,
    )


async def run(config: Config) -> None:
    logger.debug("Starting fetcher counter with configuration {}", config)
    commits = await sampled_commits(config.nixpkgs, interval=config.interval)
    async with FetcherDatabase(config.database) as database:
        completed = await database.completed_commits()
        pending = [sample for sample in commits if sample.commit not in completed]
        logger.debug("Skipping completed commit hashes: {}", sorted(completed))
        logger.info(
            "Found {} sampled commits; {} remain",
            len(commits),
            len(pending),
        )

        for position, sample in enumerate(pending, start=1):
            logger.info(
                "Processing {} ({}/{})",
                sample.commit,
                position,
                len(pending),
            )
            await checkout(config.nixpkgs, sample.commit)
            fetchers = await discover_fetchers(
                config.nixpkgs,
                config.expression,
                commit=sample.commit,
            )
            logger.debug("Active fetchers at {}: {}", sample.commit, fetchers)
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
    )
    if not config.nixpkgs.is_dir():
        raise SystemExit(f"Nixpkgs checkout does not exist: {config.nixpkgs}")
    if not config.expression.is_file():
        raise SystemExit(
            f"fetcher discovery expression does not exist: {config.expression}"
        )
    config.database.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(config))
