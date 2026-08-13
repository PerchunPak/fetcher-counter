import argparse
import asyncio
import itertools
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

from loguru import logger
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    TaskID,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

from fetcher_counter.counting import (
    IncrementalCountError,
    count_fetchers,
    update_fetcher_counts,
)
from fetcher_counter.database import FetcherDatabase
from fetcher_counter.discovery import FetcherDiscoveryError, discover_fetchers
from fetcher_counter.history import (
    SampledCommit,
    WorktreeRequest,
    checkout,
    provision_worktrees,
    sampled_commits,
    worktree_pool_lock,
)

DEFAULT_INTERVAL = 50
DEFAULT_FULL_SCAN_INTERVAL = 25
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_WORKERS = 1
LOG_LEVELS = ("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL")
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    + "<level>{level: <8}</level> | "
    + "[<cyan>{extra[shard]}</cyan>] "
    + "<level>{message}</level>"
)
COORDINATOR_SHARD = "main"
console = Console(stderr=True)


@dataclass(frozen=True, slots=True)
class Config:
    nixpkgs: Path
    database: Path
    expression: Path
    interval: int = DEFAULT_INTERVAL
    full_scan_interval: int = DEFAULT_FULL_SCAN_INTERVAL
    log_level: str = DEFAULT_LOG_LEVEL
    workers: int = DEFAULT_WORKERS
    # `None` means "derive the pool from the resolved Nixpkgs path", which
    # keeps an omitted `--worktrees-dir` distinguishable from one passed
    # explicitly.
    worktrees_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class PendingSample:
    global_index: int
    sample: SampledCommit
    newer_sample: SampledCommit | None


@dataclass(frozen=True, slots=True)
class ActiveShard:
    index: int
    pending: list[PendingSample]


@dataclass(frozen=True, slots=True)
class ProgressTasks:
    total: TaskID
    shard: TaskID


class ShardError(RuntimeError):
    def __init__(self, shard_index: int, cause: BaseException) -> None:
        self.shard_index: int = shard_index
        self.cause: BaseException = cause
        super().__init__(f"shard {shard_index} failed: {cause}")


def default_expression() -> Path:
    packaged = Path(__file__).with_name("get-fetchers.nix")
    if packaged.is_file():
        return packaged
    return Path(__file__).resolve().parents[2] / "get-fetchers.nix"


def default_worktrees_dir(nixpkgs: Path) -> Path:
    """Derive the worker pool path from an already-resolved Nixpkgs path.

    Resolving first matters: a symlinked `--nixpkgs` argument must not get a
    different pool than the checkout it points at.
    """
    return nixpkgs.parent / f".{nixpkgs.name}-fetcher-counter-worktrees"


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
        "--workers",
        type=_positive_int,
        default=DEFAULT_WORKERS,
        help="concurrent history shards (default: 1)",
    )
    _ = parser.add_argument(
        "--worktrees-dir",
        type=Path,
        default=None,
        help="persistent worker worktree pool (default: next to --nixpkgs)",
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
        workers=namespace.workers,
        worktrees_dir=namespace.worktrees_dir,
    )


def progress_columns() -> tuple[str | ProgressColumn, ...]:
    return (
        "[progress.description]{task.description}",
        MofNCompleteColumn(),
        BarColumn(),
        "[",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        "]",
    )


def configure_logging(log_level: str) -> None:
    logger.remove()
    _ = logger.configure(extra={"shard": COORDINATOR_SHARD})
    _ = logger.add(
        lambda message: console.print(Text.from_ansi(message), end=""),
        level=log_level,
        format=LOG_FORMAT,
        colorize=console.is_terminal,
    )


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


def build_pending(
    indexed_commits: Sequence[tuple[int, SampledCommit]],
    completed: set[str],
) -> list[PendingSample]:
    """Select the samples still to process, with their newer neighbour.

    The neighbour is the immediately newer sample, completed or not, so a
    resumed run still counts incrementally from an adjacent stored row. Only
    the newest sample of the whole history has none.
    """
    return [
        PendingSample(
            global_index=global_index,
            sample=sample,
            newer_sample=(indexed_commits[position - 1][1] if position else None),
        )
        for position, (global_index, sample) in enumerate(indexed_commits)
        if sample.commit not in completed
    ]


def _shard_boundary(item: PendingSample, completed: set[str]) -> PendingSample:
    """Drop a neighbour that another shard is about to process.

    A stored neighbour is safe to read: its row exists before the run starts
    and no shard ever rewrites it. A neighbour that is itself pending belongs
    to the previous shard, so reusing it would make
    full-scan-versus-incremental depend on which shard got there first.
    """
    if item.newer_sample is None or item.newer_sample.commit in completed:
        return item
    return replace(item, newer_sample=None)


def split_pending(
    pending: Sequence[PendingSample],
    workers: int,
    completed: set[str],
) -> list[ActiveShard]:
    """Split the pending samples into contiguous, evenly sized shards.

    Splitting the pending samples rather than the whole history is what keeps
    every worker busy on a resumed run, where most of the history is already
    stored. The cost is that boundaries move between runs, so a different
    sample forfeits its neighbour each time; correctness does not depend on
    them, because stored rows are keyed by commit.
    """
    total = len(pending)
    bounds = [total * index // workers for index in range(workers + 1)]
    shards: list[ActiveShard] = []
    for index, (start, end) in enumerate(itertools.pairwise(bounds)):
        items = list(pending[start:end])
        if items:
            items[0] = _shard_boundary(items[0], completed)
        shards.append(ActiveShard(index=index, pending=items))
    return shards


async def run_shard(
    config: Config,
    *,
    worktree: Path,
    pending: Sequence[PendingSample],
    database: FetcherDatabase,
    progress: Progress,
    tasks: ProgressTasks,
) -> None:
    for position, item in enumerate(pending, start=1):
        sample = item.sample
        logger.info("Processing {} ({}/{})", sample.commit, position, len(pending))
        try:
            with timed("Checkout", sample.commit):
                await checkout(worktree, sample.commit)
            try:
                with timed("Discovery", sample.commit):
                    fetchers = await discover_fetchers(
                        worktree,
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
                item.global_index + 1
            ) % config.full_scan_interval == 0
            if scheduled_full_scan:
                logger.debug(
                    "Using scheduled full scan at {} (sample iteration {})",
                    sample.commit,
                    item.global_index + 1,
                )
            if item.newer_sample is not None and not scheduled_full_scan:
                newer_counts = await database.counts_for_commit(
                    item.newer_sample.commit,
                    fetchers,
                )
                if newer_counts is not None:
                    try:
                        with timed("Incremental count", sample.commit):
                            counts = await update_fetcher_counts(
                                worktree,
                                item.newer_sample.commit,
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
                            item.newer_sample.commit,
                            sample.commit,
                        )

            if counts is None:
                logger.debug("Using full fetcher scan at {}", sample.commit)
                with timed("Full scan", sample.commit):
                    counts = await count_fetchers(
                        worktree,
                        sample.commit,
                        fetchers,
                    )
            logger.debug("Persisting counts at {}: {}", sample.commit, counts)
            _ = await database.store(sample.commit, sample.date, counts)
        finally:
            progress.advance(tasks.shard)
            progress.advance(tasks.total)


async def _run_labelled_shard(
    config: Config,
    *,
    shard: ActiveShard,
    worktree: Path,
    database: FetcherDatabase,
    progress: Progress,
    tasks: ProgressTasks,
) -> None:
    with logger.contextualize(shard=f"shard {shard.index}"):
        logger.info(
            "Processing {} pending samples in {}",
            len(shard.pending),
            worktree,
        )
        await run_shard(
            config,
            worktree=worktree,
            pending=shard.pending,
            database=database,
            progress=progress,
            tasks=tasks,
        )


async def run_single_worker(config: Config) -> None:
    """Process the whole history sequentially in `config.nixpkgs` itself.

    No worktree pool is created or locked, so the documented behaviour of
    mutating the supplied checkout in place stays intact.
    """
    commits = await sampled_commits(config.nixpkgs, interval=config.interval)
    async with FetcherDatabase(config.database) as database:
        completed = await database.completed_commits()
        pending = build_pending(list(enumerate(commits)), completed)
        logger.debug("Skipping completed commit hashes: {}", sorted(completed))
        logger.info(
            "Found {} sampled commits; {} remain",
            len(commits),
            len(pending),
        )
        with Progress(*progress_columns(), console=console) as progress:
            tasks = ProgressTasks(
                total=progress.add_task("Total", total=len(pending)),
                shard=progress.add_task("Shard 0", total=len(pending)),
            )
            await run_shard(
                config,
                worktree=config.nixpkgs,
                pending=pending,
                database=database,
                progress=progress,
                tasks=tasks,
            )


def _shard_outcome_failures(
    outcomes: Sequence[tuple[ActiveShard, BaseException | None]],
) -> list[ShardError]:
    cancelled = [
        shard.index
        for shard, result in outcomes
        if isinstance(result, asyncio.CancelledError)
    ]
    if cancelled:
        logger.error("Cancelled shards: {}", cancelled)
        raise asyncio.CancelledError(
            f"shards were cancelled: {', '.join(map(str, cancelled))}"
        )
    return [
        ShardError(shard.index, result)
        for shard, result in outcomes
        if isinstance(result, Exception)
    ]


async def run_parallel(config: Config) -> None:
    """Process contiguous shards of the history concurrently.

    The pool lock is taken before anything shared is touched, because
    sampling writes the saved history tip and opening the database can
    migrate it. Every shard shares this process' single database
    connection, so its `asyncio.Lock` keeps all writes serialized.
    """
    worktrees_dir = config.worktrees_dir
    if worktrees_dir is None:
        worktrees_dir = default_worktrees_dir(config.nixpkgs.resolve())

    with worktree_pool_lock(worktrees_dir):
        commits = await sampled_commits(config.nixpkgs, interval=config.interval)
        async with FetcherDatabase(config.database) as database:
            completed = await database.completed_commits()
            logger.debug("Skipping completed commit hashes: {}", sorted(completed))
            pending = build_pending(list(enumerate(commits)), completed)
            active_shards = [
                shard
                for shard in split_pending(pending, config.workers, completed)
                if shard.pending
            ]
            total_pending = sum(len(shard.pending) for shard in active_shards)
            logger.info(
                "Found {} sampled commits; {} remain across {} of {} shards",
                len(commits),
                total_pending,
                len(active_shards),
                config.workers,
            )
            if not active_shards:
                return

            worktrees = await provision_worktrees(
                repository=config.nixpkgs,
                pool_dir=worktrees_dir,
                requests=[
                    WorktreeRequest(
                        index=shard.index,
                        initial_commit=shard.pending[0].sample.commit,
                    )
                    for shard in active_shards
                ],
            )
            with Progress(*progress_columns(), console=console) as progress:
                total_task = progress.add_task("Total", total=total_pending)
                shard_tasks = {
                    shard.index: progress.add_task(
                        f"Shard {shard.index}",
                        total=len(shard.pending),
                    )
                    for shard in active_shards
                }
                results = await asyncio.gather(
                    *(
                        _run_labelled_shard(
                            config,
                            shard=shard,
                            worktree=worktrees[shard.index],
                            database=database,
                            progress=progress,
                            tasks=ProgressTasks(
                                total=total_task,
                                shard=shard_tasks[shard.index],
                            ),
                        )
                        for shard in active_shards
                    ),
                    return_exceptions=True,
                )

    failures = _shard_outcome_failures(
        list(zip(active_shards, results, strict=True))
    )
    if failures:
        raise ExceptionGroup("one or more shards failed", failures)


async def run(config: Config) -> None:
    if config.full_scan_interval < 1:
        raise ValueError("full scan interval must be positive")
    if config.workers < 1:
        raise ValueError("workers must be positive")
    logger.debug("Starting fetcher counter with configuration {}", config)
    if config.workers == 1:
        await run_single_worker(config)
        return
    await run_parallel(config)


def main() -> None:
    parsed = parse_args()
    nixpkgs = parsed.nixpkgs.resolve()
    config = Config(
        nixpkgs=nixpkgs,
        database=parsed.database.resolve(),
        expression=parsed.expression.resolve(),
        interval=parsed.interval,
        full_scan_interval=parsed.full_scan_interval,
        log_level=parsed.log_level,
        workers=parsed.workers,
        worktrees_dir=(
            parsed.worktrees_dir.resolve()
            if parsed.worktrees_dir is not None
            else default_worktrees_dir(nixpkgs)
        ),
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
