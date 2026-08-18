import argparse
import asyncio
import itertools
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import override

from loguru import logger
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    Task,
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
from fetcher_counter.materialization import (
    MaterializedWorktree,
    NativeCheckoutRequiredError,
    state_path_for_worker,
)

DEFAULT_INTERVAL = 50
DEFAULT_FULL_SCAN_INTERVAL = 25
DEFAULT_NATIVE_CHECKOUT_INTERVAL = 25
# Incremental materialization only beats a native checkout while the delta is
# small. Measured on a Nixpkgs worktree with 50k files: at 471 changed paths it
# took 0.43s against 0.79s for a native checkout, but at 8360 paths it took
# 3.40s against 2.57s, and at 12772 paths 5.28s against 3.90s. The crossover
# sits near 2000 paths. Sampling with `--no-first-parent` reaches well past it,
# because consecutive samples straddle merge boundaries: over 400 real adjacent
# pairs at interval 5 the median was 26 changed paths but the 99th percentile
# was 8360. The byte limit bounds peak memory for the same transitions.
DEFAULT_MAX_INCREMENTAL_PATHS = 2000
DEFAULT_MAX_INCREMENTAL_BYTES = 32 * 1024 * 1024
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
    native_checkout_interval: int = DEFAULT_NATIVE_CHECKOUT_INTERVAL
    max_incremental_paths: int = DEFAULT_MAX_INCREMENTAL_PATHS
    max_incremental_bytes: int = DEFAULT_MAX_INCREMENTAL_BYTES
    log_level: str = DEFAULT_LOG_LEVEL
    workers: int = DEFAULT_WORKERS
    reverse: bool = False
    first_parent: bool = True
    # `None` means "derive the pool from the resolved Nixpkgs path", which
    # keeps an omitted `--worktrees-dir` distinguishable from one passed
    # explicitly.
    worktrees_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class PendingSample:
    global_index: int
    sample: SampledCommit
    previous_sample: SampledCommit | None


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
        help="commit sampling interval (default: 50)",
    )
    _ = parser.add_argument(
        "--full-scan-interval",
        type=_positive_int,
        default=DEFAULT_FULL_SCAN_INTERVAL,
        help="sampled iterations between full scans (default: 25)",
    )
    _ = parser.add_argument(
        "--native-checkout-interval",
        type=_positive_int,
        default=DEFAULT_NATIVE_CHECKOUT_INTERVAL,
        help="sampled iterations between native checkouts (default: 25)",
    )
    _ = parser.add_argument(
        "--max-incremental-paths",
        type=_positive_int,
        default=DEFAULT_MAX_INCREMENTAL_PATHS,
        help="changed paths above which a native checkout is used"
        + f" (default: {DEFAULT_MAX_INCREMENTAL_PATHS})",
    )
    _ = parser.add_argument(
        "--max-incremental-bytes",
        type=_positive_int,
        default=DEFAULT_MAX_INCREMENTAL_BYTES,
        help="delta bytes above which a native checkout is used"
        + f" (default: {DEFAULT_MAX_INCREMENTAL_BYTES})",
    )
    _ = parser.add_argument(
        "--workers",
        type=_positive_int,
        default=DEFAULT_WORKERS,
        help="concurrent history shards (default: 1)",
    )
    _ = parser.add_argument(
        "--reverse",
        action="store_true",
        help="traverse sampled commits from oldest to newest",
    )
    _ = parser.add_argument(
        "--no-first-parent",
        dest="first_parent",
        action="store_false",
        help="sample all commits reachable from the history tip",
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
        native_checkout_interval=namespace.native_checkout_interval,
        max_incremental_paths=namespace.max_incremental_paths,
        max_incremental_bytes=namespace.max_incremental_bytes,
        log_level=namespace.log_level,
        workers=namespace.workers,
        reverse=namespace.reverse,
        first_parent=namespace.first_parent,
        worktrees_dir=namespace.worktrees_dir,
    )


class CommitRateColumn(ProgressColumn):
    @override
    def render(self, task: Task) -> Text:
        speed = task.finished_speed if task.finished else task.speed
        if speed is None:
            return Text("-- commits/min", style="progress.data.speed")
        if speed >= 1:
            return Text(
                f"{speed:.2f} commits/s",
                style="progress.data.speed",
            )
        return Text(
            f"{speed * 60:.2f} commits/min",
            style="progress.data.speed",
        )


def progress_columns() -> tuple[str | ProgressColumn, ...]:
    return (
        "[progress.description]{task.description}",
        MofNCompleteColumn(),
        BarColumn(),
        CommitRateColumn(),
        "[",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        "]",
    )


def configure_logging(log_level: str) -> None:
    logger.remove()
    _ = logger.configure(extra={"shard": COORDINATOR_SHARD})
    _ = logger.add(
        lambda message: console.print(Text.from_ansi(message)),
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


def indexed_for_traversal(
    commits: Sequence[SampledCommit],
    *,
    reverse: bool,
) -> list[tuple[int, SampledCommit]]:
    indexed_commits = list(enumerate(commits))
    if reverse:
        indexed_commits.reverse()
    return indexed_commits


def build_pending(
    indexed_commits: Sequence[tuple[int, SampledCommit]],
    completed: set[str],
) -> list[PendingSample]:
    """Select the samples still to process, with their previous neighbour.

    The neighbour immediately precedes the sample in traversal order, completed
    or not, so a resumed run still counts incrementally from an adjacent stored
    row. Only the first sample in the traversal has none.
    """
    return [
        PendingSample(
            global_index=global_index,
            sample=sample,
            previous_sample=(
                indexed_commits[position - 1][1] if position else None
            ),
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
    if item.previous_sample is None or item.previous_sample.commit in completed:
        return item
    return replace(item, previous_sample=None)


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
    materialized: MaterializedWorktree,
    pending: Sequence[PendingSample],
    database: FetcherDatabase,
    progress: Progress,
    tasks: ProgressTasks,
    restore_pristine: bool,
) -> None:
    worktree = materialized.path
    try:
        for position, item in enumerate(pending, start=1):
            sample = item.sample
            logger.info(
                "Processing {} ({}/{})", sample.commit, position, len(pending)
            )
            try:
                scheduled_native_checkout = (
                    item.global_index + 1
                ) % config.native_checkout_interval == 0
                requires_native_checkout = (
                    scheduled_native_checkout
                    or materialized.current_commit is None
                    or materialized.recovery_required
                )
                if scheduled_native_checkout:
                    logger.debug(
                        "Using scheduled native checkout at {} "
                        + "(sample iteration {})",
                        sample.commit,
                        item.global_index + 1,
                    )
                if requires_native_checkout:
                    with timed("Native checkout", sample.commit):
                        await materialized.native_checkout(sample.commit)
                else:
                    try:
                        with timed("Materialization", sample.commit):
                            await materialized.materialize(
                                sample.commit,
                                max_paths=config.max_incremental_paths,
                                max_bytes=config.max_incremental_bytes,
                            )
                    except NativeCheckoutRequiredError as error:
                        # A deliberate policy choice, not a failure: the
                        # transition is too large or uses a tree mode the
                        # materializer does not reproduce.
                        logger.debug(
                            "Using native checkout from {} to {}: {}",
                            materialized.current_commit,
                            sample.commit,
                            error,
                        )
                        with timed("Chosen native checkout", sample.commit):
                            await materialized.native_checkout(sample.commit)
                    except Exception as error:  # noqa: BLE001
                        logger.warning(
                            "Incremental materialization from {} to {} failed; "
                            + "using native checkout: {}",
                            materialized.current_commit,
                            sample.commit,
                            error,
                        )
                        with timed("Fallback native checkout", sample.commit):
                            await materialized.native_checkout(sample.commit)
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
                if item.previous_sample is not None and not scheduled_full_scan:
                    previous_counts = await database.counts_for_commit(
                        item.previous_sample.commit,
                        fetchers,
                    )
                    if previous_counts is not None:
                        try:
                            with timed("Incremental count", sample.commit):
                                counts = await update_fetcher_counts(
                                    worktree,
                                    item.previous_sample.commit,
                                    sample.commit,
                                    previous_counts,
                                )
                        except IncrementalCountError as error:
                            logger.warning(
                                "Incremental count at {} failed; "
                                + "using full scan: {}",
                                sample.commit,
                                error,
                            )
                        else:
                            logger.debug(
                                "Used counts from adjacent commit {} for {}",
                                item.previous_sample.commit,
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
                if tasks.total != tasks.shard:
                    progress.advance(tasks.total)
    finally:
        if restore_pristine:
            with timed(
                "Final native checkout",
                materialized.current_commit or "none",
            ):
                await materialized.restore_pristine()


async def _run_labelled_shard(
    config: Config,
    *,
    shard: ActiveShard,
    materialized: MaterializedWorktree,
    database: FetcherDatabase,
    progress: Progress,
    tasks: ProgressTasks,
) -> None:
    with logger.contextualize(shard=f"shard {shard.index}"):
        logger.info(
            "Processing {} pending samples in {}",
            len(shard.pending),
            materialized.path,
        )
        await run_shard(
            config,
            materialized=materialized,
            pending=shard.pending,
            database=database,
            progress=progress,
            tasks=tasks,
            restore_pristine=True,
        )


async def run_single_worker(config: Config) -> None:
    """Process the whole history sequentially in `config.nixpkgs` itself.

    No worktree pool is created or locked, so the documented behaviour of
    mutating the supplied checkout in place stays intact.
    """
    async with FetcherDatabase(config.database) as database:
        completed = await database.completed_commits()
        commits = await sampled_commits(
            config.nixpkgs,
            interval=config.interval,
            first_parent=config.first_parent,
            completed=completed,
        )
        pending = build_pending(
            indexed_for_traversal(commits, reverse=config.reverse), completed
        )
        logger.opt(lazy=True).debug(
            "Skipping completed commit hashes: {}",
            lambda: sorted(completed),
        )
        logger.info(
            "Found {} sampled commits; {} remain",
            len(commits),
            len(pending),
        )
        with Progress(*progress_columns(), console=console) as progress:
            task = progress.add_task("Total", total=len(pending))
            await run_shard(
                config,
                materialized=MaterializedWorktree(
                    repository=config.nixpkgs,
                    path=config.nixpkgs,
                    checkout_function=checkout,
                ),
                pending=pending,
                database=database,
                progress=progress,
                tasks=ProgressTasks(total=task, shard=task),
                restore_pristine=False,
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
        async with FetcherDatabase(config.database) as database:
            completed = await database.completed_commits()
            commits = await sampled_commits(
                config.nixpkgs,
                interval=config.interval,
                first_parent=config.first_parent,
                completed=completed,
            )
            logger.opt(lazy=True).debug(
                "Skipping completed commit hashes: {}",
                lambda: sorted(completed),
            )
            pending = build_pending(
                indexed_for_traversal(commits, reverse=config.reverse), completed
            )
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
                            materialized=MaterializedWorktree(
                                repository=config.nixpkgs,
                                path=worktrees[shard.index],
                                state_path=state_path_for_worker(
                                    worktrees_dir,
                                    shard.index,
                                ),
                                checkout_function=checkout,
                            ),
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
    if config.native_checkout_interval < 1:
        raise ValueError("native checkout interval must be positive")
    if config.max_incremental_paths < 1:
        raise ValueError("maximum incremental paths must be positive")
    if config.max_incremental_bytes < 1:
        raise ValueError("maximum incremental bytes must be positive")
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
    # Adjust the parsed configuration rather than rebuilding one field by
    # field. Re-listing every field meant a newly added option was parsed and
    # then silently dropped here, so the program ran with its default while
    # accepting the flag without complaint.
    config = replace(
        parsed,
        nixpkgs=nixpkgs,
        database=parsed.database.resolve(),
        expression=parsed.expression.resolve(),
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
