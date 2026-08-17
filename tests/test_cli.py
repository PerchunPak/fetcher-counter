import asyncio
import sqlite3
import sys
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import ClassVar, Self

import pytest
from loguru import logger
from rich.progress import ProgressSample, Task, TaskID
from rich.text import Text

from fetcher_counter import cli
from fetcher_counter.cli import Config, parse_args, run
from fetcher_counter.counting import IncrementalCountError
from fetcher_counter.database import FetcherDatabase
from fetcher_counter.discovery import FetcherDiscoveryError
from fetcher_counter.history import (
    SampledCommit,
    WorktreePoolLockedError,
    WorktreeRequest,
    checkout as history_checkout,
    worktree_pool_lock,
)


@pytest.fixture(autouse=True)
def fake_materializer(monkeypatch: pytest.MonkeyPatch) -> None:
    class CheckoutBackedMaterializer:
        def __init__(
            self,
            repository: Path,
            path: Path,
            state_path: Path | None = None,
            checkout_function: Callable[
                [Path, str], Awaitable[None]
            ] = history_checkout,
        ) -> None:
            _ = repository, state_path
            self.path: Path = path
            self.checkout_function: Callable[[Path, str], Awaitable[None]] = (
                checkout_function
            )
            self.current_commit: str | None = None
            self.native_commit: str | None = None
            self.recovery_required: bool = False

        async def native_checkout(self, commit: str) -> None:
            await self.checkout_function(self.path, commit)
            self.current_commit = commit
            self.native_commit = commit

        async def materialize(self, commit: str) -> None:
            await self.checkout_function(self.path, commit)
            self.current_commit = commit
            self.native_commit = commit

        async def restore_pristine(self) -> None:
            return None

    monkeypatch.setattr(cli, "MaterializedWorktree", CheckoutBackedMaterializer)


def test_parse_args_uses_project_defaults() -> None:
    config = parse_args([])

    assert config.nixpkgs == Path("nixpkgs")
    assert config.database == Path("data/fetchers.sqlite3")
    assert config.interval == 50
    assert config.full_scan_interval == 25
    assert config.native_checkout_interval == 25
    assert config.log_level == "INFO"
    assert config.workers == 1
    assert config.reverse is False
    assert config.first_parent is True
    assert config.worktrees_dir is None


def test_parse_args_accepts_full_scan_interval() -> None:
    config = parse_args(["--full-scan-interval", "7"])

    assert config.full_scan_interval == 7


def test_parse_args_accepts_native_checkout_interval() -> None:
    config = parse_args(["--native-checkout-interval", "7"])

    assert config.native_checkout_interval == 7
    assert config.full_scan_interval == 25


def test_parse_args_accepts_reverse() -> None:
    config = parse_args(["--reverse"])

    assert config.reverse is True


def test_parse_args_accepts_no_first_parent() -> None:
    config = parse_args(["--no-first-parent"])

    assert config.first_parent is False


def test_parse_args_rejects_nonpositive_full_scan_interval() -> None:
    with pytest.raises(SystemExit):
        _ = parse_args(["--full-scan-interval", "0"])


def test_parse_args_rejects_nonpositive_native_checkout_interval() -> None:
    with pytest.raises(SystemExit):
        _ = parse_args(["--native-checkout-interval", "0"])


def test_parse_args_accepts_case_insensitive_log_level() -> None:
    config = parse_args(["--log-level", "debug"])

    assert config.log_level == "DEBUG"


@pytest.mark.asyncio
async def test_run_rejects_nonpositive_full_scan_interval(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="full scan interval must be positive"):
        await run(
            Config(
                nixpkgs=tmp_path / "nixpkgs",
                database=tmp_path / "fetchers.sqlite3",
                expression=tmp_path / "get-fetchers.nix",
                full_scan_interval=0,
            )
        )


@pytest.mark.asyncio
async def test_run_rejects_nonpositive_native_checkout_interval(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError, match="native checkout interval must be positive"
    ):
        await run(
            Config(
                nixpkgs=tmp_path / "nixpkgs",
                database=tmp_path / "fetchers.sqlite3",
                expression=tmp_path / "get-fetchers.nix",
                native_checkout_interval=0,
            )
        )


def test_configure_logging_replaces_default_sink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    printed: list[tuple[Text, str]] = []

    class FakeConsole:
        is_terminal: bool = True

        def print(self, message: Text, *, end: str = "\n") -> None:
            printed.append((message, end))

    class FakeLogger:
        def remove(self) -> None:
            calls.append(("remove",))

        def configure(self, *, extra: dict[str, str]) -> None:
            calls.append(("configure", extra))

        def add(
            self,
            sink: Callable[[str], None],
            *,
            level: str,
            format: str,
            colorize: bool,
        ) -> int:
            calls.append(("add", level, format, colorize))
            sink("\x1b[31mwarning\x1b[0m\n")
            return 1

    monkeypatch.setattr(cli, "console", FakeConsole())
    monkeypatch.setattr(cli, "logger", FakeLogger())

    cli.configure_logging("WARNING")

    assert calls == [
        ("remove",),
        ("configure", {"shard": "main"}),
        ("add", "WARNING", cli.LOG_FORMAT, True),
    ]
    assert printed[0][0].plain == "warning"
    assert printed[0][1] == "\n"


def progress_task(*, speed: float | None, finished: bool = False) -> Task:
    task = Task(
        id=TaskID(0),
        description="Total",
        total=10,
        completed=2,
        _get_time=lambda: 1.0,
    )
    if finished:
        task.finished_time = 1.0
        task.finished_speed = speed
    elif speed is not None:
        task.start_time = 0.0
        task._progress.append(ProgressSample(0.0, 0.0))
        task._progress.append(ProgressSample(1.0, speed))
    return task


def test_commit_rate_column_shows_placeholder_without_speed() -> None:
    rendered = cli.CommitRateColumn().render(progress_task(speed=None))

    assert rendered.plain == "-- commits/min"


def test_commit_rate_column_shows_commits_per_second() -> None:
    rendered = cli.CommitRateColumn().render(progress_task(speed=2.5))

    assert rendered.plain == "2.50 commits/s"


def test_commit_rate_column_shows_commits_per_minute() -> None:
    rendered = cli.CommitRateColumn().render(progress_task(speed=0.25))

    assert rendered.plain == "15.00 commits/min"


def test_commit_rate_column_keeps_finished_speed() -> None:
    rendered = cli.CommitRateColumn().render(
        progress_task(speed=1.5, finished=True)
    )

    assert rendered.plain == "1.50 commits/s"


def test_progress_columns_match_nupd_layout() -> None:
    columns = cli.progress_columns()

    assert columns[0] == "[progress.description]{task.description}"
    assert type(columns[1]).__name__ == "MofNCompleteColumn"
    assert type(columns[2]).__name__ == "BarColumn"
    assert type(columns[3]).__name__ == "CommitRateColumn"
    assert columns[4] == "["
    assert type(columns[5]).__name__ == "TimeElapsedColumn"
    assert type(columns[6]).__name__ == "TimeRemainingColumn"
    assert columns[7] == "]"


def test_log_format_labels_coordinator_and_shard_messages() -> None:
    messages: list[str] = []
    _ = logger.configure(extra={"shard": cli.COORDINATOR_SHARD})
    sink_id = logger.add(
        messages.append,
        level="INFO",
        format=cli.LOG_FORMAT,
        colorize=False,
    )
    try:
        logger.info("coordinator message")
        with logger.contextualize(shard="shard 3"):
            logger.info("shard message")
    finally:
        logger.remove(sink_id)

    assert "[main] coordinator message" in messages[0]
    assert "[shard 3] shard message" in messages[1]


@pytest.mark.asyncio
async def test_run_skips_completed_commits_and_persists_active_fetchers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    samples = [
        SampledCommit("first", "2026-01-01T00:00:00Z"),
        SampledCommit("second", "2026-01-02T00:00:00Z"),
        SampledCommit("third", "2026-01-03T00:00:00Z"),
    ]
    database_path = tmp_path / "fetchers.sqlite3"
    async with FetcherDatabase(database_path) as database:
        _ = await database.store("first", samples[0].date, {"oldFetcher": 1})

    checkouts: list[str] = []

    async def fake_samples(
        _repository: Path,
        *,
        interval: int,
        first_parent: bool,
        completed: set[str] | None = None,
    ) -> list[SampledCommit]:
        _ = completed
        assert first_parent is False
        assert interval == 50
        assert completed == {"first"}
        return samples

    async def fake_checkout(_repository: Path, commit: str) -> None:
        checkouts.append(commit)

    async def fake_discovery(
        _nixpkgs: Path,
        _expression: Path,
        *,
        commit: str,
    ) -> list[str]:
        return ["fetchurl"] if commit == "second" else ["fetchzip"]

    async def fake_counts(
        _repository: Path,
        commit: str,
        fetchers: list[str],
    ) -> dict[str, int]:
        return {fetchers[0]: 2 if commit == "second" else 3}

    monkeypatch.setattr(cli, "sampled_commits", fake_samples)
    monkeypatch.setattr(cli, "checkout", fake_checkout)
    monkeypatch.setattr(cli, "discover_fetchers", fake_discovery)
    monkeypatch.setattr(cli, "count_fetchers", fake_counts)

    messages: list[str] = []
    sink_id = logger.add(messages.append, level="DEBUG", format="{message}")
    try:
        await run(
            Config(
                nixpkgs=tmp_path / "nixpkgs",
                database=database_path,
                expression=tmp_path / "get-fetchers.nix",
                first_parent=False,
            )
        )
    finally:
        logger.remove(sink_id)

    connection = sqlite3.connect(database_path)
    rows = connection.execute(
        """SELECT "commit", "oldFetcher", "fetchurl", "fetchzip"
        FROM "fetchers" ORDER BY "date"
        """
    ).fetchall()
    connection.close()

    assert checkouts == ["second", "third"]
    assert any(
        "Skipping completed commit hashes: ['first']" in line for line in messages
    )
    assert any(
        "Active fetchers at second: ['fetchurl']" in line for line in messages
    )
    assert any(
        "Persisting counts at third: {'fetchzip': 3}" in line for line in messages
    )
    assert rows == [
        ("first", 1, None, None),
        ("second", None, 2, None),
        ("third", None, None, 3),
    ]


@pytest.mark.asyncio
async def test_run_persists_failed_evaluation_and_continues(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    samples = [
        SampledCommit("failed", "2026-01-01T00:00:00Z"),
        SampledCommit("successful", "2026-01-02T00:00:00Z"),
    ]
    counted: list[str] = []

    async def fake_samples(
        _repository: Path,
        *,
        interval: int,
        first_parent: bool,
        completed: set[str] | None = None,
    ) -> list[SampledCommit]:
        _ = completed
        assert first_parent is True
        assert interval == 50
        return samples

    async def fake_checkout(_repository: Path, _commit: str) -> None:
        return None

    async def fake_discovery(
        _nixpkgs: Path,
        _expression: Path,
        *,
        commit: str,
    ) -> list[str]:
        if commit == "failed":
            raise FetcherDiscoveryError("evaluation failed")
        return ["fetchurl"]

    async def fake_counts(
        _repository: Path,
        commit: str,
        fetchers: list[str],
    ) -> dict[str, int]:
        counted.append(commit)
        return {fetchers[0]: 4}

    monkeypatch.setattr(cli, "sampled_commits", fake_samples)
    monkeypatch.setattr(cli, "checkout", fake_checkout)
    monkeypatch.setattr(cli, "discover_fetchers", fake_discovery)
    monkeypatch.setattr(cli, "count_fetchers", fake_counts)
    database_path = tmp_path / "fetchers.sqlite3"

    await run(
        Config(
            nixpkgs=tmp_path / "nixpkgs",
            database=database_path,
            expression=tmp_path / "get-fetchers.nix",
        )
    )

    connection = sqlite3.connect(database_path)
    rows = connection.execute(
        'SELECT "commit", "is_skipped", "fetchurl" FROM "fetchers" ORDER BY "date"'
    ).fetchall()
    connection.close()

    assert counted == ["successful"]
    assert rows == [("failed", 1, None), ("successful", 0, 4)]


@pytest.mark.asyncio
async def test_run_uses_full_baseline_then_adjacent_incremental_counts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    samples = [
        SampledCommit("newest", "2026-01-03T00:00:00Z"),
        SampledCommit("older", "2026-01-02T00:00:00Z"),
        SampledCommit("oldest", "2026-01-01T00:00:00Z"),
    ]
    full_scans: list[str] = []
    updates: list[tuple[str, str, dict[str, int]]] = []

    async def fake_samples(
        _repository: Path,
        *,
        interval: int,
        first_parent: bool,
        completed: set[str] | None = None,
    ) -> list[SampledCommit]:
        _ = completed
        assert first_parent is True
        assert interval == 50
        return samples

    async def fake_checkout(_repository: Path, _commit: str) -> None:
        return None

    async def fake_discovery(
        _nixpkgs: Path,
        _expression: Path,
        *,
        commit: str,
    ) -> list[str]:
        assert commit in {"newest", "older", "oldest"}
        return ["fetchurl"]

    async def fake_counts(
        _repository: Path,
        commit: str,
        _fetchers: list[str],
    ) -> dict[str, int]:
        full_scans.append(commit)
        return {"fetchurl": 10}

    async def fake_update(
        _repository: Path,
        newer_commit: str,
        older_commit: str,
        newer_counts: dict[str, int],
    ) -> dict[str, int]:
        updates.append((newer_commit, older_commit, newer_counts))
        return {"fetchurl": newer_counts["fetchurl"] - 1}

    monkeypatch.setattr(cli, "sampled_commits", fake_samples)
    monkeypatch.setattr(cli, "checkout", fake_checkout)
    monkeypatch.setattr(cli, "discover_fetchers", fake_discovery)
    monkeypatch.setattr(cli, "count_fetchers", fake_counts)
    monkeypatch.setattr(cli, "update_fetcher_counts", fake_update)
    database_path = tmp_path / "fetchers.sqlite3"

    await run(
        Config(
            nixpkgs=tmp_path / "nixpkgs",
            database=database_path,
            expression=tmp_path / "get-fetchers.nix",
        )
    )

    connection = sqlite3.connect(database_path)
    rows = connection.execute(
        'SELECT "commit", "fetchurl" FROM "fetchers" ORDER BY "date" DESC'
    ).fetchall()
    connection.close()

    assert full_scans == ["newest"]
    assert updates == [
        ("newest", "older", {"fetchurl": 10}),
        ("older", "oldest", {"fetchurl": 9}),
    ]
    assert rows == [("newest", 10), ("older", 9), ("oldest", 8)]


@pytest.mark.asyncio
async def test_run_forces_full_scan_at_configured_interval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    samples = [
        SampledCommit(f"commit-{position}", f"2026-01-{position:02d}T00:00:00Z")
        for position in range(1, 7)
    ]
    full_scans: list[str] = []
    updates: list[tuple[str, str]] = []

    async def fake_samples(
        _repository: Path,
        *,
        interval: int,
        first_parent: bool,
        completed: set[str] | None = None,
    ) -> list[SampledCommit]:
        _ = completed
        assert first_parent is True
        assert interval == 50
        return samples

    async def fake_checkout(_repository: Path, _commit: str) -> None:
        return None

    async def fake_discovery(
        _nixpkgs: Path,
        _expression: Path,
        *,
        commit: str,
    ) -> list[str]:
        assert commit.startswith("commit-")
        return ["fetchurl"]

    async def fake_counts(
        _repository: Path,
        commit: str,
        _fetchers: list[str],
    ) -> dict[str, int]:
        full_scans.append(commit)
        return {"fetchurl": 10}

    async def fake_update(
        _repository: Path,
        newer_commit: str,
        older_commit: str,
        newer_counts: dict[str, int],
    ) -> dict[str, int]:
        updates.append((newer_commit, older_commit))
        return newer_counts

    monkeypatch.setattr(cli, "sampled_commits", fake_samples)
    monkeypatch.setattr(cli, "checkout", fake_checkout)
    monkeypatch.setattr(cli, "discover_fetchers", fake_discovery)
    monkeypatch.setattr(cli, "count_fetchers", fake_counts)
    monkeypatch.setattr(cli, "update_fetcher_counts", fake_update)

    await run(
        Config(
            nixpkgs=tmp_path / "nixpkgs",
            database=tmp_path / "fetchers.sqlite3",
            expression=tmp_path / "get-fetchers.nix",
            full_scan_interval=5,
        )
    )

    assert full_scans == ["commit-1", "commit-5"]
    assert ("commit-3", "commit-4") in updates
    assert ("commit-4", "commit-5") not in updates
    assert ("commit-5", "commit-6") in updates
    assert len(updates) == 4


@pytest.mark.asyncio
async def test_run_uses_history_position_for_scheduled_full_scan_after_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    samples = [
        SampledCommit(f"commit-{position}", f"2026-01-{position:02d}T00:00:00Z")
        for position in range(1, 26)
    ]
    database_path = tmp_path / "fetchers.sqlite3"
    async with FetcherDatabase(database_path) as database:
        for sample in samples[:-1]:
            assert await database.store(
                sample.commit, sample.date, {"fetchurl": 10}
            )

    full_scans: list[str] = []

    async def fake_samples(
        _repository: Path,
        *,
        interval: int,
        first_parent: bool,
        completed: set[str] | None = None,
    ) -> list[SampledCommit]:
        _ = completed
        assert first_parent is True
        assert interval == 50
        return samples

    async def fake_checkout(_repository: Path, commit: str) -> None:
        assert commit == "commit-25"

    async def fake_discovery(
        _nixpkgs: Path,
        _expression: Path,
        *,
        commit: str,
    ) -> list[str]:
        assert commit == "commit-25"
        return ["fetchurl"]

    async def fake_counts(
        _repository: Path,
        commit: str,
        _fetchers: list[str],
    ) -> dict[str, int]:
        full_scans.append(commit)
        return {"fetchurl": 9}

    async def fail_update(
        _repository: Path,
        _newer_commit: str,
        _older_commit: str,
        _newer_counts: dict[str, int],
    ) -> dict[str, int]:
        pytest.fail("the scheduled full scan should not use incremental counts")

    monkeypatch.setattr(cli, "sampled_commits", fake_samples)
    monkeypatch.setattr(cli, "checkout", fake_checkout)
    monkeypatch.setattr(cli, "discover_fetchers", fake_discovery)
    monkeypatch.setattr(cli, "count_fetchers", fake_counts)
    monkeypatch.setattr(cli, "update_fetcher_counts", fail_update)

    await run(
        Config(
            nixpkgs=tmp_path / "nixpkgs",
            database=database_path,
            expression=tmp_path / "get-fetchers.nix",
        )
    )

    assert full_scans == ["commit-25"]


@pytest.mark.asyncio
async def test_run_resumes_from_adjacent_completed_row(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    samples = [
        SampledCommit("newest", "2026-01-02T00:00:00Z"),
        SampledCommit("older", "2026-01-01T00:00:00Z"),
    ]
    database_path = tmp_path / "fetchers.sqlite3"
    async with FetcherDatabase(database_path) as database:
        assert await database.store("newest", samples[0].date, {"fetchurl": 5})

    async def fake_samples(
        _repository: Path,
        *,
        interval: int,
        first_parent: bool,
        completed: set[str] | None = None,
    ) -> list[SampledCommit]:
        _ = completed
        assert first_parent is True
        assert interval == 50
        return samples

    async def fake_checkout(_repository: Path, _commit: str) -> None:
        return None

    async def fake_discovery(
        _nixpkgs: Path,
        _expression: Path,
        *,
        commit: str,
    ) -> list[str]:
        assert commit == "older"
        return ["fetchurl"]

    async def fail_full_scan(
        _repository: Path,
        _commit: str,
        _fetchers: list[str],
    ) -> dict[str, int]:
        pytest.fail("an adjacent completed row should avoid a full scan")

    async def fake_update(
        _repository: Path,
        newer_commit: str,
        older_commit: str,
        newer_counts: dict[str, int],
    ) -> dict[str, int]:
        assert (newer_commit, older_commit) == ("newest", "older")
        assert newer_counts == {"fetchurl": 5}
        return {"fetchurl": 4}

    monkeypatch.setattr(cli, "sampled_commits", fake_samples)
    monkeypatch.setattr(cli, "checkout", fake_checkout)
    monkeypatch.setattr(cli, "discover_fetchers", fake_discovery)
    monkeypatch.setattr(cli, "count_fetchers", fail_full_scan)
    monkeypatch.setattr(cli, "update_fetcher_counts", fake_update)

    await run(
        Config(
            nixpkgs=tmp_path / "nixpkgs",
            database=database_path,
            expression=tmp_path / "get-fetchers.nix",
        )
    )

    connection = sqlite3.connect(database_path)
    row = connection.execute(
        'SELECT "fetchurl" FROM "fetchers" WHERE "commit" = "older"'
    ).fetchone()
    connection.close()
    assert row == (4,)


@pytest.mark.asyncio
async def test_reverse_run_resumes_from_adjacent_completed_row(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    samples = [
        SampledCommit("newest", "2026-01-03T00:00:00Z"),
        SampledCommit("middle", "2026-01-02T00:00:00Z"),
        SampledCommit("oldest", "2026-01-01T00:00:00Z"),
    ]
    database_path = tmp_path / "fetchers.sqlite3"
    async with FetcherDatabase(database_path) as database:
        assert await database.store("oldest", samples[2].date, {"fetchurl": 3})

    async def fake_samples(
        _repository: Path,
        *,
        interval: int,
        first_parent: bool,
        completed: set[str] | None = None,
    ) -> list[SampledCommit]:
        _ = completed
        assert first_parent is True
        assert interval == 50
        return samples

    async def fake_checkout(_repository: Path, commit: str) -> None:
        assert commit in {"middle", "newest"}

    async def fake_discovery(
        _nixpkgs: Path,
        _expression: Path,
        *,
        commit: str,
    ) -> list[str]:
        assert commit in {"middle", "newest"}
        return ["fetchurl"]

    async def fail_full_scan(
        _repository: Path,
        _commit: str,
        _fetchers: list[str],
    ) -> dict[str, int]:
        pytest.fail("an adjacent completed row should avoid a full scan")

    updates: list[tuple[str, str, dict[str, int]]] = []

    async def fake_update(
        _repository: Path,
        base_commit: str,
        target_commit: str,
        base_counts: dict[str, int],
    ) -> dict[str, int]:
        updates.append((base_commit, target_commit, base_counts))
        return {"fetchurl": base_counts["fetchurl"] + 1}

    monkeypatch.setattr(cli, "sampled_commits", fake_samples)
    monkeypatch.setattr(cli, "checkout", fake_checkout)
    monkeypatch.setattr(cli, "discover_fetchers", fake_discovery)
    monkeypatch.setattr(cli, "count_fetchers", fail_full_scan)
    monkeypatch.setattr(cli, "update_fetcher_counts", fake_update)

    await run(
        Config(
            nixpkgs=tmp_path / "nixpkgs",
            database=database_path,
            expression=tmp_path / "get-fetchers.nix",
            reverse=True,
        )
    )

    assert updates == [
        ("oldest", "middle", {"fetchurl": 3}),
        ("middle", "newest", {"fetchurl": 4}),
    ]


@pytest.mark.asyncio
async def test_run_falls_back_when_incremental_count_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    samples = [
        SampledCommit("newest", "2026-01-02T00:00:00Z"),
        SampledCommit("older", "2026-01-01T00:00:00Z"),
    ]
    database_path = tmp_path / "fetchers.sqlite3"
    async with FetcherDatabase(database_path) as database:
        assert await database.store("newest", samples[0].date, {"fetchurl": 5})

    async def fake_samples(
        _repository: Path,
        *,
        interval: int,
        first_parent: bool,
        completed: set[str] | None = None,
    ) -> list[SampledCommit]:
        _ = completed
        assert first_parent is True
        assert interval == 50
        return samples

    async def fake_checkout(_repository: Path, _commit: str) -> None:
        return None

    async def fake_discovery(
        _nixpkgs: Path,
        _expression: Path,
        *,
        commit: str,
    ) -> list[str]:
        assert commit == "older"
        return ["fetchurl"]

    async def fake_update(
        _repository: Path,
        _newer_commit: str,
        _older_commit: str,
        _newer_counts: dict[str, int],
    ) -> dict[str, int]:
        raise IncrementalCountError("bad diff")

    async def fake_counts(
        _repository: Path,
        commit: str,
        _fetchers: list[str],
    ) -> dict[str, int]:
        assert commit == "older"
        return {"fetchurl": 4}

    monkeypatch.setattr(cli, "sampled_commits", fake_samples)
    monkeypatch.setattr(cli, "checkout", fake_checkout)
    monkeypatch.setattr(cli, "discover_fetchers", fake_discovery)
    monkeypatch.setattr(cli, "count_fetchers", fake_counts)
    monkeypatch.setattr(cli, "update_fetcher_counts", fake_update)

    await run(
        Config(
            nixpkgs=tmp_path / "nixpkgs",
            database=database_path,
            expression=tmp_path / "get-fetchers.nix",
        )
    )

    connection = sqlite3.connect(database_path)
    row = connection.execute(
        'SELECT "fetchurl" FROM "fetchers" WHERE "commit" = "older"'
    ).fetchone()
    connection.close()
    assert row == (4,)


def samples_named(*names: str) -> list[SampledCommit]:
    return [
        SampledCommit(name, f"2026-01-{position:02d}T00:00:00Z")
        for position, name in enumerate(names, start=1)
    ]


def indexed(samples: list[SampledCommit]) -> list[tuple[int, SampledCommit]]:
    return list(enumerate(samples))


def test_reverse_traversal_preserves_global_history_indices() -> None:
    samples = samples_named("newest", "middle", "oldest")

    traversed = cli.indexed_for_traversal(samples, reverse=True)

    assert traversed == [(2, samples[2]), (1, samples[1]), (0, samples[0])]


class RecordingProgress:
    instances: ClassVar[list[RecordingProgress]] = []

    def __init__(self, *_columns: object, console: object) -> None:
        self.console: object = console
        self.tasks: list[tuple[int, str, int]] = []
        self.advanced: list[int] = []
        self.instances.append(self)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def add_task(self, description: str, *, total: int) -> int:
        task_id = len(self.tasks)
        self.tasks.append((task_id, description, total))
        return task_id

    def advance(self, task_id: int) -> None:
        self.advanced.append(task_id)


def install_recording_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> list[RecordingProgress]:
    RecordingProgress.instances = []
    monkeypatch.setattr(cli, "Progress", RecordingProgress)
    return RecordingProgress.instances


class Environment:
    """The expensive stages of a run, replaced by recording fakes."""

    def __init__(self, samples: list[SampledCommit]) -> None:
        self.samples: list[SampledCommit] = samples
        self.sampled: int = 0
        self.checkouts: list[tuple[Path, str]] = []
        self.native_checkouts: list[str] = []
        self.materializations: list[str] = []
        self.restorations: list[str] = []
        self.full_scans: list[str] = []
        self.updates: list[tuple[str, str]] = []
        self.requests: list[WorktreeRequest] = []
        self.pools: list[tuple[Path, Path]] = []
        self.provisions: int = 0
        self.cancelled: list[str] = []


def install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    samples: list[SampledCommit],
    *,
    fetchers: list[str] | None = None,
    on_checkout: Callable[[Path, str], Awaitable[None]] | None = None,
    on_sample: Callable[[], None] | None = None,
    counts: Callable[[str], dict[str, int]] | None = None,
) -> Environment:
    environment = Environment(samples)

    async def fake_samples(
        _repository: Path,
        *,
        interval: int,
        first_parent: bool,
        completed: set[str] | None = None,
    ) -> list[SampledCommit]:
        _ = completed
        assert first_parent is True
        assert interval == cli.DEFAULT_INTERVAL
        environment.sampled += 1
        if on_sample is not None:
            on_sample()
        return samples

    async def fake_checkout(repository: Path, commit: str) -> None:
        environment.checkouts.append((repository, commit))
        if on_checkout is not None:
            await on_checkout(repository, commit)

    async def fake_discovery(
        _nixpkgs: Path,
        _expression: Path,
        *,
        commit: str,
    ) -> list[str]:
        assert commit
        return list(fetchers) if fetchers is not None else ["fetchurl"]

    async def fake_counts(
        _repository: Path,
        commit: str,
        sample_fetchers: list[str],
    ) -> dict[str, int]:
        environment.full_scans.append(commit)
        if counts is not None:
            return counts(commit)
        return dict.fromkeys(sample_fetchers, 10)

    async def fake_update(
        _repository: Path,
        newer_commit: str,
        older_commit: str,
        newer_counts: dict[str, int],
    ) -> dict[str, int]:
        environment.updates.append((newer_commit, older_commit))
        return dict(newer_counts)

    async def fake_provision(
        *,
        repository: Path,
        pool_dir: Path,
        requests: Sequence[WorktreeRequest],
    ) -> dict[int, Path]:
        environment.provisions += 1
        environment.pools.append((repository, pool_dir))
        environment.requests.extend(requests)
        worktrees: dict[int, Path] = {}
        for request in requests:
            worker = pool_dir / f"worker-{request.index}"
            worker.mkdir(parents=True, exist_ok=True)
            worktrees[request.index] = worker
        return worktrees

    monkeypatch.setattr(cli, "sampled_commits", fake_samples)
    monkeypatch.setattr(cli, "checkout", fake_checkout)
    monkeypatch.setattr(cli, "discover_fetchers", fake_discovery)
    monkeypatch.setattr(cli, "count_fetchers", fake_counts)
    monkeypatch.setattr(cli, "update_fetcher_counts", fake_update)
    monkeypatch.setattr(cli, "provision_worktrees", fake_provision)
    return environment


def install_recording_materializer(
    monkeypatch: pytest.MonkeyPatch,
    environment: Environment,
    *,
    fail_materialization: set[str] | None = None,
) -> None:
    failures: set[str] = (
        fail_materialization if fail_materialization is not None else set()
    )

    class RecordingMaterializer:
        def __init__(
            self,
            repository: Path,
            path: Path,
            state_path: Path | None = None,
            checkout_function: Callable[
                [Path, str], Awaitable[None]
            ] = history_checkout,
        ) -> None:
            _ = repository, state_path
            self.path: Path = path
            self.checkout_function: Callable[[Path, str], Awaitable[None]] = (
                checkout_function
            )
            self.current_commit: str | None = None
            self.native_commit: str | None = None
            self.recovery_required: bool = False

        async def native_checkout(self, commit: str) -> None:
            environment.native_checkouts.append(commit)
            await self.checkout_function(self.path, commit)
            self.current_commit = commit
            self.native_commit = commit
            self.recovery_required = False

        async def materialize(self, commit: str) -> None:
            environment.materializations.append(commit)
            if commit in failures:
                self.recovery_required = True
                raise RuntimeError("materialization failed")
            self.current_commit = commit

        async def restore_pristine(self) -> None:
            if self.current_commit == self.native_commit:
                return
            assert self.current_commit is not None
            environment.restorations.append(self.current_commit)
            await self.checkout_function(self.path, self.current_commit)
            self.native_commit = self.current_commit

    monkeypatch.setattr(cli, "MaterializedWorktree", RecordingMaterializer)


def parallel_config(tmp_path: Path, **overrides: object) -> Config:
    settings: dict[str, object] = {
        "nixpkgs": tmp_path / "nixpkgs",
        "database": tmp_path / "fetchers.sqlite3",
        "expression": tmp_path / "get-fetchers.nix",
        "workers": 2,
        "worktrees_dir": tmp_path / "pool",
    }
    settings.update(overrides)
    return Config(**settings)  # pyright: ignore[reportArgumentType]


def stored_counts(database_path: Path, column: str) -> dict[str, int | None]:
    connection = sqlite3.connect(database_path)
    try:
        rows = connection.execute(
            f'SELECT "commit", "{column}" FROM "fetchers"'  # noqa: S608
        ).fetchall()
    finally:
        connection.close()
    return {str(row[0]): row[1] for row in rows}


def stored_fetcher_counts(database_path: Path) -> dict[str, dict[str, int]]:
    """Read each row as its non-null fetcher columns.

    Physical column order and the winner of a case-only collision depend on
    which shard stored first, so tests must never assume either.
    """
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute('SELECT * FROM "fetchers"').fetchall()
    finally:
        connection.close()
    return {
        str(row["commit"]): {
            column: int(row[column])
            for column in row.keys()  # noqa: SIM118
            if column not in {"commit", "date", "is_skipped"}
            and row[column] is not None
        }
        for row in rows
    }


def test_parse_args_accepts_worker_settings() -> None:
    config = parse_args(["--workers", "4", "--worktrees-dir", "pool"])

    assert config.workers == 4
    assert config.worktrees_dir == Path("pool")


def test_parse_args_rejects_nonpositive_workers() -> None:
    with pytest.raises(SystemExit):
        _ = parse_args(["--workers", "0"])


@pytest.mark.asyncio
async def test_run_rejects_nonpositive_workers(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="workers must be positive"):
        await run(parallel_config(tmp_path, workers=0))


def shard_pending(
    samples: list[SampledCommit],
    workers: int,
    completed: set[str] | None = None,
) -> list[cli.ActiveShard]:
    stored: set[str] = completed if completed is not None else set()
    return cli.split_pending(
        cli.build_pending(indexed(samples), stored),
        workers,
        stored,
    )


def test_shards_split_pending_work_contiguously_and_exhaustively() -> None:
    samples = samples_named(*(f"commit-{number}" for number in range(7)))

    shards = shard_pending(samples, 3)

    assert [shard.index for shard in shards] == [0, 1, 2]
    assert [len(shard.pending) for shard in shards] == [2, 2, 3]
    assert [item.global_index for shard in shards for item in shard.pending] == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
    ]


def test_shards_stay_balanced_when_most_of_the_history_is_stored() -> None:
    samples = samples_named(*(f"commit-{number}" for number in range(6)))
    completed = {f"commit-{number}" for number in range(4)}

    shards = shard_pending(samples, 2, completed)

    assert [
        [item.sample.commit for item in shard.pending] for shard in shards
    ] == [["commit-4"], ["commit-5"]]


def test_more_workers_than_pending_samples_keeps_shard_indices() -> None:
    shards = shard_pending(samples_named("newest", "oldest"), 4)

    assert [shard.index for shard in shards if shard.pending] == [1, 3]
    assert [
        [item.sample.commit for item in shard.pending]
        for shard in shards
        if shard.pending
    ] == [["newest"], ["oldest"]]


def test_pending_samples_keep_global_indices_and_neighbours() -> None:
    samples = samples_named("a", "b", "c", "d")

    shards = shard_pending(samples, 2)

    assert [item.global_index for item in shards[1].pending] == [2, 3]
    assert shards[0].pending[1].previous_sample == samples[0]
    assert shards[1].pending[1].previous_sample == samples[2]


def test_a_shard_boundary_reuses_an_already_stored_neighbour() -> None:
    samples = samples_named("a", "b", "c", "d")

    shards = shard_pending(samples, 2, {"a", "b"})

    assert [item.sample.commit for item in shards[0].pending] == ["c"]
    assert shards[0].pending[0].previous_sample == samples[1]


def test_a_shard_boundary_never_uses_another_shard_pending_neighbour() -> None:
    samples = samples_named("a", "b", "c", "d")

    shards = shard_pending(samples, 2)

    assert [item.sample.commit for item in shards[1].pending] == ["c", "d"]
    assert shards[1].pending[0].previous_sample is None
    assert shards[0].pending[0].previous_sample is None


@pytest.mark.asyncio
async def test_single_worker_uses_the_checkout_without_a_pool(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    environment = install_fakes(monkeypatch, samples_named("newest", "oldest"))
    config = parallel_config(tmp_path, workers=1, worktrees_dir=None)

    await run(config)

    assert environment.checkouts == [
        (config.nixpkgs, "newest"),
        (config.nixpkgs, "oldest"),
    ]
    assert environment.provisions == 0
    assert not list(tmp_path.glob(".nixpkgs-fetcher-counter-worktrees*"))  # noqa: ASYNC240


@pytest.mark.asyncio
async def test_single_worker_reverse_processes_oldest_to_newest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    environment = install_fakes(
        monkeypatch, samples_named("newest", "middle", "oldest")
    )
    config = parallel_config(
        tmp_path,
        workers=1,
        worktrees_dir=None,
        reverse=True,
    )

    await run(config)

    assert environment.checkouts == [
        (config.nixpkgs, "oldest"),
        (config.nixpkgs, "middle"),
        (config.nixpkgs, "newest"),
    ]
    assert environment.full_scans == ["oldest"]
    assert environment.updates == [("oldest", "middle"), ("middle", "newest")]


@pytest.mark.asyncio
async def test_single_worker_shows_only_total_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _ = install_fakes(monkeypatch, samples_named("newest", "oldest"))
    progress_instances = install_recording_progress(monkeypatch)

    await run(parallel_config(tmp_path, workers=1, worktrees_dir=None))

    assert len(progress_instances) == 1
    progress = progress_instances[0]
    assert progress.tasks == [(0, "Total", 2)]
    assert progress.advanced == [0, 0]


@pytest.mark.asyncio
async def test_skipped_evaluation_advances_single_worker_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _ = install_fakes(monkeypatch, samples_named("failed"))
    progress_instances = install_recording_progress(monkeypatch)

    async def fail_discovery(
        _nixpkgs: Path,
        _expression: Path,
        *,
        commit: str,
    ) -> list[str]:
        assert commit == "failed"
        raise FetcherDiscoveryError("cannot evaluate")

    monkeypatch.setattr(cli, "discover_fetchers", fail_discovery)

    await run(parallel_config(tmp_path, workers=1, worktrees_dir=None))

    progress = progress_instances[0]
    assert progress.advanced == [0]
    connection = sqlite3.connect(tmp_path / "fetchers.sqlite3")
    row = connection.execute(
        'SELECT "is_skipped" FROM "fetchers" WHERE "commit" = "failed"'
    ).fetchone()
    connection.close()
    assert row == (1,)


@pytest.mark.asyncio
async def test_parallel_run_shows_total_and_each_active_shard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _ = install_fakes(monkeypatch, samples_named("a", "b", "c", "d"))
    progress_instances = install_recording_progress(monkeypatch)

    await run(parallel_config(tmp_path))

    progress = progress_instances[0]
    assert progress.tasks == [
        (0, "Total", 4),
        (1, "Shard 0", 2),
        (2, "Shard 1", 2),
    ]
    assert progress.advanced.count(0) == 4
    assert progress.advanced.count(1) == 2
    assert progress.advanced.count(2) == 2


@pytest.mark.asyncio
async def test_failed_sample_advances_progress_before_propagating(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fail_checkout(_repository: Path, _commit: str) -> None:
        raise RuntimeError("checkout failed")

    _ = install_fakes(
        monkeypatch,
        samples_named("failed"),
        on_checkout=fail_checkout,
    )
    progress_instances = install_recording_progress(monkeypatch)

    with pytest.raises(RuntimeError, match="checkout failed"):
        await run(parallel_config(tmp_path, workers=1, worktrees_dir=None))

    assert progress_instances[0].advanced == [0]


@pytest.mark.asyncio
async def test_parallel_run_processes_every_shard_in_its_own_worktree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    samples = samples_named("a", "b", "c", "d")
    environment = install_fakes(monkeypatch, samples)
    config = parallel_config(tmp_path)

    await run(config)

    assert environment.provisions == 1
    assert [
        (request.index, request.initial_commit) for request in environment.requests
    ] == [(0, "a"), (1, "c")]
    assert sorted(environment.checkouts) == [
        (tmp_path / "pool" / "worker-0", "a"),
        (tmp_path / "pool" / "worker-0", "b"),
        (tmp_path / "pool" / "worker-1", "c"),
        (tmp_path / "pool" / "worker-1", "d"),
    ]
    assert set(stored_counts(config.database, "fetchurl")) == {"a", "b", "c", "d"}


@pytest.mark.asyncio
async def test_parallel_reverse_shards_oldest_to_newest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    environment = install_fakes(monkeypatch, samples_named("a", "b", "c", "d"))
    config = parallel_config(tmp_path, reverse=True)

    await run(config)

    assert [
        (request.index, request.initial_commit) for request in environment.requests
    ] == [(0, "d"), (1, "b")]
    assert sorted(environment.checkouts) == [
        (tmp_path / "pool" / "worker-0", "c"),
        (tmp_path / "pool" / "worker-0", "d"),
        (tmp_path / "pool" / "worker-1", "a"),
        (tmp_path / "pool" / "worker-1", "b"),
    ]


@pytest.mark.asyncio
async def test_scheduled_full_scans_follow_the_global_history_position(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    samples = samples_named(*(f"commit-{number}" for number in range(6)))
    environment = install_fakes(monkeypatch, samples)

    await run(parallel_config(tmp_path, full_scan_interval=3))

    assert sorted(environment.full_scans) == [
        "commit-0",
        "commit-2",
        "commit-3",
        "commit-5",
    ]


@pytest.mark.asyncio
async def test_native_checkout_and_full_scan_cadences_are_independent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    samples = samples_named(*(f"commit-{number}" for number in range(12)))
    environment = install_fakes(monkeypatch, samples)
    install_recording_materializer(monkeypatch, environment)

    await run(
        parallel_config(
            tmp_path,
            workers=1,
            worktrees_dir=None,
            native_checkout_interval=3,
            full_scan_interval=4,
        )
    )

    assert environment.native_checkouts == [
        "commit-0",
        "commit-2",
        "commit-5",
        "commit-8",
        "commit-11",
    ]
    assert environment.materializations == [
        "commit-1",
        "commit-3",
        "commit-4",
        "commit-6",
        "commit-7",
        "commit-9",
        "commit-10",
    ]
    assert environment.full_scans == [
        "commit-0",
        "commit-3",
        "commit-7",
        "commit-11",
    ]
    assert ("commit-1", "commit-2") in environment.updates
    assert ("commit-2", "commit-3") not in environment.updates


@pytest.mark.asyncio
async def test_materialization_failure_falls_back_without_forcing_full_scan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    environment = install_fakes(monkeypatch, samples_named("first", "second"))
    install_recording_materializer(
        monkeypatch,
        environment,
        fail_materialization={"second"},
    )

    await run(
        parallel_config(
            tmp_path,
            workers=1,
            worktrees_dir=None,
            native_checkout_interval=25,
        )
    )

    assert environment.native_checkouts == ["first", "second"]
    assert environment.materializations == ["second"]
    assert environment.full_scans == ["first"]
    assert environment.updates == [("first", "second")]


@pytest.mark.asyncio
async def test_parallel_shards_restore_incremental_final_trees(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    environment = install_fakes(monkeypatch, samples_named("a", "b", "c", "d"))
    install_recording_materializer(monkeypatch, environment)

    await run(parallel_config(tmp_path, native_checkout_interval=25))

    assert sorted(environment.native_checkouts) == ["a", "c"]
    assert sorted(environment.materializations) == ["b", "d"]
    assert sorted(environment.restorations) == ["b", "d"]


@pytest.mark.asyncio
@pytest.mark.parametrize("completed", [False, True])
async def test_a_run_without_pending_work_only_takes_the_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    completed: bool,
) -> None:
    samples = samples_named("a", "b") if completed else []
    environment = install_fakes(monkeypatch, samples)
    config = parallel_config(tmp_path)
    if completed:
        async with FetcherDatabase(config.database) as database:
            for sample in samples:
                stored = await database.store(sample.commit, sample.date, {})
                assert stored

    await run(config)

    pool = tmp_path / "pool"
    assert environment.provisions == 0
    assert environment.checkouts == []
    assert (pool / "coordinator.lock").is_file()
    assert list(pool.iterdir()) == [pool / "coordinator.lock"]
    with worktree_pool_lock(pool):
        pass


@pytest.mark.asyncio
async def test_the_pool_lock_is_held_before_sampling_and_database_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = parallel_config(tmp_path)
    contended: list[bool] = []

    def on_sample() -> None:
        assert config.database.exists()
        try:
            with worktree_pool_lock(tmp_path / "pool"):
                contended.append(False)
        except WorktreePoolLockedError:
            contended.append(True)

    _ = install_fakes(
        monkeypatch,
        samples_named("a", "b"),
        on_sample=on_sample,
    )

    await run(config)

    assert contended == [True]


@pytest.mark.asyncio
async def test_a_second_coordinator_fails_before_touching_anything(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    environment = install_fakes(monkeypatch, samples_named("a", "b"))
    config = parallel_config(tmp_path)

    with (
        worktree_pool_lock(tmp_path / "pool"),
        pytest.raises(WorktreePoolLockedError),
    ):
        await run(config)

    assert environment.sampled == 0
    assert not config.database.exists()


@pytest.mark.asyncio
async def test_run_derives_the_pool_from_the_nixpkgs_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    nixpkgs = tmp_path / "nixpkgs"
    nixpkgs.mkdir()
    environment = install_fakes(monkeypatch, samples_named("a", "b"))

    await run(parallel_config(tmp_path, worktrees_dir=None))

    pool = tmp_path / ".nixpkgs-fetcher-counter-worktrees"
    assert environment.pools == [(nixpkgs, pool)]
    assert (pool / "coordinator.lock").is_file()
    assert (pool / "worker-0").is_dir()


@pytest.mark.asyncio
async def test_shards_make_progress_concurrently(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    started = asyncio.Barrier(2)

    async def on_checkout(_repository: Path, _commit: str) -> None:
        _ = await asyncio.wait_for(started.wait(), timeout=5)

    _ = install_fakes(
        monkeypatch,
        samples_named("a", "b", "c", "d"),
        on_checkout=on_checkout,
    )
    config = parallel_config(tmp_path)

    await run(config)

    assert set(stored_counts(config.database, "fetchurl")) == {"a", "b", "c", "d"}


@pytest.mark.asyncio
async def test_concurrent_shards_store_distinct_fetcher_columns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def on_checkout(_repository: Path, _commit: str) -> None:
        await asyncio.sleep(0)

    def counts(commit: str) -> dict[str, int]:
        return {f"fetch{commit}": ord(commit[0])}

    def discovery_counts(commit: str) -> dict[str, int]:
        return counts(commit)

    _ = install_fakes(
        monkeypatch,
        samples_named("a", "c"),
        on_checkout=on_checkout,
        counts=discovery_counts,
    )
    config = parallel_config(tmp_path)

    await run(config)

    assert stored_counts(config.database, "fetcha") == {"a": ord("a"), "c": None}
    assert stored_counts(config.database, "fetchc") == {"a": None, "c": ord("c")}


@pytest.mark.asyncio
async def test_concurrent_shards_resolve_case_colliding_fetchers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def counts(commit: str) -> dict[str, int]:
        name = "fetchFromGitHub" if commit == "a" else "fetchFromGithub"
        return {name: 1 if commit == "a" else 2}

    _ = install_fakes(monkeypatch, samples_named("a", "c"), counts=counts)
    config = parallel_config(tmp_path)

    await run(config)

    rows = stored_fetcher_counts(config.database)
    values = {commit: sorted(counts.values()) for commit, counts in rows.items()}
    assert values == {"a": [1], "c": [2]}
    columns = {column for counts in rows.values() for column in counts}
    assert {column.casefold() for column in columns} == {
        "fetchfromgithub",
        "fetchfromgithub__case_collision_1",
    }


@pytest.mark.asyncio
async def test_a_failing_shard_lets_its_siblings_finish(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def on_checkout(_repository: Path, commit: str) -> None:
        if commit == "a":
            raise RuntimeError("shard 0 broke")
        await asyncio.sleep(0)

    _ = install_fakes(
        monkeypatch,
        samples_named("a", "b", "c", "d"),
        on_checkout=on_checkout,
    )
    config = parallel_config(tmp_path)

    with pytest.raises(ExceptionGroup) as error:
        await run(config)

    failures = error.value.exceptions
    assert len(failures) == 1
    failure = failures[0]
    assert isinstance(failure, cli.ShardError)
    assert failure.shard_index == 0
    assert isinstance(failure.cause, RuntimeError)
    assert set(stored_counts(config.database, "fetchurl")) == {"c", "d"}
    with worktree_pool_lock(tmp_path / "pool"):
        pass


@pytest.mark.asyncio
async def test_an_independently_cancelled_shard_reraises_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def on_checkout(_repository: Path, commit: str) -> None:
        if commit == "a":
            raise asyncio.CancelledError
        await asyncio.sleep(0)

    _ = install_fakes(
        monkeypatch,
        samples_named("a", "b", "c", "d"),
        on_checkout=on_checkout,
    )
    config = parallel_config(tmp_path)

    with pytest.raises(asyncio.CancelledError):
        await run(config)

    assert set(stored_counts(config.database, "fetchurl")) == {"c", "d"}
    with worktree_pool_lock(tmp_path / "pool"):
        pass


@pytest.mark.asyncio
async def test_external_cancellation_cancels_every_shard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    running = asyncio.Barrier(2)
    cancelled: list[str] = []

    async def on_checkout(_repository: Path, commit: str) -> None:
        try:
            _ = await running.wait()
            _ = await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(commit)
            raise

    _ = install_fakes(
        monkeypatch,
        samples_named("a", "b", "c", "d"),
        on_checkout=on_checkout,
    )
    config = parallel_config(tmp_path)
    counting = asyncio.create_task(run(config))
    await asyncio.sleep(0.05)

    _ = counting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await counting

    assert sorted(cancelled) == ["a", "c"]
    with worktree_pool_lock(tmp_path / "pool"):
        pass


def test_main_derives_the_pool_from_the_resolved_nixpkgs_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    nixpkgs = tmp_path / "nixpkgs"
    nixpkgs.mkdir()
    link = tmp_path / "link-to-nixpkgs"
    link.symlink_to(nixpkgs)
    expression = tmp_path / "get-fetchers.nix"
    _ = expression.write_text("[]")
    configurations: list[Config] = []

    async def fake_run(config: Config) -> None:
        configurations.append(config)

    def fake_configure_logging(_log_level: str) -> None:
        return None

    monkeypatch.setattr(cli, "run", fake_run)
    monkeypatch.setattr(cli, "configure_logging", fake_configure_logging)

    for checkout_path in (nixpkgs, link):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "fetcher-counter",
                "--nixpkgs",
                str(checkout_path),
                "--database",
                str(tmp_path / "data" / "fetchers.sqlite3"),
                "--expression",
                str(expression),
                "--workers",
                "2",
            ],
        )
        cli.main()

    pool = tmp_path / ".nixpkgs-fetcher-counter-worktrees"
    assert [config.worktrees_dir for config in configurations] == [pool, pool]
    assert [config.nixpkgs for config in configurations] == [nixpkgs, nixpkgs]


def test_main_resolves_an_explicit_worktrees_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    nixpkgs = tmp_path / "nixpkgs"
    nixpkgs.mkdir()
    expression = tmp_path / "get-fetchers.nix"
    _ = expression.write_text("[]")
    configurations: list[Config] = []

    async def fake_run(config: Config) -> None:
        configurations.append(config)

    def fake_configure_logging(_log_level: str) -> None:
        return None

    monkeypatch.setattr(cli, "run", fake_run)
    monkeypatch.setattr(cli, "configure_logging", fake_configure_logging)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fetcher-counter",
            "--nixpkgs",
            str(nixpkgs),
            "--database",
            str(tmp_path / "fetchers.sqlite3"),
            "--expression",
            str(expression),
            "--worktrees-dir",
            str(tmp_path / "pool" / ".." / "pool"),
        ],
    )

    cli.main()

    assert configurations[0].worktrees_dir == tmp_path / "pool"


@pytest.mark.asyncio
async def test_a_resumed_parallel_run_balances_pending_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    samples = samples_named(*(f"commit-{number}" for number in range(6)))
    environment = install_fakes(monkeypatch, samples)
    config = parallel_config(tmp_path)
    async with FetcherDatabase(config.database) as database:
        for sample in samples[:4]:
            stored = await database.store(sample.commit, sample.date, {})
            assert stored

    await run(config)

    assert environment.checkouts == [
        (tmp_path / "pool" / "worker-0", "commit-4"),
        (tmp_path / "pool" / "worker-1", "commit-5"),
    ]


@pytest.mark.asyncio
async def test_a_shard_boundary_counts_from_a_stored_neighbour(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    samples = samples_named("a", "b", "c", "d")
    environment = install_fakes(monkeypatch, samples)
    config = parallel_config(tmp_path)
    async with FetcherDatabase(config.database) as database:
        stored = await database.store("a", samples[0].date, {"fetchurl": 5})
        assert stored

    await run(config)

    assert environment.updates == [("a", "b"), ("c", "d")]
    assert environment.full_scans == ["c"]
