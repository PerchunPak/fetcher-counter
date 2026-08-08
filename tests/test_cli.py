import sqlite3
import sys
from pathlib import Path

import pytest
from loguru import logger

from fetcher_counter import cli
from fetcher_counter.cli import Config, parse_args, run
from fetcher_counter.counting import IncrementalCountError
from fetcher_counter.database import FetcherDatabase
from fetcher_counter.discovery import FetcherDiscoveryError
from fetcher_counter.history import SampledCommit


def test_parse_args_uses_project_defaults() -> None:
    config = parse_args([])

    assert config.nixpkgs == Path("nixpkgs")
    assert config.database == Path("data/fetchers.sqlite3")
    assert config.interval == 50
    assert config.log_level == "INFO"


def test_parse_args_accepts_case_insensitive_log_level() -> None:
    config = parse_args(["--log-level", "debug"])

    assert config.log_level == "DEBUG"


def test_configure_logging_replaces_default_sink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []

    class FakeLogger:
        def remove(self) -> None:
            calls.append(("remove",))

        def add(self, sink: object, *, level: str) -> int:
            calls.append(("add", sink, level))
            return 1

    monkeypatch.setattr(cli, "logger", FakeLogger())

    cli.configure_logging("WARNING")

    assert calls == [("remove",), ("add", sys.stderr, "WARNING")]


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
    ) -> list[SampledCommit]:
        assert interval == 50
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
    ) -> list[SampledCommit]:
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
    ) -> list[SampledCommit]:
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
    ) -> list[SampledCommit]:
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
    ) -> list[SampledCommit]:
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
