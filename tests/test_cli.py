import sqlite3
from pathlib import Path

import pytest
from loguru import logger

from fetcher_counter import cli
from fetcher_counter.cli import Config, parse_args, run
from fetcher_counter.database import FetcherDatabase
from fetcher_counter.discovery import FetcherDiscoveryError
from fetcher_counter.history import SampledCommit


def test_parse_args_uses_project_defaults() -> None:
    config = parse_args([])

    assert config.nixpkgs == Path("nixpkgs")
    assert config.database == Path("data/fetchers.sqlite3")
    assert config.interval == 50


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
