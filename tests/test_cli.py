import sqlite3
from pathlib import Path

import pytest

from fetcher_counter import cli
from fetcher_counter.cli import Config, parse_args, run
from fetcher_counter.database import FetcherDatabase
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

    await run(
        Config(
            nixpkgs=tmp_path / "nixpkgs",
            database=database_path,
            expression=tmp_path / "get-fetchers.nix",
        )
    )

    connection = sqlite3.connect(database_path)
    rows = connection.execute(
        """SELECT "commit", "oldFetcher", "fetchurl", "fetchzip"
        FROM "fetchers" ORDER BY "date"
        """
    ).fetchall()
    connection.close()

    assert checkouts == ["second", "third"]
    assert rows == [
        ("first", 1, None, None),
        ("second", None, 2, None),
        ("third", None, None, 3),
    ]
