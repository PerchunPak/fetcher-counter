import sqlite3
from pathlib import Path

import pytest

from fetcher_counter.database import FetcherDatabase, quote_identifier


@pytest.mark.asyncio
async def test_database_evolves_columns_and_preserves_nulls(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fetchers.sqlite3"
    database = FetcherDatabase(path)
    await database.initialize()

    assert await database.store("first", "2026-01-01T00:00:00Z", {"fetchurl": 2})
    assert await database.store("second", "2026-01-02T00:00:00Z", {"fetchzip": 3})
    assert not await database.store(
        "second", "2026-01-02T00:00:00Z", {"fetchzip": 99}
    )
    assert await database.completed_commits() == {"first", "second"}
    await database.close()

    connection = sqlite3.connect(path)
    columns = [
        str(row[1]) for row in connection.execute('PRAGMA table_info("fetchers")')
    ]
    rows = connection.execute(
        'SELECT "commit", "fetchurl", "fetchzip" FROM "fetchers" ORDER BY "commit"'
    ).fetchall()
    tables = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    connection.close()

    assert columns == ["commit", "date", "fetchurl", "fetchzip"]
    assert rows == [("first", 2, None), ("second", None, 3)]
    assert tables == [("fetchers",)]


@pytest.mark.asyncio
async def test_store_rolls_back_columns_and_row_on_failure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fetchers.sqlite3"
    async with FetcherDatabase(path) as database:
        _ = database._connection.execute(
            """CREATE TRIGGER fail_insert
            BEFORE INSERT ON fetchers
            BEGIN SELECT RAISE(ABORT, 'fail'); END"""
        )
        database._connection.commit()
        with pytest.raises(sqlite3.IntegrityError, match="fail"):
            _ = await database.store(
                "broken",
                "2026-01-01T00:00:00Z",
                {"newFetcher": 1},
            )

    connection = sqlite3.connect(path)
    columns = [
        str(row[1]) for row in connection.execute('PRAGMA table_info("fetchers")')
    ]
    rows = connection.execute('SELECT * FROM "fetchers"').fetchall()
    connection.close()

    assert columns == ["commit", "date"]
    assert rows == []


def test_quote_identifier_escapes_quotes_and_rejects_nul() -> None:
    assert quote_identifier('fetch"quoted') == '"fetch""quoted"'
    with pytest.raises(ValueError, match="NUL"):
        _ = quote_identifier("fetch\0url")
