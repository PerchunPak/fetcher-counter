import asyncio
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Self

from loguru import logger


def quote_identifier(identifier: str) -> str:
    if "\0" in identifier:
        raise ValueError("SQLite identifiers cannot contain NUL characters")
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


class FetcherDatabase:
    def __init__(self, path: Path) -> None:
        logger.debug("Opening SQLite database at {}", path)
        self._connection: sqlite3.Connection = sqlite3.connect(path)
        self._lock: asyncio.Lock = asyncio.Lock()

    async def initialize(self) -> None:
        async with self._lock:
            _ = self._connection.execute(
                """CREATE TABLE IF NOT EXISTS "fetchers" (
                    "commit" TEXT PRIMARY KEY,
                    "date" TEXT NOT NULL
                )"""
            )
            self._connection.commit()
            logger.debug("Initialized fetchers table")

    async def completed_commits(self) -> set[str]:
        async with self._lock:
            rows = self._connection.execute(
                'SELECT "commit" FROM "fetchers"'
            ).fetchall()
        commits = {str(row[0]) for row in rows}
        logger.debug("Loaded {} completed commits", len(commits))
        return commits

    async def store(
        self,
        commit: str,
        date: str,
        counts: Mapping[str, int],
    ) -> bool:
        async with self._lock:
            if self._connection.execute(
                'SELECT 1 FROM "fetchers" WHERE "commit" = ?', (commit,)
            ).fetchone():
                logger.debug("Commit {} is already stored", commit)
                return False

            existing_columns = {
                str(row[1])
                for row in self._connection.execute(
                    'PRAGMA table_info("fetchers")'
                )
            }
            new_fetchers = sorted(counts.keys() - existing_columns)
            logger.debug(
                "Storing {} with {} counts; adding columns {}",
                commit,
                len(counts),
                new_fetchers,
            )
            _ = self._connection.execute("BEGIN")
            try:
                for fetcher in new_fetchers:
                    column = quote_identifier(fetcher)
                    _ = self._connection.execute(
                        f'ALTER TABLE "fetchers" ADD COLUMN {column} INTEGER'
                    )

                names = ["commit", "date", *counts]
                columns = ", ".join(quote_identifier(name) for name in names)
                placeholders = ", ".join("?" for _ in names)
                values: list[str | int] = [commit, date, *counts.values()]
                _ = self._connection.execute(
                    f'INSERT INTO "fetchers" ({columns}) VALUES ({placeholders})',  # noqa: S608
                    values,
                )
            except BaseException:
                logger.debug("Rolling back database transaction for {}", commit)
                self._connection.rollback()
                raise
            else:
                self._connection.commit()
                logger.debug("Committed database row for {}", commit)
            return True

    async def close(self) -> None:
        async with self._lock:
            logger.debug("Closing SQLite database")
            self._connection.close()

    async def __aenter__(self) -> Self:
        await self.initialize()
        return self

    async def __aexit__(
        self,
        _exception_type: object,
        _exception: object,
        _traceback: object,
    ) -> None:
        await self.close()
