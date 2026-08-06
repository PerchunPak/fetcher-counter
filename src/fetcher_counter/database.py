import asyncio
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Self


def quote_identifier(identifier: str) -> str:
    if "\0" in identifier:
        raise ValueError("SQLite identifiers cannot contain NUL characters")
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


class FetcherDatabase:
    def __init__(self, path: Path) -> None:
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

    async def completed_commits(self) -> set[str]:
        async with self._lock:
            rows = self._connection.execute(
                'SELECT "commit" FROM "fetchers"'
            ).fetchall()
        return {str(row[0]) for row in rows}

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
                return False

            existing_columns = {
                str(row[1])
                for row in self._connection.execute(
                    'PRAGMA table_info("fetchers")'
                )
            }
            _ = self._connection.execute("BEGIN")
            try:
                for fetcher in sorted(counts.keys() - existing_columns):
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
                self._connection.rollback()
                raise
            else:
                self._connection.commit()
            return True

    async def close(self) -> None:
        async with self._lock:
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
