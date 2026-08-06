import asyncio
import sqlite3
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Self

from loguru import logger


def quote_identifier(identifier: str) -> str:
    if "\0" in identifier:
        raise ValueError("SQLite identifiers cannot contain NUL characters")
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


def resolve_column_names(
    fetchers: Iterable[str],
    existing_columns: set[str],
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    used = {column.casefold() for column in existing_columns}
    for fetcher in sorted(set(fetchers)):
        if fetcher in existing_columns or fetcher.casefold() not in used:
            column = fetcher
        else:
            suffix = 1
            while True:
                candidate = f"{fetcher}__case_collision_{suffix}"
                if (
                    candidate in existing_columns
                    or candidate.casefold() not in used
                ):
                    column = candidate
                    break
                suffix += 1
        resolved[fetcher] = column
        used.add(column.casefold())
    return resolved


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
                    "date" TEXT NOT NULL,
                    "is_skipped" INTEGER NOT NULL DEFAULT 0
                )"""
            )
            columns = {
                str(row[1])
                for row in self._connection.execute(
                    'PRAGMA table_info("fetchers")'
                )
            }
            if "is_skipped" not in columns:
                logger.debug("Adding is_skipped column to existing database")
                _ = self._connection.execute(
                    """ALTER TABLE "fetchers" ADD COLUMN
                    "is_skipped" INTEGER NOT NULL DEFAULT 0"""
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
        *,
        is_skipped: bool = False,
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
            column_names = resolve_column_names(counts, existing_columns)
            new_columns = sorted(set(column_names.values()) - existing_columns)
            aliases = {
                fetcher: column
                for fetcher, column in column_names.items()
                if fetcher != column
            }
            logger.debug(
                "Storing {} with {} counts (is_skipped={})",
                commit,
                len(counts),
                is_skipped,
            )
            logger.debug(
                "Database columns to add: {}; aliases: {}",
                new_columns,
                aliases,
            )
            _ = self._connection.execute("BEGIN")
            try:
                for name in new_columns:
                    column = quote_identifier(name)
                    _ = self._connection.execute(
                        f'ALTER TABLE "fetchers" ADD COLUMN {column} INTEGER'
                    )

                names = [
                    "commit",
                    "date",
                    "is_skipped",
                    *(column_names[fetcher] for fetcher in counts),
                ]
                columns = ", ".join(quote_identifier(name) for name in names)
                placeholders = ", ".join("?" for _ in names)
                values: list[str | int] = [
                    commit,
                    date,
                    int(is_skipped),
                    *counts.values(),
                ]
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
