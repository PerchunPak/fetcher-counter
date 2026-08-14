#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pillow>=11.0",
# ]
# ///
# ruff: noqa: S603, S607, T201

import argparse
import calendar
import math
import sqlite3
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BACKGROUND = "#191826"
NO_COMMITS = "#211F30"
MISSING = "#332B45"
PARTIAL_LOW = "#5B4D73"
PARTIAL_MID = "#7B6698"
PROCESSED = "#9C7DBF"
TEXT = "#C9C3D6"
MUTED_TEXT = "#89839A"


@dataclass(frozen=True, slots=True)
class Commit:
    hash: str
    date: datetime

    @property
    def year(self) -> int:
        return self.date.year


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render yearly Nixpkgs commit-processing coverage grids."
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
        help="Fetcher Counter SQLite database (default: ./data/fetchers.sqlite3)",
    )
    _ = parser.add_argument(
        "--output",
        type=Path,
        default=Path("commit-coverage.png"),
        help="Output PNG path (default: ./commit-coverage.png)",
    )
    _ = parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="Oldest-anchored commit sampling interval (default: 1)",
    )
    _ = parser.add_argument(
        "--rows",
        type=int,
        default=7,
        help="Rows in each yearly grid (default: 7; one week per column)",
    )
    _ = parser.add_argument(
        "--columns",
        type=int,
        default=53,
        help="Columns in each yearly grid (default: 53)",
    )
    _ = parser.add_argument(
        "--cell-size",
        type=int,
        default=13,
        help="Cell width and height in pixels (default: 13)",
    )
    _ = parser.add_argument(
        "--gap",
        type=int,
        default=4,
        help="Gap between cells in pixels (default: 4)",
    )
    _ = parser.add_argument(
        "--tip",
        default="refs/fetcher-counter/history-tip",
        help="Git revision to visualize (default: saved Fetcher Counter tip)",
    )
    _ = parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Omit the shared legend and overall summary",
    )
    args = parser.parse_args()
    for name in ("interval", "rows", "columns", "cell_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.gap < 0:
        parser.error("--gap must not be negative")
    return args


def git_output(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or "unknown Git error"
        raise SystemExit(f"git {' '.join(arguments)} failed: {message}")
    return result.stdout


def resolve_tip(repository: Path, requested_tip: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "--verify", requested_tip],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    if requested_tip == "refs/fetcher-counter/history-tip":
        return git_output(repository, "rev-parse", "HEAD").strip()
    message = result.stderr.strip() or "revision does not exist"
    raise SystemExit(f"cannot resolve --tip {requested_tip!r}: {message}")


def sampled_commits(repository: Path, tip: str, interval: int) -> list[Commit]:
    lines = git_output(
        repository,
        "log",
        "--first-parent",
        "--reverse",
        "--format=%H%x00%cI",
        tip,
    ).splitlines()
    commits: list[Commit] = []
    for line in lines[::interval]:
        commit_hash, date = line.split("\0", maxsplit=1)
        commits.append(
            Commit(
                hash=commit_hash,
                date=datetime.fromisoformat(date),
            )
        )
    return commits


def completed_commits(database: Path) -> set[str]:
    uri = f"file:{database.resolve()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True, timeout=5) as connection:
            rows = connection.execute('SELECT "commit" FROM "fetchers"')
            return {str(row[0]) for row in rows}
    except sqlite3.Error as error:
        raise SystemExit(f"cannot read {database}: {error}") from error


def mix(left: str, right: str, amount: float) -> str:
    left_rgb = tuple(int(left[index : index + 2], 16) for index in (1, 3, 5))
    right_rgb = tuple(int(right[index : index + 2], 16) for index in (1, 3, 5))
    mixed = tuple(
        round(start + (end - start) * amount)
        for start, end in zip(left_rgb, right_rgb, strict=True)
    )
    return "#" + "".join(f"{channel:02X}" for channel in mixed)


def coverage_color(coverage: float | None) -> str:
    if coverage is None:
        return NO_COMMITS
    if coverage <= 0:
        return MISSING
    if coverage >= 1:
        return PROCESSED
    if coverage < 0.5:
        return mix(PARTIAL_LOW, PARTIAL_MID, coverage * 2)
    return mix(PARTIAL_MID, PROCESSED, (coverage - 0.5) * 2)


def aggregate_coverage(
    commits: list[Commit], completed: set[str], cells: int
) -> list[float | None]:
    year = commits[0].year
    start = date(year, 1, 1)
    days = 366 if calendar.isleap(year) else 365
    if cells < days:
        message = f"the {year} grid needs {days} cells; grid provides {cells}"
        raise SystemExit(message)

    totals = [0] * cells
    processed = [0] * cells
    for commit in commits:
        index = (commit.date.date() - start).days
        totals[index] += 1
        processed[index] += commit.hash in completed
    return [
        done / total if total else None
        for done, total in zip(processed, totals, strict=True)
    ]


def draw_key(
    draw: ImageDraw.ImageDraw,
    *,
    top: int,
    left: int,
    width: int,
    processed: int,
    total: int,
) -> None:
    font = ImageFont.load_default(size=12)
    marker_size = 9
    items = (
        ("Processed", PROCESSED),
        ("Partial", PARTIAL_MID),
        ("Missing", MISSING),
        ("No commits", NO_COMMITS),
    )
    x = left
    for label, color in items:
        draw.rounded_rectangle(
            (x, top + 1, x + marker_size, top + marker_size + 1),
            radius=2,
            fill=color,
        )
        x += marker_size + 5
        draw.text((x, top), label, fill=TEXT, font=font)
        x += math.ceil(draw.textlength(label, font=font)) + 14

    summary = f"{processed:,} / {total:,} commits ({processed / total:.1%})"
    summary_width = math.ceil(draw.textlength(summary, font=font))
    draw.text((width - summary_width - 8, top), summary, fill=TEXT, font=font)


def month_columns(year: int, rows: int) -> dict[int, int]:
    start = date(year, 1, 1)
    return {
        month: (date(year, month, 1) - start).days // rows
        for month in range(1, 13)
    }


def draw_year(
    draw: ImageDraw.ImageDraw,
    *,
    commits: list[Commit],
    completed: set[str],
    top: int,
    grid_left: int,
    rows: int,
    columns: int,
    cell_size: int,
    gap: int,
) -> None:
    stride = cell_size + gap
    cells = rows * columns
    coverage = aggregate_coverage(commits, completed, cells)
    month_font = ImageFont.load_default(size=10)
    year_font = ImageFont.load_default(size=14)
    percent_font = ImageFont.load_default(size=10)
    grid_top = top + 15
    grid_height = rows * stride - gap
    year = commits[0].year
    processed = sum(commit.hash in completed for commit in commits)

    draw.text(
        (8, grid_top + grid_height // 2 - 15),
        str(year),
        fill=TEXT,
        font=year_font,
    )
    draw.text(
        (8, grid_top + grid_height // 2 + 2),
        f"{processed / len(commits):.0%}",
        fill=MUTED_TEXT,
        font=percent_font,
    )

    for month, column in month_columns(year, rows).items():
        draw.text(
            (grid_left + column * stride, top),
            calendar.month_abbr[month],
            fill=MUTED_TEXT,
            font=month_font,
        )

    for index, value in enumerate(coverage):
        column = index // rows
        row = index % rows
        x = grid_left + column * stride
        y = grid_top + row * stride
        draw.rounded_rectangle(
            (x, y, x + cell_size - 1, y + cell_size - 1),
            radius=max(2, cell_size // 4),
            fill=coverage_color(value),
        )


def render(
    yearly_commits: dict[int, list[Commit]],
    *,
    completed: set[str],
    rows: int,
    columns: int,
    cell_size: int,
    gap: int,
    output: Path,
    legend: bool,
) -> None:
    padding = 8
    left_margin = 62
    stride = cell_size + gap
    grid_width = columns * stride - gap
    grid_height = rows * stride - gap
    header_height = 28 if legend else 4
    panel_height = 15 + grid_height + 15
    width = left_margin + grid_width + padding
    height = header_height + len(yearly_commits) * panel_height + padding
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)

    all_commits = [
        commit for commits in yearly_commits.values() for commit in commits
    ]
    processed = sum(commit.hash in completed for commit in all_commits)
    if legend:
        draw_key(
            draw,
            top=8,
            left=left_margin,
            width=width,
            processed=processed,
            total=len(all_commits),
        )

    for panel, commits in enumerate(yearly_commits.values()):
        draw_year(
            draw,
            commits=commits,
            completed=completed,
            top=header_height + panel * panel_height,
            grid_left=left_margin,
            rows=rows,
            columns=columns,
            cell_size=cell_size,
            gap=gap,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main() -> None:
    args = parse_args()
    repository = args.nixpkgs.resolve()
    database = args.database.resolve()
    if not repository.is_dir():
        raise SystemExit(f"Nixpkgs checkout does not exist: {repository}")
    if not database.is_file():
        raise SystemExit(f"database does not exist: {database}")

    tip = resolve_tip(repository, args.tip)
    commits = sampled_commits(repository, tip, args.interval)
    if not commits:
        raise SystemExit("the selected history contains no commits")
    completed = completed_commits(database)
    yearly_commits: dict[int, list[Commit]] = defaultdict(list)
    for commit in commits:
        yearly_commits[commit.year].append(commit)

    render(
        dict(yearly_commits),
        completed=completed,
        rows=args.rows,
        columns=args.columns,
        cell_size=args.cell_size,
        gap=args.gap,
        output=args.output,
        legend=not args.no_legend,
    )

    processed = sum(commit.hash in completed for commit in commits)
    selected_hashes = {commit.hash for commit in commits}
    unmatched = len(completed - selected_hashes)
    print(f"Wrote {args.output.resolve()}")
    ratio = processed / len(commits)
    print(f"Processed: {processed:,} / {len(commits):,} ({ratio:.1%})")
    if unmatched:
        print(f"Stored rows outside the selected sampled history: {unmatched:,}")


if __name__ == "__main__":
    main()
