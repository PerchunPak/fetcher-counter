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
import os
import sqlite3
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from functools import cache, partial
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

Font = ImageFont.FreeTypeFont | ImageFont.ImageFont

EPOCH_ORDINAL = date(1970, 1, 1).toordinal()
SECONDS_PER_DAY = 86400
DEFAULT_TIP = "refs/fetcher-counter/history-tip"
# Below this many commits the extra Git processes cost more than they save.
PARALLEL_THRESHOLD = 20_000


@dataclass(slots=True)
class YearPanel:
    year: int
    coverage: list[float | None]
    processed: int
    total: int


@dataclass(slots=True)
class _YearTally:
    year: int
    start_day: int
    totals: list[int]
    completed: list[int] = field(default_factory=list)
    commits: int = 0
    processed: int = 0


def default_jobs() -> int:
    available = os.process_cpu_count() or 1
    return max(1, min(available - 1, 12))


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
        default=DEFAULT_TIP,
        help="Git revision to visualize (default: saved Fetcher Counter tip)",
    )
    _ = parser.add_argument(
        "--jobs",
        type=int,
        default=default_jobs(),
        help=f"Parallel Git date lookups (default: {default_jobs()})",
    )
    _ = parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Omit the shared legend and overall summary",
    )
    args = parser.parse_args()
    for name in ("interval", "rows", "columns", "cell_size", "jobs"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.gap < 0:
        parser.error("--gap must not be negative")
    return args


def decode_error(stream: bytes) -> str:
    return stream.decode(errors="replace").strip()


def git_run(
    repository: Path, *arguments: str, stdin: bytes | None = None
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        input=stdin,
    )


def git_output(
    repository: Path, *arguments: str, stdin: bytes | None = None
) -> bytes:
    result = git_run(repository, *arguments, stdin=stdin)
    if result.returncode != 0:
        message = decode_error(result.stderr) or "unknown Git error"
        raise SystemExit(f"git {' '.join(arguments)} failed: {message}")
    return result.stdout


def resolve_tip(repository: Path, requested_tip: str) -> str:
    result = git_run(repository, "rev-parse", "--verify", requested_tip)
    if result.returncode == 0:
        return result.stdout.decode().strip()
    if requested_tip == DEFAULT_TIP:
        return git_output(repository, "rev-parse", "HEAD").decode().strip()
    message = decode_error(result.stderr) or "revision does not exist"
    raise SystemExit(f"cannot resolve --tip {requested_tip!r}: {message}")


def sampled_hashes(repository: Path, tip: str, interval: int) -> list[str]:
    """Oldest-first first-parent hashes, sampled every ``interval`` commits.

    Only hashes are asked for here: making Git format commit dates during the
    traversal is roughly seven times slower, so dates are looked up afterwards
    for the sampled commits alone.
    """
    walk = git_output(
        repository,
        "rev-list",
        "--first-parent",
        "--reverse",
        tip,
    ).split()
    return [line.decode() for line in walk[::interval]]


def _offset_seconds(offset: bytes, known: dict[bytes, int]) -> int:
    """Seconds to add to an epoch timestamp for a raw "±HHMM" Git offset."""
    seconds = known.get(offset)
    if seconds is None:
        sign = -1 if offset[:1] == b"-" else 1
        seconds = sign * (int(offset[1:3]) * 3600 + int(offset[3:5]) * 60)
        known[offset] = seconds
    return seconds


def _lookup_days(repository: Path, chunk: list[str]) -> dict[str, int]:
    output = git_output(
        repository,
        "log",
        "--no-walk=unsorted",
        "--format=%H%x00%cd",
        "--date=raw",
        "--stdin",
        stdin="\n".join(chunk).encode(),
    )
    # `--date=raw` is "<epoch seconds> <±HHMM>"; shifting by the offset keeps the
    # commit's own local calendar day, as an ISO timestamp would. Only a handful
    # of distinct offsets occur, so their arithmetic is memoised.
    offsets: dict[bytes, int] = {}
    days: dict[str, int] = {}
    for line in output.splitlines():
        commit_hash, _, raw_date = line.partition(b"\0")
        timestamp, _, offset = raw_date.partition(b" ")
        seconds = int(timestamp) + _offset_seconds(offset, offsets)
        days[commit_hash.decode()] = seconds // SECONDS_PER_DAY
    return days


def commit_days(repository: Path, hashes: list[str], jobs: int) -> dict[str, int]:
    """Map each hash to its committer-local day, counted from 1970-01-01."""
    if jobs < 2 or len(hashes) < PARALLEL_THRESHOLD:
        days = _lookup_days(repository, hashes)
    else:
        size = math.ceil(len(hashes) / jobs)
        chunks = [
            hashes[start : start + size] for start in range(0, len(hashes), size)
        ]
        days = {}
        with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
            for chunk_days in pool.map(partial(_lookup_days, repository), chunks):
                days.update(chunk_days)
    if len(days) != len(hashes):
        missing = len(hashes) - len(days)
        raise SystemExit(f"git did not report a date for {missing:,} commits")
    return days


def completed_commits(database: Path) -> set[str]:
    uri = f"file:{database.resolve()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True, timeout=5) as connection:
            rows = connection.execute('SELECT "commit" FROM "fetchers"').fetchall()
    except sqlite3.Error as error:
        raise SystemExit(f"cannot read {database}: {error}") from error
    return {row[0] for row in rows}


@cache
def year_of(day: int) -> int:
    return date.fromordinal(EPOCH_ORDINAL + day).year


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


def build_panels(
    hashes: list[str], days: dict[str, int], completed: set[str], cells: int
) -> list[YearPanel]:
    """Bucket every sampled commit into its year grid in a single pass."""
    tallies: dict[int, _YearTally] = {}
    for commit_hash in hashes:
        day = days[commit_hash]
        year = year_of(day)
        tally = tallies.get(year)
        if tally is None:
            length = 366 if calendar.isleap(year) else 365
            if cells < length:
                needed = f"the {year} grid needs {length} cells"
                raise SystemExit(f"{needed}; grid provides {cells}")
            tally = _YearTally(
                year=year,
                start_day=date(year, 1, 1).toordinal() - EPOCH_ORDINAL,
                totals=[0] * cells,
                completed=[0] * cells,
            )
            tallies[year] = tally
        index = day - tally.start_day
        tally.totals[index] += 1
        tally.commits += 1
        if commit_hash in completed:
            tally.completed[index] += 1
            tally.processed += 1

    return [
        YearPanel(
            year=tally.year,
            coverage=[
                done / total if total else None
                for done, total in zip(tally.completed, tally.totals, strict=True)
            ],
            processed=tally.processed,
            total=tally.commits,
        )
        for tally in tallies.values()
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
    panel: YearPanel,
    top: int,
    grid_left: int,
    rows: int,
    cell_size: int,
    gap: int,
    month_font: Font,
    year_font: Font,
    percent_font: Font,
) -> None:
    stride = cell_size + gap
    grid_top = top + 15
    grid_height = rows * stride - gap
    radius = max(2, cell_size // 4)

    draw.text(
        (8, grid_top + grid_height // 2 - 15),
        str(panel.year),
        fill=TEXT,
        font=year_font,
    )
    draw.text(
        (8, grid_top + grid_height // 2 + 2),
        f"{panel.processed / panel.total:.0%}",
        fill=MUTED_TEXT,
        font=percent_font,
    )

    for month, column in month_columns(panel.year, rows).items():
        draw.text(
            (grid_left + column * stride, top),
            calendar.month_abbr[month],
            fill=MUTED_TEXT,
            font=month_font,
        )

    for index, value in enumerate(panel.coverage):
        column, row = divmod(index, rows)
        x = grid_left + column * stride
        y = grid_top + row * stride
        draw.rounded_rectangle(
            (x, y, x + cell_size - 1, y + cell_size - 1),
            radius=radius,
            fill=coverage_color(value),
        )


def render(
    panels: list[YearPanel],
    *,
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
    height = header_height + len(panels) * panel_height + padding
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)

    if legend:
        draw_key(
            draw,
            top=8,
            left=left_margin,
            width=width,
            processed=sum(panel.processed for panel in panels),
            total=sum(panel.total for panel in panels),
        )

    month_font = ImageFont.load_default(size=10)
    year_font = ImageFont.load_default(size=14)
    percent_font = ImageFont.load_default(size=10)
    for index, panel in enumerate(panels):
        draw_year(
            draw,
            panel=panel,
            top=header_height + index * panel_height,
            grid_left=left_margin,
            rows=rows,
            cell_size=cell_size,
            gap=gap,
            month_font=month_font,
            year_font=year_font,
            percent_font=percent_font,
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

    # The database read is independent of the Git work, so it runs alongside it.
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending_completed = pool.submit(completed_commits, database)
        tip = resolve_tip(repository, args.tip)
        hashes = sampled_hashes(repository, tip, args.interval)
        if not hashes:
            raise SystemExit("the selected history contains no commits")
        days = commit_days(repository, hashes, args.jobs)
        completed = pending_completed.result()
    panels = build_panels(hashes, days, completed, args.rows * args.columns)

    render(
        panels,
        rows=args.rows,
        columns=args.columns,
        cell_size=args.cell_size,
        gap=args.gap,
        output=args.output,
        legend=not args.no_legend,
    )

    processed = sum(panel.processed for panel in panels)
    unmatched = len(completed.difference(hashes))
    print(f"Wrote {args.output.resolve()}")
    ratio = processed / len(hashes)
    print(f"Processed: {processed:,} / {len(hashes):,} ({ratio:.1%})")
    if unmatched:
        print(f"Stored rows outside the selected sampled history: {unmatched:,}")


if __name__ == "__main__":
    main()
