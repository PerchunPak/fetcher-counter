# Fetcher Counter

Count every mention of top-level Nixpkgs fetchers across the first-parent history
of Nixpkgs and incrementally store the results in SQLite.

## Requirements

- Python 3.14 and [uv](https://docs.astral.sh/uv/)
- Git
- Nix with `nix-instantiate`
- A Nixpkgs checkout at `./nixpkgs` (or a path passed with `--nixpkgs`)

## Usage

```console
uv sync
uv run fetcher-counter
```

The default database is `./data/fetchers.sqlite3`. Paths and the sampling
interval can be overridden:

```console
uv run fetcher-counter \
  --nixpkgs /path/to/nixpkgs \
  --database /path/to/fetchers.sqlite3 \
  --interval 50
```

The process checks out historical revisions directly in the supplied Nixpkgs
checkout. It deliberately does not restore the original revision after success
or failure. Do not run it against a checkout whose current state you need to
preserve.

## History and counting semantics

The program obtains `HEAD` history with `--first-parent --reverse` and samples
commit indices `0, 50, 100, ...`. Anchoring the interval at the oldest commit
means later Nixpkgs commits append samples without changing previously selected
ones. The initial history tip is saved as
`refs/fetcher-counter/history-tip` inside the Nixpkgs repository, so a restart
continues to see the full history even though `HEAD` was left at a historical
revision. Checking out a newer descendant before another run advances that
saved tip.

At each sampled revision, `get-fetchers.nix` evaluates the top-level package set
and selects attribute names matching `fetch.*`. It handles the historical
`default.nix`, `pkgs/top-level/all-packages.nix`, and
`pkgs/system/all-packages.nix` entry points. Revisions predating all three are
represented by an empty active fetcher set. If Nix evaluation fails, the commit
is persisted with `is_skipped = 1` and no fetcher counts, then processing
continues with the next sample.

All active fetchers are counted concurrently with `git grep -F -w -o` over
`*.nix` files. The `-o` output ensures multiple occurrences on one line are
counted separately.

## Database

The database contains exactly one table, `fetchers`:

| commit | date | is_skipped | fetchFromGitHub | fetchFromGitLab | ... |
| --- | --- | ---: | ---: | ---: | --- |
| `553f8e50...` | `2026-01-01T00:00:00+00:00` | 0 | 5000 | 3000 | ... |

`commit` is the primary key. `is_skipped` is one when fetcher discovery failed
and zero for successfully counted revisions. Existing databases are migrated
with zero for their prior rows. Each dynamically discovered fetcher receives a
nullable integer column through SQLite's supported `ALTER TABLE ... ADD COLUMN`
operation. A row only writes columns for fetchers active at that revision, so a
fetcher that has not appeared yet or has since disappeared is `NULL`, not zero.
An active fetcher with no textual matches is stored as zero.

Schema changes and each commit row are written in one transaction. Existing
commit hashes are loaded at startup and skipped, making interrupted runs
resumable without duplicate rows.
