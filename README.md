# Fetcher Counter

Count every mention of top-level Nixpkgs fetchers across the first-parent history
of Nixpkgs and incrementally store the results in SQLite.

## Requirements

- Python 3.14 and [uv](https://docs.astral.sh/uv/)
- Git
- [`ripgrep`](https://github.com/BurntSushi/ripgrep)
- Nix with `nix-instantiate`
- A Nixpkgs checkout at `./nixpkgs` (or a path passed with `--nixpkgs`)

## Usage

```console
uv sync
uv run fetcher-counter
```

The default database is `./data/fetchers.sqlite3`. Paths, the commit sampling
interval, and the full-scan interval can be overridden:

```console
uv run fetcher-counter \
  --nixpkgs /path/to/nixpkgs \
  --database /path/to/fetchers.sqlite3 \
  --interval 50 \
  --full-scan-interval 25 \
  --workers 4 \
  --reverse \
  --worktrees-dir /path/to/worktree-pool \
  --log-level DEBUG
```

`--log-level` accepts `TRACE`, `DEBUG`, `INFO`, `SUCCESS`, `WARNING`, `ERROR`, or
`CRITICAL` and defaults to `INFO`. Each log line is labelled with the shard that
emitted it, or with `main` for the coordinator itself.

With the default `--workers 1`, the process checks out historical revisions
directly in the supplied Nixpkgs checkout. It deliberately does not restore the
original revision after success or failure. Do not run it against a checkout
whose current state you need to preserve.

## Parallel shards

`--workers N` with `N` greater than one splits the samples still to be counted
into `N` contiguous shards and processes them concurrently inside one process.
Splitting the pending work rather than the whole history keeps every worker busy
on a resumed run, where most of the history is already stored; the cost is that
shard boundaries move between runs, so a different sample forfeits its neighbor
each time. Stored rows are keyed by commit, so nothing depends on where the
boundaries fell in an earlier run. The
supplied `--nixpkgs` checkout is then only used to compute history and as the
source for worker worktrees; it is no longer checked out to every historical
revision itself.

Each shard works in a persistent Git worktree under a managed pool directory,
`.<nixpkgs>-fetcher-counter-worktrees` next to the resolved `--nixpkgs` path by
default and overridable with `--worktrees-dir`:

```
<pool>/coordinator.lock
<pool>/worker-0
<pool>/worker-1
```

The pool is locked with an advisory lock on `coordinator.lock` for the whole
run, before history is sampled and before the database is opened. A second
invocation using the same pool fails immediately instead of waiting. The pool
directory and its lock file are created even when no work turns out to be
pending; the lock file stays on disk between runs, because the advisory lock
rather than the file marks ownership.

Worker worktrees are reused across runs, but only when Git reports them as
registered, unlocked, not prunable, and entirely pristine, including untracked
and ignored files. A stray `.nix` file would otherwise be counted after a later
historical checkout un-ignores it. Nothing in the pool is ever cleaned, reset,
or deleted: unexpected state is reported so it can be inspected manually.

Shards share the single database connection, so all writes stay serialized and
each shard's progress is durable on its own. If one shard fails, its siblings
finish first and the failure is reported afterwards. Because the shard that
writes first decides them, the physical order of fetcher columns and the winner
of a case-only collision vary between runs; stored values do not.

Concurrent `fetcher-counter` invocations against the same Nixpkgs repository or
the same database file are unsupported regardless of worktree pool. The pool
lock protects one pool, not a repository or a database.

## History and counting semantics

The program obtains the saved history tip with `--first-parent`. Sampling is
anchored at the oldest commit using indices `0, 50, 100, ...`, but the selected
commits are traversed from newest to oldest by default. `--reverse` traverses the
same selected commits from oldest to newest without changing their stable sample
indices. Pulling new Nixpkgs commits therefore does not shift or reset previously
selected historical commits; new samples are added only when the oldest-anchored
interval reaches them. The initial history
tip is saved as
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

The first revision without a valid adjacent result uses one ripgrep process to
scan all Nix files for every active fetcher as fixed-string, whole-word patterns.
Each matching source line is processed once, and each fetcher is counted at most
once per line even if it appears repeatedly. Each count is therefore the number
of matching lines across Nix files.

Each revision normally reuses the stored counts from the adjacent sample that
precedes it in traversal order: newer under the default order and older with
`--reverse`. A zero-context Git diff transforms those counts from the preceding
tree into counts for the current tree. By default, every 25th sampled position
forces a full ripgrep scan as a periodic correctness checkpoint; this cadence can
be changed with `--full-scan-interval`. The program also falls back to a full
scan when the adjacent row is missing or skipped, the active fetcher set
changed, the sample is first in traversal order, or the diff cannot produce
trustworthy non-negative counts. This lets an interrupted run resume
incrementally from its adjacent completed row while keeping the safety checkpoints
anchored to stable positions in the sampled history.

Every sample keeps the history position that decides its full-scan checkpoint,
regardless of traversal direction or worker count. A pending sample reuses its
immediately preceding neighbor in traversal order, completed or not, exactly as
a sequential run does. The one exception is the first sample of a shard whose
neighbor is itself pending in another shard: that sample uses a full scan,
because reading a row another shard is about to write would make the choice
depend on scheduling order. A boundary sample whose neighbor was already stored
by an earlier run still counts incrementally.

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
An active fetcher with no textual matches is stored as zero. SQLite column names
are case-insensitive, while Nix attribute names are not. Case-only collisions
therefore receive a deterministic `__case_collision_N` suffix; for example,
`fetchFromGithub` is stored as `fetchFromGithub__case_collision_1` when
`fetchFromGitHub` already exists.

Schema changes and each commit row are written in one transaction. Existing
commit hashes are loaded at startup and skipped, making interrupted runs
resumable without duplicate rows.
