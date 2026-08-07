# Fetcher Counter project guidance

## Project purpose

This project incrementally counts top-level Nixpkgs fetcher mentions across
sampled first-parent history and stores the results in SQLite. The production
CLI is `fetcher-counter`, implemented in `src/fetcher_counter/`.

## Required workflow

- DO NOT SPAWN AGENTS. DO NOT SPAWN AGENTS. DO NOT SPAWN AGENTS.
- Work directly on `main`; do not create feature branches.
- Do not create temporary checkouts or Git worktrees.
- Make logical changes as separate commits.
- `./nixpkgs` is intentionally mutated by detached historical checkouts. Do not
  restore its original revision after a run or failure.
- Never include the resulting `nixpkgs` Gitlink/working-tree modification in an
  implementation commit unless explicitly requested.
- Prefer to use `pytest -q` instead of `pytest`

## SQLite invariants

- Use exactly one SQLite table named `fetchers`.
- Use one SQLite connection guarded by an `asyncio.Lock`.
- The base columns are `commit`, `date`, and `is_skipped`; each discovered
  fetcher receives a nullable integer column.
- Insert only columns for fetchers active at that revision. Inactive, removed,
  and not-yet-introduced fetchers must remain `NULL`; an active fetcher with no
  matches is zero.
