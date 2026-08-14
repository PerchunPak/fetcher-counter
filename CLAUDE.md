# Fetcher Counter project guidance

## Project purpose

This project incrementally counts top-level Nixpkgs fetcher mentions across
sampled first-parent history and stores the results in SQLite. The production
CLI is `fetcher-counter`, implemented in `src/fetcher_counter/`.

## Required workflow

- DO NOT SPAWN AGENTS. DO NOT SPAWN AGENTS. DO NOT SPAWN AGENTS.
- Work directly on `main`; do not create feature branches.
- Do not create development checkouts or Git worktrees for this repository.
  Runtime-managed Nixpkgs worker worktrees are part of the application and must
  not be cleaned, reset, or deleted.
- Make logical changes as separate commits.
- With the default single-worker mode, `./nixpkgs` is intentionally mutated by
  detached historical checkouts. Do not restore its original revision after a
  run or failure.
- Never include the resulting `nixpkgs` Gitlink/working-tree modification in an
  implementation commit unless explicitly requested.
- Prefer to use `pytest -q` instead of `pytest`
- After you've made your changes, make a commit. This will also run all
  required verification checks. Do not run `git push` or open a PR.

## SQLite invariants

- Use exactly one SQLite table named `fetchers`.
- Use one SQLite connection guarded by an `asyncio.Lock`.
- The base columns are `commit`, `date`, and `is_skipped`; each discovered
  fetcher receives a nullable integer column.
- Insert only columns for fetchers active at that revision. Inactive, removed,
  and not-yet-introduced fetchers must remain `NULL`; an active fetcher with no
  matches is zero.
