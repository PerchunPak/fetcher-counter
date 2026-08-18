# Tests exercise raw byte paths for parity with Git on Linux.
# ruff: noqa: PTH101, PTH103, PTH115, PTH118, PTH120, PTH123, PTH211

import os
import shutil
import sqlite3
import stat
import subprocess
from pathlib import Path

import pytest

from fetcher_counter import cli, materialization
from fetcher_counter.cli import Config, run
from fetcher_counter.discovery import FetcherDiscoveryError
from fetcher_counter.history import SampledCommit, run_git as git_command
from fetcher_counter.materialization import (
    MaterializationError,
    MaterializedWorktree,
    NativeCheckoutRequiredError,
    TreeDelta,
    parse_batch_blobs,
    parse_raw_tree_delta,
)
from tests.conftest import GIT, run_git


def raw_record(
    old_mode: bytes = b"100644",
    new_mode: bytes = b"100755",
    old_object: bytes = b"a" * 40,
    new_object: bytes = b"b" * 40,
    status: bytes = b"M",
    path: bytes = b"value.txt",
) -> bytes:
    return (
        b":"
        + b" ".join((old_mode, new_mode, old_object, new_object, status))
        + b"\0"
        + path
        + b"\0"
    )


def test_raw_delta_parser_preserves_path_bytes() -> None:
    path = b"dir/space tab\tline\nquote'\\-\xff"

    assert parse_raw_tree_delta(raw_record(path=path)) == [
        TreeDelta(
            old_mode=b"100644",
            new_mode=b"100755",
            old_object=b"a" * 40,
            new_object=b"b" * 40,
            status=b"M",
            path=path,
        )
    ]


@pytest.mark.parametrize(
    "output",
    [
        b":100644 100644 " + b"a" * 40 + b" " + b"b" * 40 + b" M\0path",
        raw_record(status=b"R100"),
        raw_record(old_mode=b"10064x"),
        raw_record(path=b"../escape"),
        b"broken\0path\0",
    ],
)
def test_raw_delta_parser_rejects_malformed_records(output: bytes) -> None:
    with pytest.raises(MaterializationError):
        _ = parse_raw_tree_delta(output)


def test_raw_delta_parser_requests_checkout_for_gitlinks() -> None:
    with pytest.raises(NativeCheckoutRequiredError, match="gitlink"):
        _ = parse_raw_tree_delta(
            raw_record(old_mode=b"000000", new_mode=b"160000", status=b"A")
        )


def test_batch_parser_reads_empty_binary_and_large_blobs() -> None:
    objects = [b"a" * 40, b"b" * 40, b"c" * 40]
    contents = [b"", b"binary\0data", b"x" * 100_000]
    output = b"".join(
        object_id
        + b" blob "
        + str(len(content)).encode()
        + b"\n"
        + content
        + b"\n"
        for object_id, content in zip(objects, contents, strict=True)
    )

    assert parse_batch_blobs(output, objects) == dict(
        zip(objects, contents, strict=True)
    )


@pytest.mark.parametrize(
    "output",
    [
        b"a" * 40 + b" missing\n",
        b"a" * 40 + b" tree 0\n\n",
        b"a" * 40 + b" blob nope\n",
        b"a" * 40 + b" blob 4\nabc\n",
        b"a" * 40 + b" blob 0\n\nextra",
    ],
)
def test_batch_parser_rejects_bad_output(output: bytes) -> None:
    with pytest.raises(MaterializationError):
        _ = parse_batch_blobs(output, [b"a" * 40])


def write_tree(repository: Path, tree: dict[bytes, tuple[str, bytes]]) -> str:
    for child in repository.iterdir():
        if child.name == ".git":
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    root = os.fsencode(repository)
    for relative, (kind, content) in tree.items():
        absolute = os.path.join(root, relative)
        os.makedirs(os.path.dirname(absolute), exist_ok=True)
        if kind == "symlink":
            os.symlink(content, absolute)
        else:
            with open(absolute, "wb") as output:
                _ = output.write(content)
            os.chmod(absolute, 0o755 if kind == "executable" else 0o644)
    _ = subprocess.run(  # noqa: S603
        [GIT, "-C", str(repository), "add", "-A"], check=True
    )
    _ = subprocess.run(  # noqa: S603
        [
            GIT,
            "-C",
            str(repository),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "tree",
        ],
        check=True,
    )
    return run_git(repository, "rev-parse", "HEAD")


def snapshot(path: Path) -> dict[bytes, tuple[str, bytes, bool]]:
    root = os.fsencode(path)
    result: dict[bytes, tuple[str, bytes, bool]] = {}
    for current, directories, files in os.walk(root, followlinks=False):
        directories[:] = [name for name in directories if name != b".git"]
        visible_files = [name for name in files if name != b".git"]
        for name in [*directories, *visible_files]:
            absolute = os.path.join(current, name)
            relative = os.path.relpath(absolute, root)
            mode = os.lstat(absolute).st_mode
            if stat.S_ISDIR(mode):
                continue
            if stat.S_ISLNK(mode):
                result[relative] = ("symlink", os.readlink(absolute), False)
            else:
                with open(absolute, "rb") as source:
                    content = source.read()
                result[relative] = (
                    "file",
                    content,
                    bool(mode & stat.S_IXUSR),
                )
    return result


@pytest.mark.asyncio
async def test_materialization_matches_native_checkout_across_transitions(
    repository: Path,
    tmp_path: Path,
) -> None:
    trees = [
        {b"file": ("file", b"one"), b"gone/deep": ("file", b"remove")},
        {
            b"file": ("executable", b"two\0binary"),
            b"link": ("symlink", b"file"),
            b"new/child": ("file", b"child"),
        },
        {
            b"file/child": ("file", b"directory replaces file"),
            b"link": ("file", b"link becomes file"),
            b"new": ("file", b"directory becomes file"),
        },
        {
            b"file": ("symlink", b"new"),
            b"odd/space tab\tline\n\xff": ("file", b"odd"),
        },
    ]
    commits = [write_tree(repository, tree) for tree in trees]
    worker = tmp_path / "worker"
    control = tmp_path / "control"
    _ = run_git(repository, "worktree", "add", "--detach", str(worker), commits[0])
    _ = run_git(
        repository, "worktree", "add", "--detach", str(control), commits[0]
    )
    materialized = MaterializedWorktree(repository, worker)
    await materialized.native_checkout(commits[0])

    for commit in [*commits[1:], commits[1], commits[0], commits[-1]]:
        await materialized.materialize(commit)
        _ = run_git(control, "checkout", "--detach", "--force", commit)
        assert snapshot(worker) == snapshot(control), commit


def residue_trees() -> list[dict[bytes, tuple[str, bytes]]]:
    """Trees whose later commits add paths absent from the first commit.

    Those paths have no index entry once the worker's index is left at the
    first commit, so `checkout --force` alone cannot remove them.
    """
    return [
        {b"base": ("file", b"base")},
        {
            b"base": ("file", b"base"),
            b"added/deep/leaf": ("file", b"leaf"),
            b"link": ("symlink", b"base"),
        },
        {
            b"base": ("file", b"base"),
            b"added/deep/leaf": ("file", b"changed"),
            b"only-here": ("file", b"solo"),
        },
    ]


@pytest.mark.asyncio
async def test_native_checkout_after_materialization_leaves_no_residue(
    repository: Path,
    tmp_path: Path,
) -> None:
    commits = [write_tree(repository, tree) for tree in residue_trees()]
    worker = tmp_path / "worker"
    control = tmp_path / "control"
    _ = run_git(repository, "worktree", "add", "--detach", str(worker), commits[0])
    _ = run_git(
        repository, "worktree", "add", "--detach", str(control), commits[0]
    )
    materialized = MaterializedWorktree(repository, worker)
    await materialized.native_checkout(commits[0])
    for commit in commits[1:]:
        await materialized.materialize(commit)

    # Back to the native base, which contains none of the added paths.
    await materialized.native_checkout(commits[0])

    _ = run_git(control, "checkout", "--detach", "--force", commits[0])
    assert snapshot(worker) == snapshot(control)
    assert not (worker / "added").exists()
    assert (
        run_git(
            worker,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignored=matching",
        )
        == ""
    )


@pytest.mark.asyncio
async def test_native_checkout_overwrites_unindexed_paths_kept_by_target(
    repository: Path,
    tmp_path: Path,
) -> None:
    """Paths absent from the index but present in the target are not discarded.

    `checkout --force` writes them itself, so the minimal discard set leaves
    them alone; this pins that Git really does overwrite them.
    """
    base = write_tree(repository, {b"base": ("file", b"base")})
    middle = write_tree(
        repository,
        {b"base": ("file", b"base"), b"added/x": ("file", b"from middle")},
    )
    target = write_tree(
        repository,
        {
            b"base": ("file", b"base"),
            b"added/x": ("executable", b"from target"),
        },
    )
    worker = tmp_path / "worker"
    control = tmp_path / "control"
    _ = run_git(repository, "worktree", "add", "--detach", str(worker), base)
    _ = run_git(repository, "worktree", "add", "--detach", str(control), base)
    materialized = MaterializedWorktree(repository, worker)
    await materialized.native_checkout(base)
    await materialized.materialize(middle)

    await materialized.native_checkout(target)

    _ = run_git(control, "checkout", "--detach", "--force", target)
    assert snapshot(worker) == snapshot(control)
    assert (worker / "added" / "x").read_bytes() == b"from target"


@pytest.mark.asyncio
async def test_hot_path_native_checkout_avoids_full_tree_walk(
    repository: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commits = [write_tree(repository, tree) for tree in residue_trees()]
    worker = tmp_path / "worker"
    state_path = tmp_path / "state.json"
    _ = run_git(repository, "worktree", "add", "--detach", str(worker), commits[0])
    materialized = MaterializedWorktree(repository, worker, state_path)
    await materialized.native_checkout(commits[0])
    for commit in commits[1:]:
        await materialized.materialize(commit)

    commands: list[tuple[str, ...]] = []
    real = git_command

    async def recording(
        repo: Path, *arguments: str, input: bytes | None = None
    ) -> bytes:
        commands.append(arguments)
        return await real(repo, *arguments, input=input)

    monkeypatch.setattr(materialization, "run_git", recording)
    await materialized.native_checkout(commits[0])

    assert not [command for command in commands if command[0] == "clean"]
    assert not [command for command in commands if command[0] == "status"]


@pytest.mark.asyncio
async def test_materialization_reconciles_stale_blocking_paths(
    repository: Path,
    tmp_path: Path,
) -> None:
    first = write_tree(repository, {b"base": ("file", b"base")})
    second = write_tree(repository, {b"parent/child": ("file", b"target")})
    worker = tmp_path / "worker"
    _ = run_git(repository, "worktree", "add", "--detach", str(worker), first)
    materialized = MaterializedWorktree(repository, worker)
    await materialized.native_checkout(first)
    _ = (worker / "parent").write_text("stale")

    await materialized.materialize(second)

    assert (worker / "parent" / "child").read_bytes() == b"target"


@pytest.mark.asyncio
async def test_clean_marker_restores_logical_and_native_commits(
    repository: Path,
    tmp_path: Path,
) -> None:
    first = write_tree(repository, {b"value": ("file", b"one")})
    second = write_tree(repository, {b"value": ("file", b"two")})
    worker = tmp_path / "worker"
    state_path = tmp_path / "state.json"
    _ = run_git(repository, "worktree", "add", "--detach", str(worker), first)
    original = MaterializedWorktree(repository, worker, state_path)
    await original.native_checkout(first)
    await original.materialize(second)

    restored = MaterializedWorktree(repository, worker, state_path)

    assert restored.current_commit == second
    assert restored.native_commit == first
    assert restored.recovery_required is False


@pytest.mark.asyncio
async def test_dirty_marker_recovers_with_clean_native_checkout(
    repository: Path,
    tmp_path: Path,
) -> None:
    first = write_tree(repository, {b"value": ("file", b"one")})
    second = write_tree(repository, {b"value": ("file", b"two")})
    worker = tmp_path / "worker"
    state_path = tmp_path / "state.json"
    _ = run_git(repository, "worktree", "add", "--detach", str(worker), first)
    original = MaterializedWorktree(repository, worker, state_path)
    await original.native_checkout(first)
    original._write_state(dirty=True)
    _ = (worker / "value").write_text("interrupted")
    _ = (worker / "stale.nix").write_text("stale")

    recovered = MaterializedWorktree(repository, worker, state_path)
    await recovered.native_checkout(second)

    assert (worker / "value").read_bytes() == b"two"
    assert not (worker / "stale.nix").exists()
    assert run_git(worker, "status", "--porcelain", "--ignored=matching") == ""
    assert recovered.current_commit == second
    assert recovered.native_commit == second


@pytest.mark.asyncio
async def test_restore_recovers_dirty_state_even_when_native_commit_matches(
    repository: Path,
    tmp_path: Path,
) -> None:
    commit = write_tree(repository, {b"value": ("file", b"one")})
    worker = tmp_path / "worker"
    state_path = tmp_path / "state.json"
    _ = run_git(repository, "worktree", "add", "--detach", str(worker), commit)
    materialized = MaterializedWorktree(repository, worker, state_path)
    await materialized.native_checkout(commit)
    materialized.recovery_required = True
    materialized._write_state(dirty=True)
    _ = (worker / "stale.nix").write_text("stale")

    await materialized.restore_pristine()

    assert not (worker / "stale.nix").exists()
    assert materialized.recovery_required is False
    restored = MaterializedWorktree(repository, worker, state_path)
    assert restored.current_commit == commit
    assert restored.native_commit == commit
    assert restored.recovery_required is False


def test_malformed_marker_requires_recovery(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    _ = state_path.write_text("not json")

    materialized = MaterializedWorktree(
        tmp_path / "repository",
        tmp_path / "worker",
        state_path,
    )

    assert materialized.current_commit is None
    assert materialized.recovery_required is True


@pytest.mark.asyncio
async def test_optimized_and_native_runs_store_identical_databases(
    repository: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trees = [
        {
            b"active": ("file", b"fetchurl\n"),
            b"packages.nix": ("file", b"fetchurl\nfetchurl\n"),
        },
        {
            b"active": ("file", b"fetchurl\nfetchzip\n"),
            b"packages.nix": ("file", b"fetchurl fetchzip\nfetchzip\n"),
        },
        {
            b"active": ("file", b"fetchzip\n"),
            b"nested/packages.nix": ("file", b"fetchzip\n"),
        },
        {
            b"active": ("file", b"fail\n"),
            b"packages.nix": ("file", b"ignored\n"),
        },
        {
            b"active": ("file", b"fetchurl\n"),
            b"packages.nix": ("file", b"nothing\n"),
        },
        {
            b"active": ("file", b"fetchurl\nfetchgit\n"),
            b"packages.nix": ("file", b"fetchgit fetchurl\n"),
        },
        {
            b"active": ("file", b"fetchgit\n"),
            b"link.nix": ("symlink", b"packages.nix"),
            b"packages.nix": ("file", b"fetchgit\nfetchgit\n"),
        },
        {
            b"active": ("file", b"fetchurl\nfetchgit\n"),
            b"packages.nix": ("executable", b"fetchurl\nfetchgit\n"),
        },
    ]
    commits = [write_tree(repository, tree) for tree in trees]
    samples = [
        SampledCommit(commit, f"2026-01-{index:02d}T00:00:00Z")
        for index, commit in enumerate(commits, start=1)
    ]

    async def fake_samples(
        _repository: Path,
        *,
        interval: int,
        first_parent: bool,
        completed: set[str] | None = None,
    ) -> list[SampledCommit]:
        assert interval == 1
        assert first_parent is False
        assert completed == set()
        return samples

    async def discover(
        worktree: Path,
        _expression: Path,
        *,
        commit: str,
    ) -> list[str]:
        _ = commit
        active = (worktree / "active").read_text().splitlines()
        if active == ["fail"]:
            raise FetcherDiscoveryError("synthetic failure")
        return active

    async def count(
        worktree: Path,
        _commit: str,
        fetchers: list[str],
    ) -> dict[str, int]:
        paths = list(worktree.rglob("*.nix"))  # noqa: ASYNC240
        lines = [
            line
            for path in paths
            if not path.is_symlink()
            for line in path.read_text().splitlines()
        ]
        return {
            fetcher: sum(fetcher in line for line in lines) for fetcher in fetchers
        }

    monkeypatch.setattr(cli, "sampled_commits", fake_samples)
    monkeypatch.setattr(cli, "discover_fetchers", discover)
    monkeypatch.setattr(cli, "count_fetchers", count)

    async def execute(name: str, native_interval: int) -> Path:
        database = tmp_path / f"{name}.sqlite3"
        await run(
            Config(
                nixpkgs=repository,
                database=database,
                expression=tmp_path / "expression.nix",
                interval=1,
                full_scan_interval=1,
                native_checkout_interval=native_interval,
                workers=2,
                first_parent=False,
                worktrees_dir=tmp_path / f"{name}-pool",
            )
        )
        return database

    control = await execute("control", 1)
    optimized = await execute("optimized", 25)

    def contents(
        database: Path,
    ) -> tuple[dict[str, tuple[object, ...]], list[dict[str, object]]]:
        """Read the table keyed by column name rather than by column order.

        Fetcher columns are added when a shard first discovers the fetcher, so
        their physical order depends on how the shards interleave and is not a
        property of the recorded data. Only the named contents are compared.
        """
        connection = sqlite3.connect(database)
        try:
            schema = {
                str(column[1]): tuple(column[2:])
                for column in connection.execute(
                    'PRAGMA table_info("fetchers")'
                ).fetchall()
            }
            connection.row_factory = sqlite3.Row
            rows = [
                dict(row)
                for row in connection.execute(
                    'SELECT * FROM "fetchers" ORDER BY "commit"'
                ).fetchall()
            ]
        finally:
            connection.close()
        return schema, rows

    assert contents(optimized) == contents(control)
