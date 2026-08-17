import fcntl
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from fetcher_counter import history
from fetcher_counter.history import (
    GitCommandError,
    WorktreeError,
    WorktreePoolLockedError,
    WorktreeRecord,
    WorktreeRequest,
    parse_worktree_list,
    provision_worktrees,
    worktree_pool_lock,
)
from fetcher_counter.materialization import (
    MaterializedWorktree,
    state_path_for_worker,
)
from tests.conftest import GIT, commit_file, run_git


def porcelain(*records: tuple[str, ...]) -> bytes:
    return b"".join(
        b"".join(entry.encode() + b"\0" for entry in record) + b"\0"
        for record in records
    )


@pytest.fixture
def pool(tmp_path: Path) -> Path:
    return tmp_path / "pool"


async def provision(
    repository: Path,
    pool_dir: Path,
    *requests: WorktreeRequest,
) -> dict[int, Path]:
    return await provision_worktrees(
        repository=repository,
        pool_dir=pool_dir,
        requests=requests,
    )


def test_parses_every_porcelain_record_variant() -> None:
    output = porcelain(
        ("worktree /main repo", "HEAD abc", "branch refs/heads/main"),
        ("worktree /pool/worker-0", "HEAD def", "detached"),
        ("worktree /pool/worker-1", "HEAD 123", "detached", "locked busy"),
        ("worktree /pool/worker-2", "detached", "prunable gitdir is missing"),
        ("worktree /mirror.git", "bare"),
    )

    records = parse_worktree_list(output)

    assert [record.path for record in records] == [
        Path("/main repo"),
        Path("/pool/worker-0"),
        Path("/pool/worker-1"),
        Path("/pool/worker-2"),
        Path("/mirror.git"),
    ]
    assert records[0].attributes["branch"] == "refs/heads/main"
    assert "detached" in records[1].attributes
    assert records[2].attributes["locked"] == "busy"
    assert records[3].attributes["prunable"] == "gitdir is missing"
    assert records[4].attributes["bare"] == ""


def test_parsing_rejects_a_record_without_a_path() -> None:
    with pytest.raises(WorktreeError, match="without a path"):
        _ = parse_worktree_list(porcelain(("HEAD abc", "detached")))


@pytest.mark.asyncio
async def test_creates_worktree_at_the_requested_commit(
    repository: Path,
    pool: Path,
) -> None:
    first = commit_file(repository, 1)
    _ = commit_file(repository, 2)
    pool.mkdir()  # noqa: ASYNC240

    worktrees = await provision(repository, pool, WorktreeRequest(0, first))

    worker = pool / "worker-0"
    assert worktrees == {0: worker}
    assert run_git(worker, "rev-parse", "HEAD") == first
    detached = subprocess.run(  # noqa: S603, ASYNC221
        [GIT, "-C", str(worker), "symbolic-ref", "-q", "HEAD"],
        check=False,
        stdout=subprocess.DEVNULL,
    )
    assert detached.returncode == 1


@pytest.mark.asyncio
async def test_reuses_a_pristine_worktree(
    repository: Path,
    pool: Path,
) -> None:
    first = commit_file(repository, 1)
    second = commit_file(repository, 2)
    pool.mkdir()  # noqa: ASYNC240
    _ = await provision(repository, pool, WorktreeRequest(0, first))

    worktrees = await provision(repository, pool, WorktreeRequest(0, second))

    assert worktrees == {0: pool / "worker-0"}
    assert run_git(pool / "worker-0", "rev-parse", "HEAD") == first


@pytest.mark.asyncio
async def test_rejects_a_path_that_is_not_a_worktree(
    repository: Path,
    pool: Path,
) -> None:
    first = commit_file(repository, 1)
    worker = pool / "worker-0"
    worker.mkdir(parents=True)
    _ = (worker / "keep.txt").write_text("keep")

    with pytest.raises(WorktreeError, match="is not a worktree of"):
        _ = await provision(repository, pool, WorktreeRequest(0, first))

    assert (worker / "keep.txt").read_text() == "keep"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name",
    ["stray.nix", "value.txt", "ignored.nix", "ignored.log"],
)
async def test_rejects_a_worktree_holding_any_state(
    repository: Path,
    pool: Path,
    name: str,
) -> None:
    _ = (repository / ".gitignore").write_text("ignored.*\n")
    _ = subprocess.run(  # noqa: S603, ASYNC221
        [GIT, "-C", str(repository), "add", ".gitignore"], check=True
    )
    first = commit_file(repository, 1)
    pool.mkdir()  # noqa: ASYNC240
    _ = await provision(repository, pool, WorktreeRequest(0, first))
    stray = pool / "worker-0" / name
    _ = stray.write_text("stray")

    with pytest.raises(WorktreeError, match="is not pristine"):
        _ = await provision(repository, pool, WorktreeRequest(0, first))

    assert stray.read_text() == "stray"
    assert (pool / "worker-0" / ".git").exists()


@pytest.mark.asyncio
async def test_reuses_a_recognized_incrementally_materialized_worktree(
    repository: Path,
    pool: Path,
) -> None:
    first = commit_file(repository, 1)
    second = commit_file(repository, 2)
    pool.mkdir()  # noqa: ASYNC240
    worktrees = await provision(repository, pool, WorktreeRequest(0, first))
    worker = worktrees[0]
    materialized = MaterializedWorktree(
        repository,
        worker,
        state_path_for_worker(pool, 0),
    )
    await materialized.native_checkout(first)
    await materialized.materialize(second)

    reused = await provision(repository, pool, WorktreeRequest(0, second))

    assert reused == {0: worker}
    assert (worker / "value.txt").read_text() == "2"
    assert run_git(worker, "rev-parse", "HEAD") == first


@pytest.mark.asyncio
async def test_rejects_dirty_worker_with_identity_mismatched_marker(
    repository: Path,
    pool: Path,
) -> None:
    first = commit_file(repository, 1)
    pool.mkdir()  # noqa: ASYNC240
    worktrees = await provision(repository, pool, WorktreeRequest(0, first))
    worker = worktrees[0]
    _ = (worker / "value.txt").write_text("dirty")
    state_path = state_path_for_worker(pool, 0)
    state_path.parent.mkdir()
    _ = state_path.write_text(
        json.dumps(
            {
                "repository": str(repository / "other"),
                "worktree": str(worker),
                "current_commit": first,
                "native_commit": first,
                "dirty": False,
                "version": 1,
            }
        )
    )

    with pytest.raises(WorktreeError, match="is not pristine"):
        _ = await provision(repository, pool, WorktreeRequest(0, first))

    assert (worker / "value.txt").read_text() == "dirty"


@pytest.mark.asyncio
async def test_rejects_a_worktree_whose_status_fails(
    repository: Path,
    pool: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = commit_file(repository, 1)
    pool.mkdir()  # noqa: ASYNC240
    _ = await provision(repository, pool, WorktreeRequest(0, first))
    original = history._run_git

    async def run_git_or_fail(target: Path, *arguments: str) -> bytes:
        if arguments[0] == "status":
            raise GitCommandError("git status failed: broken repository")
        return await original(target, *arguments)

    monkeypatch.setattr(history, "_run_git", run_git_or_fail)

    with pytest.raises(GitCommandError, match="git status failed"):
        _ = await provision(repository, pool, WorktreeRequest(0, first))


@pytest.mark.asyncio
async def test_rejects_a_removed_worktree_instead_of_recreating_it(
    repository: Path,
    pool: Path,
) -> None:
    first = commit_file(repository, 1)
    pool.mkdir()  # noqa: ASYNC240
    _ = await provision(repository, pool, WorktreeRequest(0, first))
    shutil.rmtree(pool / "worker-0")

    with pytest.raises(WorktreeError, match=r"prunable|registered but missing"):
        _ = await provision(repository, pool, WorktreeRequest(0, first))


def test_rejects_a_registered_worktree_missing_from_disk(tmp_path: Path) -> None:
    missing = tmp_path / "pool" / "worker-0"
    record = WorktreeRecord(path=missing)

    with pytest.raises(WorktreeError, match="git worktree prune"):
        history._validate_registered_worktree(record, missing)


def test_registered_but_missing_paths_still_canonicalize(tmp_path: Path) -> None:
    missing = tmp_path / "link" / ".." / "pool" / "worker-0"

    assert history._canonical(missing) == tmp_path / "pool" / "worker-0"


@pytest.mark.asyncio
@pytest.mark.parametrize("attribute", ["locked", "prunable"])
async def test_rejects_locked_and_prunable_worktrees(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
) -> None:
    pool = tmp_path / "pool"
    worker = pool / "worker-0"
    worker.mkdir(parents=True)

    async def records(_repository: Path) -> dict[Path, WorktreeRecord]:
        return {
            worker: WorktreeRecord(
                path=worker,
                attributes={"worktree": str(worker), attribute: "in use"},
            )
        }

    monkeypatch.setattr(history, "_worktree_records", records)

    with pytest.raises(WorktreeError, match=f"{attribute} .in use."):
        _ = await provision(tmp_path / "repository", pool, WorktreeRequest(0, "c"))


@pytest.mark.asyncio
async def test_rejects_reusing_the_supplied_checkout(
    repository: Path,
    pool: Path,
) -> None:
    first = commit_file(repository, 1)
    pool.mkdir()  # noqa: ASYNC240
    _ = await provision(repository, pool, WorktreeRequest(0, first))
    linked = pool / "worker-0"

    with pytest.raises(WorktreeError, match="supplied Nixpkgs checkout"):
        _ = await provision(linked, pool, WorktreeRequest(0, first))


@pytest.mark.asyncio
async def test_provisions_from_a_linked_worktree(
    repository: Path,
    pool: Path,
) -> None:
    first = commit_file(repository, 1)
    linked = pool.parent / "linked"
    _ = run_git(repository, "worktree", "add", "--detach", str(linked), first)
    pool.mkdir()  # noqa: ASYNC240

    worktrees = await provision(linked, pool, WorktreeRequest(0, first))

    assert run_git(worktrees[0], "rev-parse", "HEAD") == first


@pytest.mark.asyncio
async def test_rejects_assigning_one_worktree_to_two_shards(
    repository: Path,
    pool: Path,
) -> None:
    first = commit_file(repository, 1)
    pool.mkdir()  # noqa: ASYNC240

    with pytest.raises(WorktreeError, match="already assigned to shard 0"):
        _ = await provision(
            repository,
            pool,
            WorktreeRequest(0, first),
            WorktreeRequest(0, first),
        )


@pytest.mark.asyncio
async def test_partial_failure_keeps_created_worktrees(
    repository: Path,
    pool: Path,
) -> None:
    first = commit_file(repository, 1)
    (pool / "worker-1").mkdir(parents=True)
    _ = (pool / "worker-1" / "keep.txt").write_text("keep")

    with pytest.raises(WorktreeError, match="is not a worktree of"):
        _ = await provision(
            repository,
            pool,
            WorktreeRequest(0, first),
            WorktreeRequest(1, first),
        )

    assert run_git(pool / "worker-0", "rev-parse", "HEAD") == first
    assert (pool / "worker-1" / "keep.txt").read_text() == "keep"


def test_pool_lock_creates_the_pool_and_keeps_the_lock_file(pool: Path) -> None:
    with worktree_pool_lock(pool):
        assert pool.is_dir()
        assert (pool / "coordinator.lock").is_file()

    assert (pool / "coordinator.lock").is_file()
    with worktree_pool_lock(pool):
        pass


def test_pool_lock_rejects_a_second_holder(pool: Path) -> None:
    with (
        worktree_pool_lock(pool),
        pytest.raises(WorktreePoolLockedError, match=re.escape(str(pool))),
        worktree_pool_lock(pool),
    ):
        pytest.fail("the pool lock must not be granted twice")


@pytest.mark.parametrize("failing", [True, False])
def test_pool_lock_is_released_on_every_exit(pool: Path, failing: bool) -> None:
    if failing:
        with pytest.raises(RuntimeError, match="boom"), worktree_pool_lock(pool):
            raise RuntimeError("boom")
    else:
        with worktree_pool_lock(pool):
            pass

    with worktree_pool_lock(pool):
        pass


def test_pool_lock_requests_a_nonblocking_exclusive_lock(
    pool: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations: list[int] = []

    def flock(_descriptor: int, operation: int) -> None:
        operations.append(operation)

    monkeypatch.setattr(fcntl, "flock", flock)

    with worktree_pool_lock(pool):
        pass

    assert operations == [fcntl.LOCK_EX | fcntl.LOCK_NB]


def test_pool_lock_closes_the_descriptor_when_locking_fails(
    pool: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptors: list[int] = []

    def flock(descriptor: int, _operation: int) -> None:
        descriptors.append(descriptor)
        raise BlockingIOError

    monkeypatch.setattr(fcntl, "flock", flock)

    with pytest.raises(WorktreePoolLockedError), worktree_pool_lock(pool):
        pytest.fail("the pool lock must not be granted")

    with pytest.raises(OSError, match="Bad file descriptor"):
        _ = os.fstat(descriptors[0])


def test_pool_lock_rejects_a_pool_path_that_is_a_file(pool: Path) -> None:
    _ = pool.write_text("not a pool")

    with (
        pytest.raises(WorktreeError, match="not a directory"),
        worktree_pool_lock(pool),
    ):
        pytest.fail("the pool lock must not be granted")

    assert pool.read_text() == "not a pool"


def test_pool_lock_rejects_a_lock_path_that_is_a_directory(pool: Path) -> None:
    (pool / "coordinator.lock").mkdir(parents=True)

    with (
        pytest.raises(WorktreeError, match="lock path is a directory"),
        worktree_pool_lock(pool),
    ):
        pytest.fail("the pool lock must not be granted")

    assert (pool / "coordinator.lock").is_dir()
