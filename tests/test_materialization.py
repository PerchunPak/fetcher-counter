# Tests exercise raw byte paths for parity with Git on Linux.
# ruff: noqa: PTH101, PTH103, PTH115, PTH118, PTH120, PTH123, PTH211

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

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
