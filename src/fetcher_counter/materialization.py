# Raw Git paths must remain bytes, so pathlib's text-only path operations are
# deliberately unsuitable in the filesystem layer.
# ruff: noqa: PTH101, PTH102, PTH105, PTH106, PTH108, PTH118, PTH120, PTH211

import contextlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from loguru import logger

from fetcher_counter.history import checkout, run_git

STATE_VERSION = 1
STATE_DIRECTORY = "materialization-state"
REGULAR_MODES = {b"100644", b"100755"}
SYMLINK_MODE = b"120000"
GITLINK_MODE = b"160000"
SUPPORTED_MODES = REGULAR_MODES | {SYMLINK_MODE}
ZERO_MODES = {b"000000"}


class MaterializationError(RuntimeError):
    pass


class NativeCheckoutRequiredError(MaterializationError):
    pass


@dataclass(frozen=True, slots=True)
class TreeDelta:
    old_mode: bytes
    new_mode: bytes
    old_object: bytes
    new_object: bytes
    status: bytes
    path: bytes


@dataclass(frozen=True, slots=True)
class MaterializationState:
    repository: str
    worktree: str
    current_commit: str | None
    native_commit: str | None
    dirty: bool
    version: int = STATE_VERSION


def _valid_object_id(value: bytes) -> bool:
    return len(value) in {40, 64} and all(
        byte in b"0123456789abcdef" for byte in value
    )


def _validate_path(path: bytes) -> None:
    if not path or path.startswith(b"/"):
        raise MaterializationError(
            "git returned an invalid absolute or empty path"
        )
    components = path.split(b"/")
    if any(component in {b"", b".", b".."} for component in components):
        raise MaterializationError("git returned an unsafe path")


def parse_raw_tree_delta(output: bytes) -> list[TreeDelta]:
    if not output:
        return []
    entries = output.split(b"\0")
    if entries[-1] != b"":
        raise MaterializationError("git diff-tree returned a truncated path")
    _ = entries.pop()
    if len(entries) % 2:
        raise MaterializationError("git diff-tree returned an incomplete record")

    deltas: list[TreeDelta] = []
    for header, path in zip(entries[::2], entries[1::2], strict=True):
        if not header.startswith(b":"):
            raise MaterializationError("git diff-tree returned an invalid header")
        fields = header[1:].split(b" ")
        if len(fields) != 5:
            raise MaterializationError("git diff-tree returned invalid fields")
        old_mode, new_mode, old_object, new_object, status = fields
        if (
            len(old_mode) != 6
            or len(new_mode) != 6
            or not old_mode.isdigit()
            or not new_mode.isdigit()
            or not _valid_object_id(old_object)
            or not _valid_object_id(new_object)
        ):
            raise MaterializationError("git diff-tree returned invalid metadata")
        if status not in {b"A", b"D", b"M", b"T"}:
            raise MaterializationError(
                "git diff-tree returned an unexpected change status"
            )
        if old_mode not in ZERO_MODES | SUPPORTED_MODES | {GITLINK_MODE}:
            raise NativeCheckoutRequiredError(
                f"unsupported old tree mode {old_mode.decode(errors='replace')}"
            )
        if new_mode not in ZERO_MODES | SUPPORTED_MODES | {GITLINK_MODE}:
            raise NativeCheckoutRequiredError(
                f"unsupported new tree mode {new_mode.decode(errors='replace')}"
            )
        if GITLINK_MODE in {old_mode, new_mode}:
            raise NativeCheckoutRequiredError(
                "gitlink transition requires native checkout"
            )
        _validate_path(path)
        deltas.append(
            TreeDelta(
                old_mode=old_mode,
                new_mode=new_mode,
                old_object=old_object,
                new_object=new_object,
                status=status,
                path=path,
            )
        )
    return deltas


def parse_batch_blobs(output: bytes, objects: list[bytes]) -> dict[bytes, bytes]:
    position = 0
    blobs: dict[bytes, bytes] = {}
    try:
        for expected in objects:
            header_end = output.index(b"\n", position)
            header = output[position:header_end]
            if header == expected + b" missing":
                raise MaterializationError(
                    f"git cat-file could not find object {expected.decode()}"
                )
            object_id, kind, size_raw = header.split(b" ")
            if object_id != expected or not _valid_object_id(object_id):
                raise ValueError("unexpected object id")  # noqa: TRY301
            if kind != b"blob" or not size_raw.isdigit():
                raise ValueError("unexpected object type or size")  # noqa: TRY301
            size = int(size_raw)
            start = header_end + 1
            end = start + size
            if end > len(output) or output[end : end + 1] != b"\n":
                raise ValueError("truncated object body")  # noqa: TRY301
            blobs[object_id] = output[start:end]
            position = end + 1
    except (ValueError, UnicodeDecodeError) as error:
        if isinstance(error, MaterializationError):
            raise
        raise MaterializationError(
            "git cat-file returned malformed output"
        ) from error
    if position != len(output):
        raise MaterializationError("git cat-file returned trailing output")
    return blobs


async def read_target_blobs(
    repository: Path,
    deltas: list[TreeDelta],
) -> dict[bytes, bytes]:
    objects = list(
        dict.fromkeys(
            delta.new_object
            for delta in deltas
            if delta.new_mode in SUPPORTED_MODES
        )
    )
    if not objects:
        return {}
    output = await run_git(
        repository,
        "cat-file",
        "--batch",
        input=b"\n".join(objects) + b"\n",
    )
    return parse_batch_blobs(output, objects)


def _absolute(root: bytes, relative: bytes) -> bytes:
    return os.path.join(root, relative)


def _remove_path(path: bytes) -> None:
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        return
    if stat.S_ISDIR(mode):
        shutil.rmtree(path)
    else:
        os.unlink(path)


def _ensure_parent(root: bytes, relative: bytes) -> None:
    parent = os.path.dirname(relative)
    if not parent:
        return
    current = root
    for component in parent.split(b"/"):
        current = os.path.join(current, component)
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            os.mkdir(current, 0o777)
            continue
        if not stat.S_ISDIR(mode):
            _remove_path(current)
            os.mkdir(current, 0o777)


def _write_regular(path: bytes, content: bytes, mode: bytes) -> None:
    directory = os.path.dirname(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=b".fetcher-counter-", dir=directory
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            _ = output.write(content)
            output.flush()
        os.chmod(temporary, 0o755 if mode == b"100755" else 0o644)
        _remove_path(path)
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def _write_symlink(path: bytes, target: bytes) -> None:
    directory = os.path.dirname(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=b".fetcher-counter-", dir=directory
    )
    os.close(descriptor)
    os.unlink(temporary)
    try:
        os.symlink(target, temporary)
        _remove_path(path)
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def _remove_empty_parents(root: bytes, relative: bytes) -> None:
    parent = os.path.dirname(relative)
    while parent:
        absolute = _absolute(root, parent)
        try:
            os.rmdir(absolute)
        except OSError:
            return
        parent = os.path.dirname(parent)


def apply_tree_delta(
    worktree: Path,
    deltas: list[TreeDelta],
    blobs: dict[bytes, bytes],
) -> tuple[int, int, int, int, int]:
    root = os.fsencode(worktree)
    removals = sorted(
        (delta.path for delta in deltas if delta.old_mode != b"000000"),
        key=lambda path: (path.count(b"/"), len(path)),
        reverse=True,
    )
    for relative in removals:
        _remove_path(_absolute(root, relative))

    writes = 0
    symlinks = 0
    executable_changes = 0
    byte_count = 0
    for delta in deltas:
        if delta.new_mode == b"000000":
            continue
        content = blobs.get(delta.new_object)
        if content is None:
            raise MaterializationError("target blob was not loaded")
        _ensure_parent(root, delta.path)
        absolute = _absolute(root, delta.path)
        if delta.new_mode in REGULAR_MODES:
            _write_regular(absolute, content, delta.new_mode)
            writes += 1
            byte_count += len(content)
            if (
                delta.old_mode in REGULAR_MODES
                and delta.old_mode != delta.new_mode
            ):
                executable_changes += 1
        elif delta.new_mode == SYMLINK_MODE:
            _write_symlink(absolute, content)
            symlinks += 1
        else:
            raise NativeCheckoutRequiredError("unsupported target mode")

    for delta in deltas:
        if delta.new_mode == b"000000":
            _remove_empty_parents(root, delta.path)
    return len(removals), writes, symlinks, executable_changes, byte_count


def state_path_for_worker(pool_dir: Path, index: int) -> Path:
    return pool_dir / STATE_DIRECTORY / f"worker-{index}.json"


def _canonical(path: Path) -> str:
    return str(path.resolve(strict=False))


def read_state(
    state_path: Path,
    repository: Path,
    worktree: Path,
) -> MaterializationState | None:
    try:
        raw = json.loads(state_path.read_text())
        state = MaterializationState(**raw)
    except OSError, TypeError, ValueError, json.JSONDecodeError:
        return None
    if (
        state.version != STATE_VERSION
        or state.repository != _canonical(repository)
        or state.worktree != _canonical(worktree)
    ):
        return None
    return state


def marker_allows_managed_recovery(
    state_path: Path,
    repository: Path,
    worktree: Path,
) -> bool:
    if not state_path.is_file():
        return False
    try:
        raw = json.loads(state_path.read_text())
    except OSError, ValueError, json.JSONDecodeError:
        return True
    if not isinstance(raw, dict):
        return False
    marker = cast("dict[str, object]", raw)
    return marker.get("repository") == _canonical(repository) and marker.get(
        "worktree"
    ) == _canonical(worktree)


@dataclass(slots=True)
class MaterializedWorktree:
    repository: Path
    path: Path
    state_path: Path | None = None
    checkout_function: Callable[[Path, str], Awaitable[None]] = checkout
    current_commit: str | None = None
    native_commit: str | None = None
    recovery_required: bool = False

    def __post_init__(self) -> None:
        if self.state_path is None:
            return
        state = read_state(self.state_path, self.repository, self.path)
        if state is None:
            self.recovery_required = self.state_path.exists()
            return
        self.current_commit = state.current_commit
        self.native_commit = state.native_commit
        self.recovery_required = state.dirty
        if state.dirty:
            self.current_commit = None
            self.native_commit = None

    def _write_state(self, *, dirty: bool) -> None:
        if self.state_path is None:
            return
        state = MaterializationState(
            repository=_canonical(self.repository),
            worktree=_canonical(self.path),
            current_commit=self.current_commit,
            native_commit=self.native_commit,
            dirty=dirty,
        )
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.state_path.name}.",
            dir=self.state_path.parent,
            text=True,
        )
        try:
            with os.fdopen(descriptor, "w") as output:
                json.dump(
                    {
                        "repository": state.repository,
                        "worktree": state.worktree,
                        "current_commit": state.current_commit,
                        "native_commit": state.native_commit,
                        "dirty": state.dirty,
                        "version": state.version,
                    },
                    output,
                    sort_keys=True,
                )
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.state_path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)
            raise

    async def native_checkout(self, commit: str) -> None:
        self.recovery_required = True
        self._write_state(dirty=True)
        if self.state_path is not None:
            _ = await run_git(self.path, "clean", "-ffdx")
        await self.checkout_function(self.path, commit)
        head = (await run_git(self.path, "rev-parse", "HEAD")).decode().strip()
        if head != commit:
            raise MaterializationError(
                f"native checkout selected {head} instead of {commit}"
            )
        status = await run_git(
            self.path,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignored=matching",
        )
        if status:
            raise MaterializationError("native checkout left the worktree dirty")
        self.current_commit = commit
        self.native_commit = commit
        self.recovery_required = False
        self._write_state(dirty=False)

    async def materialize(self, commit: str) -> None:
        if self.current_commit is None or self.recovery_required:
            raise NativeCheckoutRequiredError("no trusted materialized base")
        if self.current_commit == commit:
            return
        base = self.current_commit
        self._write_state(dirty=True)
        try:
            output = await run_git(
                self.repository,
                "diff-tree",
                "--raw",
                "-z",
                "--no-renames",
                "-r",
                base,
                commit,
            )
            deltas = parse_raw_tree_delta(output)
            blobs = await read_target_blobs(self.repository, deltas)
            removed, writes, symlinks, modes, bytes_written = apply_tree_delta(
                self.path,
                deltas,
                blobs,
            )
        except BaseException:
            self.recovery_required = True
            raise
        self.current_commit = commit
        self.recovery_required = False
        self._write_state(dirty=False)
        logger.debug(
            "Materialized {} to {}: {} removals, {} files, {} symlinks, "
            + "{} mode changes, {} bytes",
            base,
            commit,
            removed,
            writes,
            symlinks,
            modes,
            bytes_written,
        )

    async def restore_pristine(self) -> None:
        if self.current_commit is not None and (
            self.recovery_required or self.current_commit != self.native_commit
        ):
            await self.native_checkout(self.current_commit)
