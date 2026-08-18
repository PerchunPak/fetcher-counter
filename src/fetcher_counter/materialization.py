# Raw Git paths must remain bytes, so pathlib's text-only path operations are
# deliberately unsuitable in the filesystem layer.
# ruff: noqa: PTH101, PTH102, PTH105, PTH106, PTH108, PTH118, PTH120, PTH211

import asyncio
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
# Blob contents are held in memory while they are written, and every shard
# shares one process, so they are fetched in size-bounded batches rather than
# in one unbounded `cat-file` call. An 84 MB transition previously peaked at
# 365 MB resident, evicting the page cache that discovery and counting rely on.
MATERIALIZATION_BATCH_BYTES = 8 * 1024 * 1024


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


def parse_batch_sizes(output: bytes, objects: list[bytes]) -> dict[bytes, int]:
    """Parse `git cat-file --batch-check` output into declared blob sizes."""
    lines = output.split(b"\n")
    if lines and lines[-1] == b"":
        _ = lines.pop()
    if len(lines) != len(objects):
        raise MaterializationError(
            "git cat-file returned the wrong number of object records"
        )
    sizes: dict[bytes, int] = {}
    for line, expected in zip(lines, objects, strict=True):
        if line == expected + b" missing":
            raise MaterializationError(
                f"git cat-file could not find object {expected.decode()}"
            )
        fields = line.split(b" ")
        if len(fields) != 3:
            raise MaterializationError("git cat-file returned a malformed record")
        object_id, kind, size_raw = fields
        if object_id != expected or kind != b"blob" or not size_raw.isdigit():
            raise MaterializationError(
                "git cat-file returned an unexpected object record"
            )
        sizes[object_id] = int(size_raw)
    return sizes


def target_objects(deltas: list[TreeDelta]) -> list[bytes]:
    """List the distinct blobs a transition needs, in first-use order."""
    return list(
        dict.fromkeys(
            delta.new_object
            for delta in deltas
            if delta.new_mode in SUPPORTED_MODES
        )
    )


async def read_object_sizes(
    repository: Path,
    objects: list[bytes],
) -> dict[bytes, int]:
    if not objects:
        return {}
    output = await run_git(
        repository,
        "cat-file",
        "--batch-check",
        input=b"\n".join(objects) + b"\n",
    )
    return await asyncio.to_thread(parse_batch_sizes, output, objects)


def plan_object_batches(
    objects: list[bytes],
    sizes: dict[bytes, int],
    budget: int,
) -> list[list[bytes]]:
    """Group `objects` into batches of at most `budget` bytes of content.

    An object larger than the budget forms a batch of its own, because it
    still has to be read whole.
    """
    batches: list[list[bytes]] = []
    current: list[bytes] = []
    current_bytes = 0
    for object_id in objects:
        size = sizes[object_id]
        if current and current_bytes + size > budget:
            batches.append(current)
            current = []
            current_bytes = 0
        current.append(object_id)
        current_bytes += size
    if current:
        batches.append(current)
    return batches


async def read_target_blobs(
    repository: Path,
    deltas: list[TreeDelta],
) -> dict[bytes, bytes]:
    objects = target_objects(deltas)
    if not objects:
        return {}
    output = await run_git(
        repository,
        "cat-file",
        "--batch",
        input=b"\n".join(objects) + b"\n",
    )
    return await asyncio.to_thread(parse_batch_blobs, output, objects)


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


def _deepest_first(paths: list[bytes]) -> list[bytes]:
    return sorted(
        paths,
        key=lambda path: (path.count(b"/"), len(path)),
        reverse=True,
    )


def apply_removals(worktree: Path, deltas: list[TreeDelta]) -> int:
    """Remove every old side, deepest paths first.

    Removals must all precede writes so a path that changes type, or a
    directory that becomes a file, never collides with its own replacement.
    Emptied parents are pruned only after the writes, by
    `prune_empty_parents()`, because a write may repopulate them.
    """
    root = os.fsencode(worktree)
    removals = _deepest_first(
        [delta.path for delta in deltas if delta.old_mode != b"000000"]
    )
    for relative in removals:
        _remove_path(_absolute(root, relative))
    return len(removals)


def discard_paths(worktree: Path, paths: list[bytes]) -> int:
    """Remove `paths` and any directories they leave empty."""
    root = os.fsencode(worktree)
    ordered = _deepest_first(paths)
    for relative in ordered:
        _remove_path(_absolute(root, relative))
    for relative in ordered:
        _remove_empty_parents(root, relative)
    return len(ordered)


def apply_writes(
    worktree: Path,
    deltas: list[TreeDelta],
    blobs: dict[bytes, bytes],
) -> tuple[int, int, int, int]:
    root = os.fsencode(worktree)
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
    return writes, symlinks, executable_changes, byte_count


def prune_empty_parents(worktree: Path, deltas: list[TreeDelta]) -> None:
    root = os.fsencode(worktree)
    for delta in deltas:
        if delta.new_mode == b"000000":
            _remove_empty_parents(root, delta.path)


def apply_tree_delta(
    worktree: Path,
    deltas: list[TreeDelta],
    blobs: dict[bytes, bytes],
) -> tuple[int, int, int, int, int]:
    """Apply a whole delta in one call, in the required stage order."""
    removed = apply_removals(worktree, deltas)
    writes, symlinks, executable_changes, byte_count = apply_writes(
        worktree, deltas, blobs
    )
    prune_empty_parents(worktree, deltas)
    return removed, writes, symlinks, executable_changes, byte_count


def _apply_batch(
    worktree: Path,
    deltas: list[TreeDelta],
    output: bytes,
    objects: list[bytes],
) -> tuple[int, int, int, int]:
    """Parse one `cat-file` batch and write its paths in a single thread hop."""
    blobs = parse_batch_blobs(output, objects)
    return apply_writes(worktree, deltas, blobs)


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
    marker_dirty: bool = False

    def __post_init__(self) -> None:
        if self.state_path is None:
            return
        state = read_state(self.state_path, self.repository, self.path)
        if state is None:
            self.recovery_required = self.state_path.exists()
            self.marker_dirty = self.recovery_required
            return
        self.current_commit = state.current_commit
        self.native_commit = state.native_commit
        self.recovery_required = state.dirty
        self.marker_dirty = state.dirty
        if state.dirty:
            self.current_commit = None
            self.native_commit = None

    def _write_state(self, *, dirty: bool) -> None:
        self.marker_dirty = dirty
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

    async def _mark_dirty(self) -> None:
        """Flag the worker as mid-update, at most once per clean interval.

        Only the clean-to-dirty transition has to reach disk. A dirty marker is
        never trusted, so the commits it records carry no meaning and rewriting
        it per sample fsyncs for nothing: on Btrfs each fsync is a
        filesystem-wide transaction commit, measured at 15ms median under load,
        and all shards share this one process. A crash therefore costs one
        recovery checkout, which is what a dirty marker already means.
        """
        if self.marker_dirty:
            return
        await self._store_state(dirty=True)

    async def _store_state(self, *, dirty: bool) -> None:
        """Persist the marker without stalling the shared event loop.

        The marker is fsynced, and on Btrfs an fsync is a filesystem-wide
        transaction commit. Every shard runs in this one process, so doing
        that inline would block all of them.
        """
        await asyncio.to_thread(self._write_state, dirty=dirty)

    async def _tree_delta(self, base: str, target: str) -> list[TreeDelta]:
        output = await run_git(
            self.repository,
            "diff-tree",
            "--raw",
            "-z",
            "--no-renames",
            "-r",
            base,
            target,
        )
        return parse_raw_tree_delta(output)

    async def _discard_materialized_paths(
        self,
        base: str,
        logical: str,
        target: str,
    ) -> None:
        """Remove worktree paths a native checkout to `target` cannot reach.

        After a successful materialization the worktree matches the logical
        commit exactly and only the index is stale, so `checkout --force`
        already reconciles every path the index knows about, and it also
        writes any path the target contains. That leaves exactly one
        unreachable set: paths present in the logical commit but in neither
        the index -- still at native `base` -- nor `target`. Git holds no
        entry for those at all, so it never considers them.

        Removing precisely that set replaces a full-tree `git clean -ffdx`,
        which costs about 0.6s on a Nixpkgs worktree with 50k files. Keeping
        the set minimal matters as much as avoiding the walk: discarding a
        path the target also holds is safe but makes the checkout write it
        back, which measured 1.56s against 0.35s for the minimal set.
        """
        if logical in {base, target}:
            return
        unindexed = {
            delta.path
            for delta in await self._tree_delta(base, logical)
            if delta.old_mode == b"000000"
        }
        if not unindexed:
            return
        absent_from_target = {
            delta.path
            for delta in await self._tree_delta(logical, target)
            if delta.new_mode == b"000000"
        }
        stale = sorted(unindexed & absent_from_target)
        if stale:
            discarded = await asyncio.to_thread(discard_paths, self.path, stale)
            logger.debug(
                "Discarded {} materialized paths from {} before native checkout",
                discarded,
                self.path,
            )

    async def native_checkout(self, commit: str) -> None:
        # Decide before marking the worker dirty, because the decision depends
        # on the logical state the marker is about to invalidate.
        base = self.native_commit
        logical = self.current_commit
        targeted = (
            not self.recovery_required and base is not None and logical is not None
        )
        self.recovery_required = True
        await self._mark_dirty()
        if targeted:
            assert base is not None
            assert logical is not None
            try:
                await self._discard_materialized_paths(base, logical, commit)
            except Exception as error:  # noqa: BLE001
                logger.warning(
                    "Could not discard materialized paths in {}; "
                    + "falling back to git clean: {}",
                    self.path,
                    error,
                )
                targeted = False
        if not targeted and self.state_path is not None:
            _ = await run_git(self.path, "clean", "-ffdx")
        await self.checkout_function(self.path, commit)
        head = (await run_git(self.path, "rev-parse", "HEAD")).decode().strip()
        if head != commit:
            raise MaterializationError(
                f"native checkout selected {head} instead of {commit}"
            )
        if not targeted:
            # Recovery starts from untrusted worktree contents, so the full
            # walk is worth its cost here. It is skipped on the hot path,
            # where the discarded-path set is exact and covered by tests.
            status = await run_git(
                self.path,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--ignored=matching",
            )
            if status:
                raise MaterializationError(
                    "native checkout left the worktree dirty"
                )
        self.current_commit = commit
        self.native_commit = commit
        self.recovery_required = False
        await self._store_state(dirty=False)

    async def _apply_batches(
        self,
        deltas: list[TreeDelta],
        batches: list[list[bytes]],
    ) -> tuple[int, int, int, int]:
        by_object: dict[bytes, list[TreeDelta]] = {}
        for delta in deltas:
            if delta.new_mode in SUPPORTED_MODES:
                by_object.setdefault(delta.new_object, []).append(delta)
        writes = 0
        symlinks = 0
        modes = 0
        bytes_written = 0
        for batch in batches:
            output = await run_git(
                self.repository,
                "cat-file",
                "--batch",
                input=b"\n".join(batch) + b"\n",
            )
            batch_deltas = [
                delta for object_id in batch for delta in by_object[object_id]
            ]
            written = await asyncio.to_thread(
                _apply_batch,
                self.path,
                batch_deltas,
                output,
                batch,
            )
            writes += written[0]
            symlinks += written[1]
            modes += written[2]
            bytes_written += written[3]
        return writes, symlinks, modes, bytes_written

    async def materialize(
        self,
        commit: str,
        *,
        max_paths: int | None = None,
        max_bytes: int | None = None,
    ) -> None:
        if self.current_commit is None or self.recovery_required:
            raise NativeCheckoutRequiredError("no trusted materialized base")
        if self.current_commit == commit:
            return
        base = self.current_commit

        # Plan the whole transition before touching anything. Every read here
        # is side effect free, so a delta that is rejected or unreadable leaves
        # the worker clean and its trusted base intact, and the caller's native
        # checkout stays on the cheap non-recovery path.
        deltas = await self._tree_delta(base, commit)
        if max_paths is not None and len(deltas) > max_paths:
            raise NativeCheckoutRequiredError(
                f"{len(deltas)} changed paths exceed the incremental"
                + f" limit of {max_paths}"
            )
        objects = target_objects(deltas)
        sizes = await read_object_sizes(self.repository, objects)
        total_bytes = sum(sizes.values())
        if max_bytes is not None and total_bytes > max_bytes:
            raise NativeCheckoutRequiredError(
                f"{total_bytes} delta bytes exceed the incremental"
                + f" limit of {max_bytes}"
            )
        batches = plan_object_batches(objects, sizes, MATERIALIZATION_BATCH_BYTES)

        await self._mark_dirty()
        try:
            removed = await asyncio.to_thread(apply_removals, self.path, deltas)
            writes, symlinks, modes, bytes_written = await self._apply_batches(
                deltas, batches
            )
            await asyncio.to_thread(prune_empty_parents, self.path, deltas)
        except BaseException:
            self.recovery_required = True
            raise
        self.current_commit = commit
        self.recovery_required = False
        logger.debug(
            "Materialized {} to {}: {} removals, {} files, {} symlinks, "
            + "{} mode changes, {} bytes in {} batches",
            base,
            commit,
            removed,
            writes,
            symlinks,
            modes,
            bytes_written,
            len(batches),
        )

    async def restore_pristine(self) -> None:
        if self.current_commit is not None and (
            self.recovery_required
            or self.marker_dirty
            or self.current_commit != self.native_commit
        ):
            await self.native_checkout(self.current_commit)
