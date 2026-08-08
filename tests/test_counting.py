import shutil
import subprocess
from pathlib import Path

import pytest

from fetcher_counter.counting import (
    GrepError,
    IncrementalCountError,
    count_fetchers,
    update_fetcher_counts,
)

for command in ("git", "rg"):
    if shutil.which(command) is None:
        raise RuntimeError(f"{command} is required for these tests")

git_path = shutil.which("git")
if git_path is None:
    raise RuntimeError("git is required for these tests")
GIT: str = git_path


class Process:
    def __init__(
        self,
        returncode: int,
        stdout: bytes = b"",
        stderr: bytes = b"",
    ) -> None:
        self.returncode: int = returncode
        self.stdout: bytes = stdout
        self.stderr: bytes = stderr

    async def communicate(self) -> tuple[bytes, bytes]:
        return self.stdout, self.stderr


def initialize_repository(repository: Path) -> None:
    _ = subprocess.run([GIT, "init", "-q", str(repository)], check=True)  # noqa: S603


def commit_all(repository: Path, message: str) -> str:
    _ = subprocess.run(  # noqa: S603
        [GIT, "-C", str(repository), "add", "-A"],
        check=True,
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
            "-q",
            "-m",
            message,
        ],
        check=True,
    )
    return subprocess.check_output(  # noqa: S603
        [GIT, "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
    ).strip()


@pytest.mark.asyncio
async def test_count_fetchers_counts_matching_lines(tmp_path: Path) -> None:
    _ = (tmp_path / "package.nix").write_text(
        "fetchurl\nfetchurl\nfetchFromGitHub\n"
    )
    _ = (tmp_path / "ignored.txt").write_text("fetchurl\n")

    counts = await count_fetchers(
        tmp_path,
        "abc",
        ["fetchurl", "fetchFromGitHub", "missingFetcher"],
    )

    assert counts == {
        "fetchFromGitHub": 1,
        "fetchurl": 2,
        "missingFetcher": 0,
    }


@pytest.mark.asyncio
async def test_count_fetchers_deduplicates_lines_and_handles_overlaps(
    tmp_path: Path,
) -> None:
    _ = (tmp_path / "package.nix").write_text(
        """fetch fetch-scm fetch-scm
fetchurl fetchurl fetchzip
fetchurl
fetch.url
fetchXurl
"""
    )

    counts = await count_fetchers(
        tmp_path,
        "abc",
        ["fetch", "fetch-scm", "fetch.url", "fetchurl", "fetchzip"],
    )

    assert counts == {
        "fetch": 2,
        "fetch-scm": 1,
        "fetch.url": 1,
        "fetchurl": 2,
        "fetchzip": 1,
    }


@pytest.mark.asyncio
async def test_count_fetchers_runs_one_fixed_string_ripgrep_scan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def create_process(
        *arguments: object,
        **options: object,
    ) -> Process:
        calls.append((arguments, options))
        return Process(0, stdout=b"fetch fetch-scm\nfetchurl fetchurl\n")

    monkeypatch.setattr("asyncio.create_subprocess_exec", create_process)

    counts = await count_fetchers(
        tmp_path,
        "abc",
        ["fetchurl", "fetch", "fetch-scm", "fetchurl"],
    )

    assert counts == {"fetch": 1, "fetch-scm": 1, "fetchurl": 1}
    assert len(calls) == 1
    arguments, options = calls[0]
    assert arguments == (
        "rg",
        "--no-filename",
        "--no-line-number",
        "--color=never",
        "--fixed-strings",
        "--word-regexp",
        "-e",
        "fetch",
        "-e",
        "fetch-scm",
        "-e",
        "fetchurl",
        "--glob",
        "*.nix",
        ".",
    )
    assert options["cwd"] == tmp_path


@pytest.mark.asyncio
async def test_count_fetchers_accepts_ripgrep_no_match_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def create_process(
        *_arguments: object,
        **_options: object,
    ) -> Process:
        return Process(1)

    monkeypatch.setattr("asyncio.create_subprocess_exec", create_process)

    assert await count_fetchers(tmp_path, "abc", ["fetchurl"]) == {"fetchurl": 0}


@pytest.mark.asyncio
async def test_count_fetchers_reports_ripgrep_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def create_process(
        *_arguments: object,
        **_options: object,
    ) -> Process:
        return Process(2, stderr=b"ripgrep failed")

    monkeypatch.setattr("asyncio.create_subprocess_exec", create_process)

    with pytest.raises(GrepError, match=r"commit abc.*ripgrep failed"):
        _ = await count_fetchers(tmp_path, "abc", ["fetchurl"])


@pytest.mark.asyncio
async def test_count_fetchers_does_not_scan_without_fetchers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fail_if_called(*_arguments: object, **_options: object) -> Process:
        pytest.fail("ripgrep should not run without active fetchers")

    monkeypatch.setattr("asyncio.create_subprocess_exec", fail_if_called)

    assert await count_fetchers(tmp_path, "abc", []) == {}


@pytest.mark.asyncio
async def test_update_fetcher_counts_applies_hunk_lines_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    output = b"""diff --git a/package.nix b/package.nix
--- a/package.nix
+++ b/package.nix
@@ -1,3 +1,3 @@
-fetch fetch-scm fetch-scm
-fetchurl
---- fetchurl
+fetch-scm
+fetchzip fetchzip
++++ fetch
"""

    async def create_process(
        *arguments: object,
        **options: object,
    ) -> Process:
        calls.append((arguments, options))
        return Process(0, stdout=output)

    monkeypatch.setattr("asyncio.create_subprocess_exec", create_process)

    counts = await update_fetcher_counts(
        tmp_path,
        "newer",
        "older",
        {"fetch": 2, "fetch-scm": 1, "fetchurl": 2, "fetchzip": 0},
    )

    assert counts == {
        "fetch": 3,
        "fetch-scm": 1,
        "fetchurl": 0,
        "fetchzip": 1,
    }
    assert len(calls) == 1
    arguments, options = calls[0]
    assert arguments == (
        "git",
        "-C",
        str(tmp_path),
        "diff",
        "--unified=0",
        "--no-color",
        "--no-ext-diff",
        "--no-renames",
        "--text",
        "newer",
        "older",
        "--",
        "*.nix",
    )
    assert options["stdout"] is not None
    assert options["stderr"] is not None


@pytest.mark.asyncio
async def test_update_fetcher_counts_handles_added_and_deleted_files(
    tmp_path: Path,
) -> None:
    initialize_repository(tmp_path)
    _ = (tmp_path / "shared.nix").write_text("fetchurl\n")
    _ = (tmp_path / "old.nix").write_text("fetch-scm\n")
    older = commit_all(tmp_path, "older")

    (tmp_path / "old.nix").unlink()
    _ = (tmp_path / "new.nix").write_text("fetchzip\n")
    _ = (tmp_path / "shared.nix").write_text("fetchurl\nfetchurl\n")
    newer = commit_all(tmp_path, "newer")

    counts = await update_fetcher_counts(
        tmp_path,
        newer,
        older,
        {"fetch": 0, "fetch-scm": 0, "fetchurl": 2, "fetchzip": 1},
    )

    assert counts == {
        "fetch": 1,
        "fetch-scm": 1,
        "fetchurl": 1,
        "fetchzip": 0,
    }


@pytest.mark.asyncio
async def test_update_fetcher_counts_preserves_counts_without_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def create_process(
        *_arguments: object,
        **_options: object,
    ) -> Process:
        return Process(0)

    monkeypatch.setattr("asyncio.create_subprocess_exec", create_process)

    assert await update_fetcher_counts(
        tmp_path,
        "newer",
        "older",
        {"fetchurl": 3},
    ) == {"fetchurl": 3}


@pytest.mark.asyncio
async def test_update_fetcher_counts_reports_git_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def create_process(
        *_arguments: object,
        **_options: object,
    ) -> Process:
        return Process(128, stderr=b"bad revision")

    monkeypatch.setattr("asyncio.create_subprocess_exec", create_process)

    with pytest.raises(IncrementalCountError, match=r"newer.*older.*bad revision"):
        _ = await update_fetcher_counts(
            tmp_path,
            "newer",
            "older",
            {"fetchurl": 3},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("output", "message"),
    [
        (b"@@ -1 +0,0 @@\n-fetchurl\n", "negative"),
        (b"@@ -1,2 +0,0 @@\n-fetchurl\n", "incomplete"),
    ],
)
async def test_update_fetcher_counts_rejects_invalid_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    output: bytes,
    message: str,
) -> None:
    async def create_process(
        *_arguments: object,
        **_options: object,
    ) -> Process:
        return Process(0, stdout=output)

    monkeypatch.setattr("asyncio.create_subprocess_exec", create_process)

    with pytest.raises(IncrementalCountError, match=message):
        _ = await update_fetcher_counts(
            tmp_path,
            "newer",
            "older",
            {"fetchurl": 0},
        )
