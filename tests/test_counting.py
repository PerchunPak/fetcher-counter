import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

from fetcher_counter import counting
from fetcher_counter.counting import GrepError, count_fetcher, count_fetchers

git_path = shutil.which("git")
if git_path is None:
    raise RuntimeError("git is required for these tests")
GIT: str = git_path
for command in ("fd", "rg", "wc", "xargs"):
    if shutil.which(command) is None:
        raise RuntimeError(f"{command} is required for these tests")


@pytest.fixture
def repository(tmp_path: Path) -> tuple[Path, str]:
    _ = subprocess.run(  # noqa: S603
        [GIT, "init", "-q", str(tmp_path)], check=True
    )
    _ = (tmp_path / "package.nix").write_text(
        "fetchurl\nfetchurl\nfetchFromGitHub\n"
    )
    _ = (tmp_path / "ignored.txt").write_text("fetchurl\n")
    _ = subprocess.run(  # noqa: S603
        [GIT, "-C", str(tmp_path), "add", "."], check=True
    )
    _ = subprocess.run(  # noqa: S603
        [
            GIT,
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "fixture",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    commit = subprocess.check_output(  # noqa: S603
        [GIT, "-C", str(tmp_path), "rev-parse", "HEAD"], text=True
    ).strip()
    return tmp_path, commit


@pytest.mark.asyncio
async def test_count_fetcher_counts_matching_lines(
    repository: tuple[Path, str],
) -> None:
    path, commit = repository

    assert await count_fetcher(path, commit, "fetchurl") == 2
    assert await count_fetcher(path, commit, "missingFetcher") == 0


@pytest.mark.asyncio
async def test_count_fetcher_reports_ripgrep_errors(
    repository: tuple[Path, str],
) -> None:
    path, commit = repository

    with pytest.raises(GrepError, match="unclosed character class"):
        _ = await count_fetcher(path, commit, "[")


@pytest.mark.asyncio
async def test_count_fetchers_caches_files_and_limits_workers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scans = 0
    active = 0
    maximum_active = 0
    cached_files = b"one.nix\0two.nix\0"

    async def fake_find(_repository: Path, _commit: str) -> bytes:
        nonlocal scans
        scans += 1
        return cached_files

    async def fake_count(
        _repository: Path,
        _commit: str,
        fetcher: str,
        *,
        nix_files: bytes | None = None,
    ) -> int:
        nonlocal active, maximum_active
        assert nix_files is cached_files
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return len(fetcher)

    monkeypatch.setattr(counting, "RIPGREP_WORKERS", 2)
    monkeypatch.setattr(counting, "find_nix_files", fake_find)
    monkeypatch.setattr(counting, "count_fetcher", fake_count)

    counts = await count_fetchers(
        tmp_path,
        "abc",
        ["three", "one", "two", "four"],
    )

    assert counts == {"four": 4, "one": 3, "three": 5, "two": 3}
    assert scans == 1
    assert maximum_active == 2


@pytest.mark.asyncio
async def test_count_fetchers_does_not_scan_without_fetchers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fail_if_called(_repository: Path, _commit: str) -> bytes:
        pytest.fail("fd should not run without active fetchers")

    monkeypatch.setattr(counting, "find_nix_files", fail_if_called)

    assert await count_fetchers(tmp_path, "abc", []) == {}
