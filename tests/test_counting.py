import shutil
from pathlib import Path

import pytest

from fetcher_counter.counting import GrepError, count_fetchers

for command in ("rg",):
    if shutil.which(command) is None:
        raise RuntimeError(f"{command} is required for these tests")


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
