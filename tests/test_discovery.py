import json
from pathlib import Path

import pytest

from fetcher_counter.discovery import FetcherDiscoveryError, discover_fetchers


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
async def test_discover_fetchers_parses_sorts_and_deduplicates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arguments: tuple[object, ...] = ()
    options: dict[str, object] = {}

    async def create_process(*values: object, **kwargs: object) -> Process:
        nonlocal arguments, options
        arguments = values
        options = kwargs
        return Process(
            0, json.dumps(["fetchzip", "fetchurl", "fetchurl"]).encode()
        )

    monkeypatch.setattr("asyncio.create_subprocess_exec", create_process)
    expression = tmp_path / "get-fetchers.nix"

    result = await discover_fetchers(tmp_path, expression, commit="abc")

    assert result == ["fetchurl", "fetchzip"]
    assert arguments[:6] == (
        "nix-instantiate",
        "--eval",
        "--json",
        "--strict",
        "--argstr",
        "nixpkgsPath",
    )
    assert options["start_new_session"] is True


@pytest.mark.asyncio
async def test_discover_fetchers_reports_evaluation_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def create_process(*_args: object, **_kwargs: object) -> Process:
        return Process(1, stderr=b"evaluation failed")

    monkeypatch.setattr("asyncio.create_subprocess_exec", create_process)

    with pytest.raises(
        FetcherDiscoveryError, match=r"commit abc.*evaluation failed"
    ):
        _ = await discover_fetchers(
            tmp_path, tmp_path / "get-fetchers.nix", commit="abc"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("stdout", [b"not json", b'{"fetchurl": 1}', b"[1]"])
async def test_discover_fetchers_rejects_invalid_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stdout: bytes,
) -> None:
    async def create_process(*_args: object, **_kwargs: object) -> Process:
        return Process(0, stdout=stdout)

    monkeypatch.setattr("asyncio.create_subprocess_exec", create_process)

    with pytest.raises(FetcherDiscoveryError, match="commit abc"):
        _ = await discover_fetchers(
            tmp_path, tmp_path / "get-fetchers.nix", commit="abc"
        )
