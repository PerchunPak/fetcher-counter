import asyncio
import json
from pathlib import Path
from typing import cast

from loguru import logger

from fetcher_counter.processes import communicate_cancellable


class FetcherDiscoveryError(RuntimeError):
    pass


async def discover_fetchers(
    nixpkgs: Path,
    expression: Path,
    *,
    commit: str,
) -> list[str]:
    logger.debug(
        "Evaluating fetchers at {} with expression {}", commit, expression
    )
    process = await asyncio.create_subprocess_exec(
        "nix-instantiate",
        "--eval",
        "--json",
        "--strict",
        "--argstr",
        "nixpkgsPath",
        str(nixpkgs),
        str(expression),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await communicate_cancellable(process)
    if process.returncode != 0:
        message = stderr.decode(errors="replace").strip()
        logger.debug(
            "Nix evaluation failed at {} with exit code {}: {}",
            commit,
            process.returncode,
            message,
        )
        raise FetcherDiscoveryError(
            f"failed to discover fetchers at commit {commit}: {message}"
        )

    try:
        value = cast("object", json.loads(stdout))
    except json.JSONDecodeError as error:
        raise FetcherDiscoveryError(
            f"Nix returned invalid JSON at commit {commit}: {error}"
        ) from error
    if not isinstance(value, list):
        raise FetcherDiscoveryError(
            f"Nix returned an invalid fetcher list at commit {commit}"
        )
    values = cast("list[object]", value)
    if not all(isinstance(fetcher, str) for fetcher in values):
        raise FetcherDiscoveryError(
            f"Nix returned an invalid fetcher list at commit {commit}"
        )
    fetchers = sorted(set(cast("list[str]", values)))
    logger.debug(
        "Discovered {} fetchers at {}: {}", len(fetchers), commit, fetchers
    )
    return fetchers
