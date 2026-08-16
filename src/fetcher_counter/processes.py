import asyncio
from typing import Any

from loguru import logger


async def create_subprocess_exec(
    program: str,
    *arguments: str,
    **options: Any,
) -> asyncio.subprocess.Process:
    """Start a command outside the terminal's foreground process group.

    If cancellation arrives while the child is being created, wait for that
    child and drain it before propagating cancellation. Otherwise the caller
    could lose the only handle to a command that has already started.
    """
    creation = asyncio.create_task(
        asyncio.create_subprocess_exec(
            program,
            *arguments,
            start_new_session=True,
            **options,
        )
    )
    try:
        return await asyncio.shield(creation)
    except asyncio.CancelledError:
        try:
            process = await asyncio.shield(creation)
            _ = await communicate_cancellable(process)
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "Subprocess startup cleanup failed during cancellation: {}",
                error,
            )
        raise


async def communicate_cancellable(
    process: asyncio.subprocess.Process,
    input: bytes | None = None,
) -> tuple[bytes, bytes]:
    """Communicate with `process`, waiting it out when the caller is cancelled.

    Commands run in their own sessions, so terminal Ctrl-C cancels the caller
    without interrupting the child. The original `communicate()` call keeps
    draining its pipes until the command exits, after which cancellation is
    propagated and no subsequent command can start.
    """
    communication = asyncio.create_task(
        process.communicate() if input is None else process.communicate(input)
    )
    try:
        return await asyncio.shield(communication)
    except asyncio.CancelledError:
        try:
            _ = await asyncio.shield(communication)
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "Subprocess cleanup failed during cancellation: {}",
                error,
            )
        raise
