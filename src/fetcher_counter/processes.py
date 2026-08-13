import asyncio
import contextlib

from loguru import logger

TERMINATE_TIMEOUT = 5.0


async def communicate_cancellable(
    process: asyncio.subprocess.Process,
    *,
    terminate_timeout: float = TERMINATE_TIMEOUT,
) -> tuple[bytes, bytes]:
    """Communicate with `process`, terminating it when the caller is cancelled.

    The piped output keeps being drained while the child is asked to exit: a
    child blocked on a full pipe buffer would otherwise never terminate, so
    abandoning the original `communicate()` call and only awaiting the process
    could hang forever. Only the direct child is signalled; processes it
    spawned itself are not.
    """
    communication = asyncio.create_task(process.communicate())
    try:
        return await asyncio.shield(communication)
    except asyncio.CancelledError:
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
        try:
            _ = await asyncio.wait_for(
                asyncio.shield(communication),
                timeout=terminate_timeout,
            )
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            try:
                _ = await asyncio.shield(communication)
            except Exception as error:  # noqa: BLE001
                logger.warning("Failed to drain killed subprocess: {}", error)
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "Subprocess cleanup failed during cancellation: {}",
                error,
            )
        raise
