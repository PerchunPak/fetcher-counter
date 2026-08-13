import asyncio
from typing import cast

import pytest
from loguru import logger

from fetcher_counter.processes import communicate_cancellable


class FakeProcess:
    def __init__(
        self,
        *,
        output: tuple[bytes, bytes] = (b"out", b"err"),
        error: Exception | None = None,
        exits_on_terminate: bool = True,
        terminate_error: Exception | None = None,
        kill_error: Exception | None = None,
    ) -> None:
        self.output: tuple[bytes, bytes] = output
        self.error: Exception | None = error
        self.exits_on_terminate: bool = exits_on_terminate
        self.terminate_error: Exception | None = terminate_error
        self.kill_error: Exception | None = kill_error
        self.terminated: int = 0
        self.killed: int = 0
        self.drained: int = 0
        self.exited: asyncio.Event = asyncio.Event()

    async def communicate(self) -> tuple[bytes, bytes]:
        _ = await self.exited.wait()
        self.drained += 1
        if self.error is not None:
            raise self.error
        return self.output

    def terminate(self) -> None:
        self.terminated += 1
        if self.exits_on_terminate:
            self.exited.set()
        if self.terminate_error is not None:
            raise self.terminate_error

    def kill(self) -> None:
        self.killed += 1
        self.exited.set()
        if self.kill_error is not None:
            raise self.kill_error


def as_process(process: FakeProcess) -> asyncio.subprocess.Process:
    return cast("asyncio.subprocess.Process", process)


async def cancel_communication(
    process: FakeProcess,
    *,
    terminate_timeout: float = 5.0,
) -> None:
    task = asyncio.create_task(
        communicate_cancellable(
            as_process(process),
            terminate_timeout=terminate_timeout,
        )
    )
    await asyncio.sleep(0)
    _ = task.cancel()
    with pytest.raises(asyncio.CancelledError):
        _ = await task


@pytest.mark.asyncio
async def test_communicate_returns_process_output() -> None:
    process = FakeProcess(output=(b"stdout", b"stderr"))
    process.exited.set()

    assert await communicate_cancellable(as_process(process)) == (
        b"stdout",
        b"stderr",
    )
    assert process.terminated == 0
    assert process.killed == 0


@pytest.mark.asyncio
async def test_cancellation_terminates_and_drains_prompt_process() -> None:
    process = FakeProcess()

    await cancel_communication(process)

    assert process.terminated == 1
    assert process.killed == 0
    assert process.drained == 1


@pytest.mark.asyncio
async def test_cancellation_kills_process_that_ignores_termination() -> None:
    process = FakeProcess(exits_on_terminate=False)

    await cancel_communication(process, terminate_timeout=0.01)

    assert process.terminated == 1
    assert process.killed == 1
    assert process.drained == 1


@pytest.mark.asyncio
async def test_cancellation_tolerates_already_exited_process() -> None:
    process = FakeProcess(terminate_error=ProcessLookupError())

    await cancel_communication(process)

    assert process.terminated == 1
    assert process.drained == 1


@pytest.mark.asyncio
async def test_cancellation_tolerates_already_exited_kill() -> None:
    process = FakeProcess(
        exits_on_terminate=False,
        kill_error=ProcessLookupError(),
    )

    await cancel_communication(process, terminate_timeout=0.01)

    assert process.killed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("exits_on_terminate", [True, False])
async def test_cleanup_failure_is_logged_without_hiding_cancellation(
    exits_on_terminate: bool,
) -> None:
    process = FakeProcess(
        error=RuntimeError("transport broke"),
        exits_on_terminate=exits_on_terminate,
    )
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        await cancel_communication(process, terminate_timeout=0.01)
    finally:
        logger.remove(sink_id)

    assert any("transport broke" in message for message in messages)
