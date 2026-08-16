import asyncio
from typing import cast

import pytest
from loguru import logger

from fetcher_counter.processes import (
    communicate_cancellable,
    create_subprocess_exec,
)


class FakeProcess:
    def __init__(
        self,
        *,
        output: tuple[bytes, bytes] = (b"out", b"err"),
        error: Exception | None = None,
    ) -> None:
        self.output: tuple[bytes, bytes] = output
        self.error: Exception | None = error
        self.drained: int = 0
        self.input: bytes | None = None
        self.exited: asyncio.Event = asyncio.Event()

    async def communicate(
        self,
        input: bytes | None = None,
    ) -> tuple[bytes, bytes]:
        self.input = input
        _ = await self.exited.wait()
        self.drained += 1
        if self.error is not None:
            raise self.error
        return self.output


def as_process(process: FakeProcess) -> asyncio.subprocess.Process:
    return cast("asyncio.subprocess.Process", process)


@pytest.mark.asyncio
async def test_create_subprocess_exec_starts_a_new_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    process = FakeProcess()

    async def create_process(
        *arguments: object,
        **options: object,
    ) -> asyncio.subprocess.Process:
        calls.append((arguments, options))
        return as_process(process)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)

    created = await create_subprocess_exec(
        "git",
        "status",
        cwd="repository",
        stdout=asyncio.subprocess.PIPE,
    )

    assert created is as_process(process)
    assert calls == [
        (
            ("git", "status"),
            {
                "cwd": "repository",
                "stdout": asyncio.subprocess.PIPE,
                "start_new_session": True,
            },
        )
    ]


@pytest.mark.asyncio
async def test_cancellation_during_startup_waits_for_created_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = FakeProcess()
    creation_started = asyncio.Event()
    allow_creation = asyncio.Event()

    async def create_process(
        *_arguments: object,
        **_options: object,
    ) -> asyncio.subprocess.Process:
        creation_started.set()
        _ = await allow_creation.wait()
        return as_process(process)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    task = asyncio.create_task(create_subprocess_exec("git", "checkout"))
    _ = await creation_started.wait()

    _ = task.cancel()
    await asyncio.sleep(0)
    assert not task.done()

    allow_creation.set()
    await asyncio.sleep(0)
    assert not task.done()

    process.exited.set()
    with pytest.raises(asyncio.CancelledError):
        _ = await task

    assert process.drained == 1


@pytest.mark.asyncio
async def test_communicate_returns_process_output() -> None:
    process = FakeProcess(output=(b"stdout", b"stderr"))
    process.exited.set()

    assert await communicate_cancellable(as_process(process)) == (
        b"stdout",
        b"stderr",
    )
    assert process.drained == 1


@pytest.mark.asyncio
async def test_communicate_forwards_standard_input() -> None:
    process = FakeProcess(output=(b"stdout", b"stderr"))
    process.exited.set()

    assert await communicate_cancellable(as_process(process), b"input") == (
        b"stdout",
        b"stderr",
    )
    assert process.input == b"input"


@pytest.mark.asyncio
async def test_cancellation_waits_for_process_and_drains_output() -> None:
    process = FakeProcess()
    task = asyncio.create_task(communicate_cancellable(as_process(process)))
    await asyncio.sleep(0)

    _ = task.cancel()
    await asyncio.sleep(0)

    assert not task.done()
    assert process.drained == 0

    process.exited.set()
    with pytest.raises(asyncio.CancelledError):
        _ = await task

    assert process.drained == 1


@pytest.mark.asyncio
async def test_cleanup_failure_is_logged_without_hiding_cancellation() -> None:
    process = FakeProcess(error=RuntimeError("transport broke"))
    task = asyncio.create_task(communicate_cancellable(as_process(process)))
    await asyncio.sleep(0)
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        _ = task.cancel()
        process.exited.set()
        with pytest.raises(asyncio.CancelledError):
            _ = await task
    finally:
        logger.remove(sink_id)

    assert process.drained == 1
    assert any("transport broke" in message for message in messages)
