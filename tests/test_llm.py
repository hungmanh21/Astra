"""Tests for backend/llm.py (PLAN T3.2). A fake `completion` stands in for litellm."""

import asyncio
from types import SimpleNamespace

import pytest

from backend.config import LLMSettings
from backend.llm import LLMClient, LLMError

MESSAGES = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]


def settings(api_base: str | None = None) -> LLMSettings:
    return LLMSettings(
        model="test/model", api_base=api_base, system_prompt="be brief", max_history_tokens=1000
    )


def chunk(content: str | None) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])


class FakeCompletion:
    """Records its call and returns an async iterator over `chunks`."""

    def __init__(self, chunks, fail_on_call=None, fail_after=None):
        self.chunks = chunks
        self.fail_on_call = fail_on_call
        self.fail_after = fail_after  # raise after this many chunks
        self.kwargs: dict = {}

    async def __call__(self, **kwargs):
        self.kwargs = kwargs
        if self.fail_on_call:
            raise self.fail_on_call
        return self._iterate()

    async def _iterate(self):
        for i, c in enumerate(self.chunks):
            if self.fail_after is not None and i == self.fail_after:
                raise RuntimeError("secret internals")
            yield c


async def collect(client: LLMClient) -> list[str]:
    return [delta async for delta in client.stream(MESSAGES)]


def test_stream_yields_the_text_deltas_in_order():
    completion = FakeCompletion([chunk("Hel"), chunk("lo"), chunk(" world")])
    result = asyncio.run(collect(LLMClient(settings(), completion=completion)))
    assert result == ["Hel", "lo", " world"]


def test_request_uses_the_model_messages_and_streaming():
    completion = FakeCompletion([chunk("x")])
    asyncio.run(collect(LLMClient(settings(), completion=completion)))
    assert completion.kwargs["model"] == "test/model"
    assert completion.kwargs["messages"] == MESSAGES
    assert completion.kwargs["stream"] is True


def test_api_base_is_omitted_when_not_configured():
    completion = FakeCompletion([chunk("x")])
    asyncio.run(collect(LLMClient(settings(api_base=None), completion=completion)))
    assert "api_base" not in completion.kwargs


def test_api_base_is_passed_when_configured():
    completion = FakeCompletion([chunk("x")])
    asyncio.run(
        collect(LLMClient(settings(api_base="http://localhost:9/v1"), completion=completion))
    )
    assert completion.kwargs["api_base"] == "http://localhost:9/v1"


def test_empty_and_missing_content_is_skipped():
    usage_only = SimpleNamespace(choices=[])
    completion = FakeCompletion([chunk(None), chunk("A"), chunk(""), usage_only, chunk("B")])
    result = asyncio.run(collect(LLMClient(settings(), completion=completion)))
    assert result == ["A", "B"]


def test_failure_before_the_stream_becomes_llm_error_without_internals():
    completion = FakeCompletion([], fail_on_call=ConnectionError("secret internals"))
    with pytest.raises(LLMError) as info:
        asyncio.run(collect(LLMClient(settings(), completion=completion)))
    assert str(info.value) == "LLM request failed"
    assert "secret" not in str(info.value)
    assert isinstance(info.value.__cause__, ConnectionError)


def test_failure_in_the_middle_keeps_the_deltas_already_yielded():
    completion = FakeCompletion([chunk("A"), chunk("B"), chunk("C")], fail_after=2)
    got: list[str] = []

    async def go():
        async for delta in LLMClient(settings(), completion=completion).stream(MESSAGES):
            got.append(delta)

    with pytest.raises(LLMError, match="LLM request failed"):
        asyncio.run(go())
    assert got == ["A", "B"]


def test_failure_is_logged_with_the_original_traceback(caplog):
    completion = FakeCompletion([], fail_on_call=ConnectionError("boom"))
    with pytest.raises(LLMError):
        asyncio.run(collect(LLMClient(settings(), completion=completion)))
    assert any(r.exc_info for r in caplog.records)


def test_cancellation_is_not_turned_into_llm_error():
    async def slow(**kwargs):
        async def forever():
            yield chunk("first")
            await asyncio.sleep(60)

        return forever()

    async def go():
        got = []

        async def consume():
            async for delta in LLMClient(settings(), completion=slow).stream(MESSAGES):
                got.append(delta)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert got == ["first"]

    asyncio.run(go())


def test_importing_the_module_does_not_import_litellm():
    import subprocess
    import sys

    code = "import sys, backend.llm; sys.exit(1 if 'litellm' in sys.modules else 0)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
