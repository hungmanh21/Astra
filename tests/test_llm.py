"""Tests for backend/llm.py (PLAN T3.2). A fake `completion` stands in for litellm."""

import asyncio
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from backend.config import LLMSettings
from backend.llm import LLMClient, LLMError, fetch_model_ids

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


def test_failure_is_logged_as_one_short_line_with_details_only_at_debug(caplog):
    completion = FakeCompletion([], fail_on_call=ConnectionError("boom\nsecond line"))
    with caplog.at_level(logging.DEBUG, logger="backend.llm"):
        with pytest.raises(LLMError):
            asyncio.run(collect(LLMClient(settings(), completion=completion)))

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert errors[0].exc_info is None  # no multi-line traceback at the normal level
    assert "ConnectionError: boom second line" in errors[0].getMessage()
    assert "\n" not in errors[0].getMessage()
    debug = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert any(r.exc_info for r in debug)  # the traceback is still there when asked for


def test_a_very_long_error_message_is_cut_in_the_log(caplog):
    completion = FakeCompletion([], fail_on_call=RuntimeError("x" * 5000))
    with caplog.at_level(logging.ERROR, logger="backend.llm"):
        with pytest.raises(LLMError):
            asyncio.run(collect(LLMClient(settings(), completion=completion)))
    assert len(caplog.records[0].getMessage()) < 500


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


# --- model discovery (PLAN T5.4): `hosted_vllm/auto` asks the endpoint which model it serves ---

AUTO = "hosted_vllm/auto"
BASE = "http://gpu-box:8000/v1"


def auto_settings(model: str = AUTO, api_key: str | None = None) -> LLMSettings:
    return LLMSettings(
        model=model,
        api_base=BASE,
        system_prompt="be brief",
        max_history_tokens=1000,
        api_key=api_key,
    )


class FakeLister:
    """Stands in for `fetch_model_ids`. Records its calls; each call yields the next outcome."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)  # a list of ids, or an exception to raise
        self.calls: list[tuple[str, str | None]] = []

    async def __call__(self, api_base: str, api_key: str | None) -> list[str]:
        self.calls.append((api_base, api_key))
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def run_auto(settings_, lister, completion=None, turns: int = 1):
    completion = completion or FakeCompletion([chunk("x")])
    client = LLMClient(settings_, completion=completion, list_models=lister)

    async def go():
        for _ in range(turns):
            await collect(client)

    asyncio.run(go())
    return client, completion


def test_auto_model_is_replaced_by_the_one_the_endpoint_serves():
    lister = FakeLister(["Qwen/Qwen2.5-1.5B-Instruct"])
    _, completion = run_auto(auto_settings(), lister)
    assert completion.kwargs["model"] == "hosted_vllm/Qwen/Qwen2.5-1.5B-Instruct"
    assert completion.kwargs["api_base"] == BASE


def test_discovery_is_given_the_endpoint_and_key():
    lister = FakeLister(["m"])
    run_auto(auto_settings(api_key="sk-secret"), lister)
    assert lister.calls == [(BASE, "sk-secret")]


def test_discovery_keeps_the_provider_prefix_of_the_configured_model():
    _, completion = run_auto(auto_settings("openai/auto"), FakeLister(["my-model"]))
    assert completion.kwargs["model"] == "openai/my-model"


def test_discovery_runs_once_and_is_remembered():
    lister = FakeLister(["m1"])
    _, completion = run_auto(auto_settings(), lister, turns=3)
    assert len(lister.calls) == 1
    assert completion.kwargs["model"] == "hosted_vllm/m1"


def test_the_first_listed_model_wins_and_the_others_are_logged(caplog):
    lister = FakeLister(["first-model", "second-model"])
    with caplog.at_level(logging.INFO, logger="backend.llm"):
        _, completion = run_auto(auto_settings(), lister)
    assert completion.kwargs["model"] == "hosted_vllm/first-model"
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "first-model" in text and "second-model" in text


def test_a_fixed_model_never_triggers_discovery():
    lister = FakeLister(["other"])
    _, completion = run_auto(auto_settings("hosted_vllm/chosen"), lister)
    assert lister.calls == []
    assert completion.kwargs["model"] == "hosted_vllm/chosen"


def test_a_model_name_containing_auto_is_not_the_sentinel():
    lister = FakeLister(["other"])
    _, completion = run_auto(auto_settings("hosted_vllm/auto-model-7b"), lister)
    assert lister.calls == []
    assert completion.kwargs["model"] == "hosted_vllm/auto-model-7b"


def test_model_property_shows_the_configured_then_the_discovered_name():
    lister = FakeLister(["real-model"])
    client = LLMClient(auto_settings(), completion=FakeCompletion([chunk("x")]), list_models=lister)
    assert client.model == AUTO
    asyncio.run(collect(client))
    assert client.model == "hosted_vllm/real-model"


def test_model_property_for_a_fixed_model_is_the_configured_one():
    client = LLMClient(settings(), completion=FakeCompletion([]))
    assert client.model == "test/model"


def test_discovery_failure_becomes_llm_error_and_is_retried_next_turn():
    lister = FakeLister(ConnectionError("secret internals"), ["m1"])
    completion = FakeCompletion([chunk("x")])
    client = LLMClient(auto_settings(), completion=completion, list_models=lister)

    with pytest.raises(LLMError) as info:
        asyncio.run(collect(client))
    assert str(info.value) == "LLM request failed"
    assert isinstance(info.value.__cause__, ConnectionError)
    assert completion.kwargs == {}  # the model was never called
    assert client.model == AUTO  # nothing was remembered

    asyncio.run(collect(client))  # the endpoint is back
    assert completion.kwargs["model"] == "hosted_vllm/m1"
    assert len(lister.calls) == 2


def test_an_endpoint_serving_no_models_is_an_llm_error():
    completion = FakeCompletion([chunk("x")])
    with pytest.raises(LLMError, match="LLM request failed"):
        run_auto(auto_settings(), FakeLister([]), completion)
    assert completion.kwargs == {}


def test_api_key_is_passed_to_the_completion_only_when_configured():
    _, with_key = run_auto(auto_settings("hosted_vllm/m", api_key="sk-1"), FakeLister(["x"]))
    assert with_key.kwargs["api_key"] == "sk-1"
    _, without = run_auto(auto_settings("hosted_vllm/m"), FakeLister(["x"]))
    assert "api_key" not in without.kwargs


# --- fetch_model_ids against a real local HTTP server ---


class ModelsServer:
    """A throwaway server on 127.0.0.1 that answers GET /v1/models like vLLM does."""

    def __init__(self, body: bytes, status: int = 200) -> None:
        outer = self
        self.requests: list[tuple[str, str | None]] = []  # (path, Authorization header)

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                outer.requests.append((self.path, self.headers.get("Authorization")))
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


def models_body(*ids: str) -> bytes:
    return json.dumps(
        {"object": "list", "data": [{"id": i, "object": "model"} for i in ids]}
    ).encode()


def test_fetch_model_ids_returns_the_ids_in_order():
    with ModelsServer(models_body("org/a", "org/b")) as server:
        ids = asyncio.run(fetch_model_ids(server.url))
    assert ids == ["org/a", "org/b"]
    assert server.requests[0][0] == "/v1/models"


def test_fetch_model_ids_ignores_a_trailing_slash_on_the_base():
    with ModelsServer(models_body("m")) as server:
        asyncio.run(fetch_model_ids(server.url + "/"))
    assert server.requests[0][0] == "/v1/models"


def test_fetch_model_ids_sends_the_key_only_when_given():
    with ModelsServer(models_body("m")) as server:
        asyncio.run(fetch_model_ids(server.url, "sk-secret"))
        asyncio.run(fetch_model_ids(server.url))
    assert server.requests[0][1] == "Bearer sk-secret"
    assert server.requests[1][1] is None


@pytest.mark.parametrize(
    "body, status",
    [(b"{}", 401), (b"not json", 200), (b'{"data": [{"name": "no-id"}]}', 200)],
)
def test_fetch_model_ids_raises_on_a_bad_answer(body, status):
    with ModelsServer(body, status) as server:
        with pytest.raises(Exception):  # noqa: B017 - the kind is up to urllib and json
            asyncio.run(fetch_model_ids(server.url))


def test_fetch_model_ids_raises_when_nothing_is_listening():
    with ModelsServer(b"") as server:
        dead_url = server.url
    with pytest.raises(OSError):
        asyncio.run(fetch_model_ids(dead_url, timeout_s=2))


def test_fetch_model_ids_does_not_block_the_event_loop():
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    class SlowHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            time.sleep(0.3)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(models_body("m"))

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), SlowHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    async def go():
        task = asyncio.create_task(ticker())
        await fetch_model_ids(f"http://127.0.0.1:{httpd.server_port}/v1")
        task.cancel()

    try:
        asyncio.run(go())
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert ticks >= 10  # the loop kept running during the 0.3 s request
