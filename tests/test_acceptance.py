"""One long WebSocket session that does what the S5.2 Chrome check did (PLAN_v1 S5.2).

Two fake models stand in for Whisper (all four strategies) and Nemotron (`native`, no
segments). One connection, one server run: turns switch model, strategy and review mode, one
LLM call fails in the middle, and every turn after it still works.
"""

import dataclasses
from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient

from backend.asr.base import PartialResult
from backend.asr.registry import ASRModelConfig
from backend.main import create_app
from tests.fakes import FakeStreamingASR
from tests.helpers import FakeLLM, make_settings, pcm
from tests.test_edge_cases import receive_within, send_frames

WHISPER = "whisper-like"
NEMOTRON = "nemotron-like"


class FailsOnce(FakeLLM):
    """A FakeLLM whose `fail_on`th call (1-based) raises before the first delta."""

    def __init__(self, fail_on: int) -> None:
        super().__init__(deltas=("Hi", " there"))
        self.fail_on = fail_on

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        if len(self.calls) + 1 == self.fail_on:
            self.calls.append([dict(m) for m in messages])
            raise RuntimeError("provider down")
        async for delta in super().stream(messages):
            yield delta


@pytest.fixture(autouse=True)
def reset_fakes():
    FakeStreamingASR.reset()
    FakeStreamingASR.partials = [PartialResult("hello", ""), PartialResult("hello wor", "ld")]


def two_model_settings():
    settings = make_settings()
    models = {
        WHISPER: ASRModelConfig(adapter="tests.fakes.FakeFullASR", model_id="fake/whisper"),
        NEMOTRON: ASRModelConfig(adapter="tests.fakes.FakeStreamingASR", model_id="fake/nemotron"),
    }
    return dataclasses.replace(settings, asr_models=models, default_asr_model=WHISPER)


def read_until(ws, *kinds: str) -> list[dict]:
    """Messages up to and including the first one whose type is in `kinds`."""
    seen = []
    while not seen or seen[-1]["type"] not in kinds:
        seen.append(receive_within(ws))
    return seen


def run_turn(ws, model: str, strategy: str, review: bool, text: str | None = None) -> list[dict]:
    """One turn like the browser does it. Returns everything up to the end of the turn."""
    ws.send_json(
        {"type": "start_turn", "asr_model": model, "stream_strategy": strategy, "review": review}
    )
    started = receive_within(ws)
    assert started["type"] == "turn_started", started
    send_frames(ws, pcm(1.2))
    ws.send_json({"type": "end_turn"})
    if review:
        messages = read_until(ws, "transcript")
        ws.send_json({"type": "confirm_turn", "turn_id": started["turn_id"], "text": text})
        messages += read_until(ws, "llm_done", "error")
    else:
        messages = read_until(ws, "llm_done", "error")
    return messages


def test_models_strategies_and_review_switch_in_one_server_run():
    llm = FailsOnce(fail_on=5)
    client = TestClient(create_app(two_model_settings(), llm=llm))
    plan = [
        (WHISPER, "chunked", False),
        (WHISPER, "redecode", False),
        (WHISPER, "trimmed", True),
        (NEMOTRON, "native", False),
        (NEMOTRON, "redecode", True),  # fails in the LLM (5th call), after the transcript
        (WHISPER, "chunked", True),
        (NEMOTRON, "native", True),
        (NEMOTRON, "chunked", False),
    ]
    outcomes = []
    with client.websocket_connect("/ws") as ws:
        info = ws.receive_json()
        assert info["stream_strategies"][WHISPER]["supported"] == [
            "chunked",
            "redecode",
            "trimmed",
            "native",
        ]
        assert info["stream_strategies"][NEMOTRON]["default"] == "native"

        for model, strategy, review in plan:
            messages = run_turn(ws, model, strategy, review, text="edited" if review else None)
            transcript = next(m for m in messages if m["type"] == "transcript")
            assert transcript["stream_strategy"] == strategy
            assert transcript["text"], (model, strategy)
            outcomes.append(messages[-1]["type"])

    # the failing turn ends in a generic error after its transcript, every other turn completes
    assert outcomes == ["llm_done"] * 4 + ["error"] + ["llm_done"] * 3
    assert len(llm.calls) == len(plan)
    # each model's own stream was used only for `native`
    assert len(FakeStreamingASR.streams) == 2


def test_the_conversation_survives_the_failed_turn():
    llm = FailsOnce(fail_on=2)
    client = TestClient(create_app(two_model_settings(), llm=llm))
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        run_turn(ws, NEMOTRON, "native", review=False)
        failed = run_turn(ws, NEMOTRON, "native", review=False)
        assert failed[-1]["type"] == "error" and "provider down" not in failed[-1]["message"]
        ok = run_turn(ws, WHISPER, "chunked", review=False)
        assert ok[-1]["type"] == "llm_done"
    roles = [m["role"] for m in llm.calls[2]]
    # system, first user + reply, the failed turn's user message (kept, no reply), the new one
    assert roles == ["system", "user", "assistant", "user", "user"]
