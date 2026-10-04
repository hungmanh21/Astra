"""Tests for WebSocketTransport (PLAN T1.5), using a fake WebSocket object."""

import asyncio
from typing import Any

import pytest
from starlette.websockets import WebSocketDisconnect, WebSocketDisconnected

from backend.transport import WebSocketTransport


class FakeWebSocket:
    """Just the methods WebSocketTransport uses. `events` are returned by receive() in order."""

    def __init__(self, events: list[dict[str, Any]] | None = None) -> None:
        self.events = list(events or [])
        self.sent_json: list[Any] = []
        self.sent_bytes: list[bytes] = []
        self.send_error: Exception | None = None

    async def receive(self) -> dict[str, Any]:
        return self.events.pop(0)

    async def send_json(self, data: Any) -> None:
        if self.send_error:
            raise self.send_error
        self.sent_json.append(data)

    async def send_bytes(self, data: bytes) -> None:
        if self.send_error:
            raise self.send_error
        self.sent_bytes.append(data)


class RecordingHandler:
    def __init__(self, fail_on: str | None = None) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.fail_on = fail_on

    async def on_connect(self) -> None:
        self.calls.append(("connect", None))

    async def on_json(self, raw: str) -> None:
        self.calls.append(("json", raw))
        if self.fail_on == "json":
            raise RuntimeError("bug in handler")

    async def on_audio_frame(self, data: bytes) -> None:
        self.calls.append(("audio", data))

    async def on_disconnect(self) -> None:
        self.calls.append(("disconnect", None))


def text(s: str) -> dict[str, Any]:
    return {"type": "websocket.receive", "text": s, "bytes": None}


def binary(b: bytes) -> dict[str, Any]:
    return {"type": "websocket.receive", "text": None, "bytes": b}


LEFT = {"type": "websocket.disconnect", "code": 1001}


def test_run_dispatches_in_order_and_stops_on_disconnect():
    ws = FakeWebSocket([text('{"type": "end_turn"}'), binary(b"\x01\x02"), text("x"), LEFT])
    handler = RecordingHandler()
    asyncio.run(WebSocketTransport(ws).run(handler))
    assert handler.calls == [
        ("connect", None),
        ("json", '{"type": "end_turn"}'),
        ("audio", b"\x01\x02"),
        ("json", "x"),
        ("disconnect", None),
    ]
    assert ws.events == []  # receive() was not called again after the disconnect event


def test_run_forwards_an_empty_binary_frame():
    ws = FakeWebSocket([binary(b""), LEFT])
    handler = RecordingHandler()
    asyncio.run(WebSocketTransport(ws).run(handler))
    assert ("audio", b"") in handler.calls


def test_run_calls_on_disconnect_even_when_the_handler_raises():
    ws = FakeWebSocket([text("x"), LEFT])
    handler = RecordingHandler(fail_on="json")
    with pytest.raises(RuntimeError, match="bug in handler"):
        asyncio.run(WebSocketTransport(ws).run(handler))
    assert handler.calls[-1] == ("disconnect", None)


def test_send_json_and_send_bytes_forward_to_the_websocket():
    ws = FakeWebSocket()
    transport = WebSocketTransport(ws)

    async def go():
        await transport.send_json({"type": "session"})
        await transport.send_bytes(b"\x00\x01")

    asyncio.run(go())
    assert ws.sent_json == [{"type": "session"}]
    assert ws.sent_bytes == [b"\x00\x01"]


@pytest.mark.parametrize("error", [WebSocketDisconnect(1006), WebSocketDisconnected("closed")])
@pytest.mark.parametrize("method", ["send_json", "send_bytes"])
def test_sending_to_a_closed_socket_raises_connection_error(error, method):
    ws = FakeWebSocket()
    ws.send_error = error
    transport = WebSocketTransport(ws)
    arg = {"type": "x"} if method == "send_json" else b"x"
    with pytest.raises(ConnectionError):
        asyncio.run(getattr(transport, method)(arg))


def test_other_send_errors_are_not_swallowed():
    ws = FakeWebSocket()
    ws.send_error = ValueError("not JSON serializable")
    with pytest.raises(ValueError):
        asyncio.run(WebSocketTransport(ws).send_json({"x": object()}))
