"""The seam between the network and the session logic (SPEC 6.3 "WebRTC seam", PLAN T1.5).

`Session` only knows the `Transport` it sends through and the `SessionHandler` methods the
transport calls. It never imports FastAPI. A WebRTC transport can later replace
`WebSocketTransport` without touching session.py.
"""

from typing import Any, Protocol

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketDisconnected

# Sending to a closed socket raises one of these; the session only knows ConnectionError.
_CLOSED = (WebSocketDisconnect, WebSocketDisconnected)


class Transport(Protocol):
    """What the session uses to talk to one client."""

    async def send_json(self, data: dict[str, Any]) -> None:
        """Send one JSON message to the client. Raises ConnectionError if it is gone."""
        ...

    async def send_bytes(self, data: bytes) -> None:
        """Send one binary frame to the client (unused in v0, kept for the seam)."""
        ...


class SessionHandler(Protocol):
    """What a transport calls when something arrives. Implemented by Session."""

    async def on_connect(self) -> None: ...

    async def on_json(self, raw: str) -> None: ...

    async def on_audio_frame(self, data: bytes) -> None: ...

    async def on_disconnect(self) -> None: ...


class WebSocketTransport:
    """Adapts one accepted FastAPI WebSocket to `Transport`, plus the receive loop."""

    def __init__(self, ws: WebSocket) -> None:
        self._ws = ws

    async def send_json(self, data: dict[str, Any]) -> None:
        try:
            await self._ws.send_json(data)
        except _CLOSED as exc:
            raise ConnectionError("WebSocket closed") from exc

    async def send_bytes(self, data: bytes) -> None:
        try:
            await self._ws.send_bytes(data)
        except _CLOSED as exc:
            raise ConnectionError("WebSocket closed") from exc

    async def run(self, handler: SessionHandler) -> None:
        """Receive loop for one connection; returns when the client leaves.

        Handlers are awaited in order (the turn state machine relies on it), and
        on_disconnect runs even if a handler raises, so an in-flight turn is dropped.
        """
        await handler.on_connect()
        try:
            while True:
                event = await self._ws.receive()
                if event["type"] == "websocket.disconnect":
                    break  # receive() raises if called again after this
                text = event.get("text")
                data = event.get("bytes")
                if text is not None:
                    await handler.on_json(text)
                elif data is not None:
                    await handler.on_audio_frame(data)
        finally:
            await handler.on_disconnect()
