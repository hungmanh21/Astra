"""FastAPI app: the /ws endpoint plus the static frontend (SPEC 6.6, PLAN T1.3).

Run with:
    make run
    # or: uv run uvicorn backend.main:create_app --factory --port 8000
then open http://localhost:8000
"""

from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles

from backend.config import Settings, load_settings
from backend.session import Session
from backend.transport import WebSocketTransport

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app. A factory, so tests can pass their own Settings."""
    settings = settings or load_settings()
    app = FastAPI()
    app.state.settings = settings  # M2 adds the registry and TranscriptionService here

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        transport = WebSocketTransport(ws)
        await transport.run(Session(transport, settings))

    # Mounted last so it does not shadow /ws.
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    return app


def main() -> None:
    uvicorn.run(create_app(), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
