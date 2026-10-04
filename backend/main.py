"""FastAPI app: the /ws endpoint plus the static frontend (SPEC 6.6, PLAN T1.3).

Run with:
    make run
    # or: uv run uvicorn backend.main:create_app --factory --port 8000
then open http://localhost:8000
"""

from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles

from backend.asr.registry import ASRRegistry
from backend.asr.service import TranscriptionService
from backend.config import Settings, load_settings
from backend.llm import LLMClient
from backend.session import Session
from backend.transport import WebSocketTransport
from backend.turnlog import configure_logging

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


def create_app(
    settings: Settings | None = None,
    transcriber: TranscriptionService | None = None,
    llm: LLMClient | None = None,
) -> FastAPI:
    """Build the app. A factory, so tests can pass their own Settings, transcriber and LLM."""
    configure_logging()  # uvicorn only sets up its own loggers; ours would be dropped

    settings = settings or load_settings()

    # Models stay lazy: nothing is loaded here, the first turn on a model loads it.
    if transcriber is None:
        registry = ASRRegistry(settings.asr_models)
        transcriber = TranscriptionService(registry)

    if llm is None:
        llm = LLMClient(settings.llm)

    # Stops the ASR worker thread when the server stops.
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        transcriber.shutdown()

    app = FastAPI(lifespan=lifespan)
    app.state.settings = settings
    app.state.transcriber = transcriber
    app.state.llm = llm

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        if not _same_origin(ws):
            await ws.close(code=1008)  # policy violation; sent before accept, so an HTTP 403
            return
        await ws.accept()
        transport = WebSocketTransport(ws)
        await transport.run(Session(transport, settings, transcriber, llm))

    # Mounted last so it does not shadow /ws.
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    return app


def _same_origin(ws: WebSocket) -> bool:
    """True unless a browser on another site opened the socket.

    Browsers do not apply the same-origin policy to WebSockets, so without this any page the
    user visits could connect to localhost and spend the LLM key. Clients that send no Origin
    (scripts, tests) are not browsers and are let through.
    """
    origin = ws.headers.get("origin")
    if origin is None:
        return True
    return urlsplit(origin).netloc == ws.headers.get("host")


def main() -> None:
    uvicorn.run(create_app(), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
