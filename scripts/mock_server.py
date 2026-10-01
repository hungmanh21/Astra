"""Throwaway mock backend for developing the frontend without any ASR or LLM.

Speaks the WebSocket protocol from SPEC 6.3, serves ./frontend as static files, and
answers every turn with a fake transcript and a streamed fake reply. Each received
turn is saved as a WAV in scripts/output/ so the audio path can be checked by ear.

Run with:
    uv run python scripts/mock_server.py
then open http://localhost:8000

Needs fastapi and uvicorn[standard]. Not part of the real backend; delete it once
backend/main.py replaces it.
"""

import array
import asyncio
import json
import sys
import time
import uuid
import wave
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).parent / "output"

SAMPLE_RATE = 16_000
MAX_SECONDS = 30
MIN_SECONDS = 0.25
MODELS = ["whisper-large-v3", "parakeet-tdt-0.6b-v3", "nemotron-3.5-asr-streaming-0.6b"]

app = FastAPI()


class Turn:
    def __init__(self, model: str, review: bool):
        self.id = uuid.uuid4().hex[:8]
        self.model = model
        self.review = review
        self.pcm = bytearray()
        self.decision: asyncio.Future[str | None] | None = (
            None  # review mode: text, or None = discard
        )


class Session:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.id = uuid.uuid4().hex[:8]
        self.stage = "idle"  # idle | receiving | working | awaiting_confirm
        self.turn: Turn | None = None
        self.task: asyncio.Task | None = None

    async def error(self, message: str, turn_id: str | None = None) -> None:
        await self.ws.send_json({"type": "error", "turn_id": turn_id, "message": message})

    def finish(self) -> None:
        self.stage = "idle"
        self.turn = None

    async def on_start(self, msg: dict) -> None:
        if self.stage != "idle":
            await self.error("a turn is already in flight")
            return
        model = msg.get("asr_model")
        if model not in MODELS:
            await self.error(f"unknown ASR model: {model}")
            return
        self.turn = Turn(model, bool(msg.get("review")))
        self.stage = "receiving"
        await self.ws.send_json({"type": "turn_started", "turn_id": self.turn.id})

    async def on_audio(self, data: bytes) -> None:
        if self.stage != "receiving":
            return
        self.turn.pcm.extend(data)
        if len(self.turn.pcm) > MAX_SECONDS * SAMPLE_RATE * 2:
            turn_id = self.turn.id
            self.finish()
            await self.error(f"turn is longer than {MAX_SECONDS} s", turn_id)

    async def on_end(self) -> None:
        if self.stage != "receiving":
            return
        turn = self.turn
        size = len(turn.pcm)
        problem = None
        if size % 2:
            problem = "audio has an odd number of bytes"
        elif size / 2 / SAMPLE_RATE < MIN_SECONDS:
            problem = "recording is too short"
        elif size / 2 / SAMPLE_RATE > MAX_SECONDS:
            problem = f"turn is longer than {MAX_SECONDS} s"
        if problem:
            self.finish()
            await self.error(problem, turn.id)
            return
        self.stage = "working"
        self.task = asyncio.create_task(self.run(turn))

    async def on_decision(self, msg: dict, confirm: bool) -> None:
        turn = self.turn
        if self.stage != "awaiting_confirm" or turn is None or msg.get("turn_id") != turn.id:
            await self.error("no transcript is waiting for review", msg.get("turn_id"))
            return
        text = msg.get("text", "").strip() if confirm else None
        turn.decision.set_result(text or None)

    async def on_reset(self) -> None:
        if self.stage != "idle":
            await self.error("cannot reset while a turn is in flight")

    async def run(self, turn: Turn) -> None:
        try:
            samples = array.array("h")
            samples.frombytes(bytes(turn.pcm))
            if sys.byteorder == "big":
                samples.byteswap()
            seconds = len(samples) / SAMPLE_RATE
            peak = max(abs(s) for s in samples) / 32768

            OUT_DIR.mkdir(exist_ok=True)
            path = OUT_DIR / f"{self.id}-{turn.id}.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(SAMPLE_RATE)
                wav.writeframes(samples.tobytes())
            print(f"saved {path.name}: {seconds:.2f} s, peak {peak:.0%}")

            t0 = time.perf_counter()
            await asyncio.sleep(0.4)  # pretend to run ASR
            asr_ms = (time.perf_counter() - t0) * 1000
            text = f"[mock] {seconds:.1f} s of audio, peak level {peak:.0%}, model {turn.model}"
            await self.ws.send_json(
                {"type": "transcript", "turn_id": turn.id, "text": text, "asr_ms": asr_ms}
            )

            if turn.review:
                turn.decision = asyncio.get_running_loop().create_future()
                self.stage = "awaiting_confirm"
                confirmed = await turn.decision
                if confirmed is None:
                    return  # discarded (or empty): nothing is added
                text = confirmed
                self.stage = "working"

            reply = f'This is a mock reply. You said: "{text}"'
            t1 = time.perf_counter()
            ttft_ms = None
            for word in reply.split(" "):
                await asyncio.sleep(0.06)
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t1) * 1000
                await self.ws.send_json(
                    {"type": "llm_delta", "turn_id": turn.id, "text": word + " "}
                )
            llm_ms = (time.perf_counter() - t1) * 1000
            await self.ws.send_json(
                {
                    "type": "llm_done",
                    "turn_id": turn.id,
                    "text": reply,
                    "timings": {
                        "audio_s": seconds,
                        "asr_ms": asr_ms,
                        "llm_ttft_ms": ttft_ms,
                        "llm_total_ms": llm_ms,
                        "e2e_ms": asr_ms + llm_ms,
                    },
                }
            )
        except Exception as exc:  # a closed socket mid-turn lands here
            print(f"turn {turn.id} aborted: {exc!r}")
        finally:
            self.finish()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    session = Session(ws)
    await ws.send_json(
        {
            "type": "session",
            "session_id": session.id,
            "asr_models": MODELS,
            "default_asr_model": MODELS[0],
        }
    )
    try:
        while True:
            event = await ws.receive()
            if event["type"] == "websocket.disconnect":
                break
            if event.get("bytes") is not None:
                await session.on_audio(event["bytes"])
                continue
            msg = json.loads(event["text"])
            kind = msg.get("type")
            if kind == "start_turn":
                await session.on_start(msg)
            elif kind == "end_turn":
                await session.on_end()
            elif kind == "confirm_turn":
                await session.on_decision(msg, confirm=True)
            elif kind == "discard_turn":
                await session.on_decision(msg, confirm=False)
            elif kind == "reset":
                await session.on_reset()
            else:
                await session.error(f"unknown message type: {kind}")
    finally:
        if session.task:
            session.task.cancel()


# Mounted last so it does not shadow /ws.
app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
