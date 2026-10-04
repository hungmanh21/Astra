"""Per-connection state and the turn state machine (SPEC 6.3, FR-11, FR-13, PLAN T1.6, T1.8).

One Session per WebSocket connection. It talks to the client only through a `Transport`
(see transport.py) and imports no FastAPI types.

M1 scope: session message, start_turn, audio frames, end_turn with validation and the
debug WAV. M2 adds ASR (PLAN T2.6, T2.7). M3 adds the LLM reply (T3.4-T3.7). Review mode
comes in M4.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from backend import turnlog
from backend.asr.service import Transcription, TranscriptionService
from backend.audio import AudioError, TurnAudioBuffer
from backend.config import Settings
from backend.history import build_messages, trim_history
from backend.llm import LLMClient, LLMError
from backend.protocol import (
    ConfirmTurn,
    DiscardTurn,
    EndTurn,
    ErrorMessage,
    LlmDelta,
    LlmDone,
    ProtocolError,
    Reset,
    ServerMessage,
    SessionInfo,
    StartTurn,
    Timings,
    Transcript,
    TurnStarted,
    parse_client_message,
)
from backend.transport import Transport

log = logging.getLogger(__name__)


class TurnStage(StrEnum):
    IDLE = "idle"
    RECEIVING = "receiving"  # between start_turn and end_turn
    WORKING = "working"  # end_turn accepted, the ASR (later the LLM) is running
    AWAITING_CONFIRM = "awaiting_confirm"  # review mode: transcript sent, waiting for the user


@dataclass
class Turn:
    id: str
    asr_model: str
    review: bool
    audio: TurnAudioBuffer
    ended_at: float = 0.0  # clock reading at end_turn, for e2e_ms (SPEC 6.3.2)
    # Review mode (T4.6): the ASR result kept while the user edits, and when the wait began.
    transcription: Transcription | None = None
    review_started_at: float = 0.0


class Session:
    def __init__(
        self,
        transport: Transport,
        settings: Settings,
        transcriber: TranscriptionService,
        llm: LLMClient,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._clock = clock  # injectable so tests can drive time
        self._transport = transport
        self._settings = settings
        self._transcriber = transcriber
        self._llm = llm
        # The running ASR job of the current turn. Kept so on_disconnect can cancel it.
        self._task: asyncio.Task[None] | None = None
        self.id = uuid.uuid4().hex[:8]
        self.history: list[dict[str, str]] = []  # used from M3
        self.stage = TurnStage.IDLE
        self.turn: Turn | None = None

    # --- called by the transport ---

    async def on_connect(self) -> None:
        await self._send(
            SessionInfo(
                session_id=self.id,
                asr_models=list(self._settings.asr_models),
                default_asr_model=self._settings.default_asr_model,
            )
        )

    async def on_json(self, raw: str) -> None:
        try:
            msg = parse_client_message(raw)
        except ProtocolError as exc:
            await self._error(str(exc))
            return

        match msg:
            case StartTurn():
                await self._on_start_turn(msg)
            case EndTurn():
                await self._on_end_turn()
            case Reset():
                await self._on_reset()
            case ConfirmTurn() | DiscardTurn():
                await self._on_review_decision(msg)

    async def on_audio_frame(self, data: bytes) -> None:
        # Late frames after a rejected or ended turn are normal.
        if self.stage != TurnStage.RECEIVING or self.turn is None:
            return

        turn = self.turn
        try:
            turn.audio.append(data)
        except AudioError as exc:
            self._record_turn(turn, "rejected", error=str(exc))
            self._finish_turn()
            await self._error(str(exc), turn.id)

    async def on_disconnect(self) -> None:
        # Read the turn before cancelling: the cancelled task's `finally` clears `self.turn`.
        turn = self.turn
        if turn is not None:
            self._record_turn(turn, "dropped")

        # A turn still in the ASR must not try to answer a browser that has left.
        task = self._task
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            self._task = None

        self._finish_turn()

    # --- message handlers ---

    async def _on_start_turn(self, msg: StartTurn) -> None:
        if self.stage != TurnStage.IDLE:
            await self._error("a turn is already in flight")
            return

        if msg.asr_model not in self._settings.asr_models:
            await self._error(f"unknown ASR model: {msg.asr_model}")
            return

        self.turn = Turn(
            id=uuid.uuid4().hex[:8],
            asr_model=msg.asr_model,
            review=msg.review,
            audio=TurnAudioBuffer(self._settings.max_turn_seconds),
        )
        # State is set before the ack: audio may arrive the moment the browser sees it.
        self.stage = TurnStage.RECEIVING
        await self._send(TurnStarted(turn_id=self.turn.id))

    async def _on_end_turn(self) -> None:
        if self.stage != TurnStage.RECEIVING or self.turn is None:
            return

        turn = self.turn
        try:
            samples = turn.audio.to_float32()  # validates the whole turn
            if self._settings.debug_save_audio:
                turn.audio.save_wav(self._settings.debug_audio_dir / f"{self.id}-{turn.id}.wav")
        except AudioError as exc:
            self._record_turn(turn, "rejected", error=str(exc))
            self._finish_turn()
            await self._error(str(exc), turn.id)
            return

        # Background task: the handler returns at once, so the receive loop stays free to
        # reject a start_turn or reset while the ASR runs.
        turn.ended_at = self._clock()  # for e2e_ms (SPEC 6.3.2)
        self.stage = TurnStage.WORKING
        self._task = asyncio.create_task(self._run_turn(turn, samples))

    async def _run_turn(self, turn: Turn, samples: np.ndarray) -> None:
        """Transcribe one turn, then stream the LLM reply. Runs as a background task."""
        try:
            try:
                result = await self._transcriber.transcribe(turn.asr_model, samples)
            except Exception:
                # Catch everything: a task that dies silently leaves the stage stuck on
                # WORKING. The message is fixed on purpose, ASR error text can hold internals.
                log.exception("transcription failed for turn %s", turn.id)
                self._record_turn(turn, "asr_error", error="transcription failed")
                await self._error("transcription failed", turn.id)
                return

            if result.model_load_ms > 0:
                log.info(
                    "first use of model %s: load time %.1f ms", turn.asr_model, result.model_load_ms
                )
            await self._send(Transcript(turn_id=turn.id, text=result.text, asr_ms=result.asr_ms))

            # Review mode: wait for confirm_turn / discard_turn. A blank transcript waits too, the
            # UI shows an empty box and the user discards it.
            if turn.review:
                turn.transcription = result
                turn.review_started_at = self._clock()
                self.stage = TurnStage.AWAITING_CONFIRM
                return

            if result.text.strip() == "":
                # Nothing to ask the model; llm_done is what unlocks the UI.
                timings = Timings(audio_s=result.audio_s, asr_ms=result.asr_ms)
                await self._send(LlmDone(turn_id=turn.id, text="", timings=timings))
                self._record_turn(turn, "empty", timings)
                return

            await self._reply(turn, result, result.text)
        finally:
            if self.stage != TurnStage.AWAITING_CONFIRM:
                self._finish_turn()

    async def _reply(self, turn: Turn, result: Transcription, text: str) -> None:
        # `text` is the transcript, or the edited one in review mode. The user message stays
        # in the history even if the LLM fails (FR-14).
        self.history.append({"role": "user", "content": text})
        max_tokens = self._settings.llm.max_history_tokens
        trim_history(self.history, max_tokens)
        messages = build_messages(self._settings.llm.system_prompt, self.history)

        start = self._clock()
        first: float | None = None
        last = start
        deltas: list[str] = []
        try:
            async for delta in self._llm.stream(messages):
                last = self._clock()
                if first is None:
                    first = last
                deltas.append(delta)
                await self._send(LlmDelta(turn_id=turn.id, text=delta))
        except Exception as exc:
            if isinstance(exc, LLMError):
                log.warning("turn %s: %s", turn.id, exc)  # llm.py already logged the details
            else:
                log.exception("LLM step failed for turn %s", turn.id)
            self._record_turn(turn, "llm_error", error="LLM request failed")
            await self._error("LLM request failed", turn.id)
            return

        reply = "".join(deltas)
        if reply:  # an empty assistant message would be rejected by some providers
            self.history.append({"role": "assistant", "content": reply})
            trim_history(self.history, max_tokens)
        timings = Timings(
            audio_s=result.audio_s,
            asr_ms=result.asr_ms,
            llm_ttft_ms=None if first is None else (first - start) * 1000,
            llm_total_ms=(last - start) * 1000,
            e2e_ms=(last - turn.ended_at) * 1000,
        )
        await self._send(LlmDone(turn_id=turn.id, text=reply, timings=timings))
        self._record_turn(turn, "ok", timings)

    async def _on_reset(self) -> None:
        if self.stage != TurnStage.IDLE:
            await self._error("cannot reset while a turn is in flight")
            return
        self.history.clear()

    async def _on_review_decision(self, msg: ConfirmTurn | DiscardTurn) -> None:
        """confirm_turn / discard_turn (review mode only)."""
        turn = self.turn
        if self.stage != TurnStage.AWAITING_CONFIRM or turn is None or msg.turn_id != turn.id:
            await self._error("no transcript is waiting for review", msg.turn_id)
            return

        if isinstance(msg, DiscardTurn):
            # Nothing is added to the history, and the browser already removed the bubble.
            self._record_turn(turn, "discarded")
            self._finish_turn()
            return

        text = msg.text.strip()
        if not text:
            self._record_turn(turn, "rejected", error="transcript is empty")
            self._finish_turn()
            await self._error("transcript is empty", turn.id)
            return

        # Move the turn's clock past the user's thinking time, so e2e_ms excludes it.
        turn.ended_at += self._clock() - turn.review_started_at
        self.stage = TurnStage.WORKING
        self._task = asyncio.create_task(self._run_confirmed(turn, text))

    async def _run_confirmed(self, turn: Turn, text: str) -> None:
        """The LLM half of a reviewed turn. Runs as a background task."""
        try:
            assert turn.transcription is not None  # set when the turn started waiting
            await self._reply(turn, turn.transcription, text)
        except Exception:
            # _reply handles LLM errors itself; this keeps any other failure from dying silently.
            log.exception("reply failed for turn %s", turn.id)
            self._record_turn(turn, "llm_error", error="LLM request failed")
            await self._error("LLM request failed", turn.id)
        finally:
            self._finish_turn()

    # --- helpers ---

    async def _send(self, msg: ServerMessage) -> None:
        try:
            await self._transport.send_json(msg.model_dump(mode="json"))
        except ConnectionError:
            pass  # the browser left; nothing to tell

    async def _error(self, message: str, turn_id: str | None = None) -> None:
        await self._send(ErrorMessage(message=message, turn_id=turn_id))

    def _record_turn(
        self,
        turn: Turn,
        status: str,
        timings: Timings | None = None,
        error: str | None = None,
    ) -> None:
        """Write the turn's one structured log line (T5.3). Call it once per turn."""
        turnlog.log_turn(
            session_id=self.id,
            turn_id=turn.id,
            asr_model=turn.asr_model,
            llm_model=self._settings.llm.model,
            status=status,
            timings=timings,
            error=error,
        )

    def _finish_turn(self) -> None:
        if self.turn is not None:
            self.turn.audio.clear()
        self.turn = None
        self.stage = TurnStage.IDLE
