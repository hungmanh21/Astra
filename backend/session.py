"""Per-connection state and the turn state machine (SPEC 6.3, FR-11, FR-13, PLAN T1.6, T1.8).

One Session per WebSocket connection. It talks to the client only through a `Transport`
(see transport.py) and imports no FastAPI types.

M1 scope: session message, start_turn, audio frames, end_turn with validation and the
debug WAV. ASR, review mode and the LLM are added in M2-M4.
"""

import uuid
from dataclasses import dataclass
from enum import StrEnum

from backend.audio import AudioError, TurnAudioBuffer
from backend.config import Settings
from backend.protocol import (
    ConfirmTurn,
    DiscardTurn,
    EndTurn,
    ErrorMessage,
    ProtocolError,
    Reset,
    ServerMessage,
    SessionInfo,
    StartTurn,
    TurnStarted,
    parse_client_message,
)
from backend.transport import Transport


class TurnStage(StrEnum):
    IDLE = "idle"
    RECEIVING = "receiving"  # between start_turn and end_turn
    # M2 adds WORKING (ASR/LLM running), M4 adds AWAITING_CONFIRM (review mode).


@dataclass
class Turn:
    id: str
    asr_model: str
    review: bool
    audio: TurnAudioBuffer
    # M2: add the perf_counter() taken at end_turn, for e2e_ms (SPEC 6.3.2).


class Session:
    def __init__(self, transport: Transport, settings: Settings) -> None:
        self._transport = transport
        self._settings = settings
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

        try:
            self.turn.audio.append(data)
        except AudioError as exc:
            turn_id = self.turn.id
            self._finish_turn()
            await self._error(str(exc), turn_id)

    async def on_disconnect(self) -> None:
        # M2: also cancel the running turn task here.
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
            turn.audio.to_float32()  # validates the whole turn; M2 passes the samples to the ASR
            if self._settings.debug_save_audio:
                turn.audio.save_wav(self._settings.debug_audio_dir / f"{self.id}-{turn.id}.wav")
        except AudioError as exc:
            self._finish_turn()
            await self._error(str(exc), turn.id)
            return

        self._finish_turn()
        await self._error("ASR is not connected yet", turn.id)  # M1 only; T2.7 replaces this

    async def _on_reset(self) -> None:
        if self.stage != TurnStage.IDLE:
            await self._error("cannot reset while a turn is in flight")
            return
        self.history.clear()

    async def _on_review_decision(self, msg: ConfirmTurn | DiscardTurn) -> None:
        # M4 (T4.7) implements review mode; until then nothing can be waiting.
        await self._error("no transcript is waiting for review", msg.turn_id)

    # --- helpers ---

    async def _send(self, msg: ServerMessage) -> None:
        try:
            await self._transport.send_json(msg.model_dump(mode="json"))
        except ConnectionError:
            pass  # the browser left; nothing to tell

    async def _error(self, message: str, turn_id: str | None = None) -> None:
        await self._send(ErrorMessage(message=message, turn_id=turn_id))

    def _finish_turn(self) -> None:
        if self.turn is not None:
            self.turn.audio.clear()
        self.turn = None
        self.stage = TurnStage.IDLE
