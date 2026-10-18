"""Per-connection state and the turn state machine (SPEC 6.3, FR-11, FR-13, PLAN T1.6, T1.8).

One Session per WebSocket connection. It talks to the client only through a `Transport`
(see transport.py) and imports no FastAPI types.

M1 scope: session message, start_turn, audio frames, end_turn with validation and the
debug WAV. M2 adds ASR (PLAN T2.6, T2.7). M3 adds the LLM reply (T3.4-T3.7). Review mode
comes in M4.
"""

import asyncio
import functools
import logging
import statistics
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from backend import turnlog
from backend.asr.base import ASRError, PartialResult
from backend.asr.service import Transcription, TranscriptionService
from backend.asr.streaming import StreamingTranscription
from backend.audio import AudioError, TurnAudioBuffer, pcm16_to_float32
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
    Partial,
    ProtocolError,
    Reset,
    ServerMessage,
    SessionInfo,
    StartTurn,
    StrategyInfo,
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
    # Set by the turn's one log record. A disconnect while an error is still being sent would
    # otherwise log the same turn a second time as "dropped".
    recorded: bool = False
    # Streaming (SPEC_v1 6.4). `streaming` drives this turn's live partials; the session never
    # looks at which kind of stream is behind it.
    stream_strategy: str = ""
    streaming: StreamingTranscription | None = None
    carry: bytes = b""  # a trailing odd byte of the last frame, belongs to the next frame
    partial_count: int = 0  # `partial` messages sent for this turn
    partial_lags_ms: list[float] = field(default_factory=list)  # one per partial sent
    final_ms: float | None = None  # end_turn received -> transcript sent (SPEC_v1 6.3)


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
                max_turn_seconds=self._settings.max_turn_seconds,
                release_tail_ms=self._settings.release_tail_ms,
                stream_strategies=self._strategy_infos(),
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
            await self._close_stream(turn)
            self._finish_turn()
            await self._error(str(exc), turn.id)
            return

        samples = self._frame_samples(turn, data)
        if turn.streaming is not None and len(samples) > 0:
            turn.streaming.push(samples)

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

        # The cancelled task may have left a partial decode running.
        if turn is not None:
            await self._close_stream(turn)

        self._finish_turn()

    # --- message handlers ---

    async def _on_start_turn(self, msg: StartTurn) -> None:
        if self.stage != TurnStage.IDLE:
            await self._error("a turn is already in flight")
            return

        if msg.asr_model not in self._settings.asr_models:
            await self._error(f"unknown ASR model: {msg.asr_model}")
            return

        try:
            strategy = self._resolve_strategy(msg.asr_model, msg.stream_strategy)
            opened = await self._transcriber.open_stream(msg.asr_model, strategy)
        except ProtocolError as exc:
            await self._error(str(exc))
            return
        except Exception:
            log.exception("could not open the ASR stream for model %s", msg.asr_model)
            await self._error("could not start transcription")
            return

        # A cold model loads here, before the ack (v0 loaded it after end_turn).
        if opened.model_load_ms > 0:
            log.info(
                "first use of model %s: load time %.1f ms", msg.asr_model, opened.model_load_ms
            )

        turn = Turn(
            id=uuid.uuid4().hex[:8],
            asr_model=msg.asr_model,
            review=msg.review,
            audio=TurnAudioBuffer(self._settings.max_turn_seconds),
            stream_strategy=strategy,
        )
        turn.streaming = StreamingTranscription(
            opened.stream,
            self._transcriber.run,
            functools.partial(self._on_partial, turn),
            interval_ms=self._settings.asr_streaming.partial_interval_ms,
            clock=self._clock,  # the same clock as the timings, so partial_lag_ms is consistent
        )
        self.turn = turn
        # State is set before the ack: audio may arrive the moment the browser sees it.
        self.stage = TurnStage.RECEIVING
        await self._send(TurnStarted(turn_id=turn.id))

    async def _on_end_turn(self) -> None:
        if self.stage != TurnStage.RECEIVING or self.turn is None:
            return

        turn = self.turn
        try:
            turn.audio.to_float32()  # validates the whole turn; the stream already has the audio
            if self._settings.debug_save_audio:
                turn.audio.save_wav(self._settings.debug_audio_dir / f"{self.id}-{turn.id}.wav")
        except AudioError as exc:
            self._record_turn(turn, "rejected", error=str(exc))
            await self._close_stream(turn)
            self._finish_turn()
            await self._error(str(exc), turn.id)
            return

        # Background task: the handler returns at once, so the receive loop stays free to
        # reject a start_turn or reset while the ASR runs.
        turn.ended_at = self._clock()  # for e2e_ms (SPEC 6.3.2)
        self.stage = TurnStage.WORKING
        self._task = asyncio.create_task(self._run_turn(turn))

    async def _run_turn(self, turn: Turn) -> None:
        """Finish the turn's transcript, then stream the LLM reply. Runs as a background task."""
        try:
            try:
                assert turn.streaming is not None  # set in _on_start_turn
                final = await turn.streaming.finish()
            except Exception:
                # Catch everything: a task that dies silently leaves the stage stuck on
                # WORKING. The message is fixed on purpose, ASR error text can hold internals.
                log.exception("transcription failed for turn %s", turn.id)
                self._record_turn(turn, "asr_error", error="transcription failed")
                await self._error("transcription failed", turn.id)
                return

            result = Transcription(
                model=turn.asr_model,
                text=final.text,
                audio_s=final.duration_s,
                asr_ms=final.latency_ms,
                model_load_ms=0.0,  # a load happened in open_stream, at start_turn
            )
            turn.final_ms = (self._clock() - turn.ended_at) * 1000
            await self._send(
                Transcript(
                    turn_id=turn.id,
                    text=result.text,
                    asr_ms=result.asr_ms,
                    stream_strategy=turn.stream_strategy,
                )
            )

            # Review mode: wait for confirm_turn / discard_turn. A blank transcript waits too, the
            # UI shows an empty box and the user discards it.
            if turn.review:
                turn.transcription = result
                turn.review_started_at = self._clock()
                self.stage = TurnStage.AWAITING_CONFIRM
                return

            if result.text.strip() == "":
                # Nothing to ask the model; llm_done is what unlocks the UI.
                timings = Timings(
                    audio_s=result.audio_s, asr_ms=result.asr_ms, **self._streaming_fields(turn)
                )
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
            **self._streaming_fields(turn),
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

    # --- streaming helpers (PLAN_v1 S1.6) ---

    def _strategy_infos(self) -> dict[str, StrategyInfo]:
        """Each configured model's supported strategies and default, for the `session` message.

        Output: {model name: StrategyInfo}, in config order. Ask the transcriber:
                `self._transcriber.strategies(name)` (it does not load the model).
                A model whose adapter cannot even be created (strategies() raises an
                ASRError) is left out and logged: one broken model must not stop the browser
                from connecting and using the others.
        """
        ans: dict[str, StrategyInfo] = {}
        for name in self._settings.asr_models:
            try:
                strategies = self._transcriber.strategies(name)
            except ASRError:
                log.warning("could not get strategies for model %s", name, exc_info=True)
                continue
            ans[name] = StrategyInfo(supported=strategies.supported, default=strategies.default)
        return ans

    def _resolve_strategy(self, model: str, requested: str | None) -> str:
        """The strategy a new turn will use (FR-22).

        Input:  model - a name already checked to be in the config.
                requested - `start_turn.stream_strategy`, or None if the client sent none.
        Output: `requested` if the model supports it, or the model's default when None.
        Raises: ProtocolError, with a readable message that names the strategy and the model,
                when `requested` is not in the model's supported list. (This text goes to the
                browser as an `error`.) Any other exception from the transcriber is left alone.
        """
        strategies = self._transcriber.strategies(model)
        if requested is None:
            return strategies.default
        if requested not in strategies.supported:
            raise ProtocolError(f"{model!r} does not support strategy {requested!r}")
        return requested

    def _frame_samples(self, turn: Turn, data: bytes) -> np.ndarray:
        """The float32 samples of one binary frame, carrying a trailing odd byte over.

        A frame may end in the middle of a 16-bit sample. The turn buffer keeps raw bytes (so
        validation is unchanged), but the stream needs whole samples.
        Input:  data - one frame, PCM16 little-endian (any length, even odd or empty).
                `turn.carry` - the leftover byte of the previous frame (b"" or one byte).
        Output: float32 samples for every complete 2-byte pair in `turn.carry + data`
                (empty array if there are none). Sets `turn.carry` to the one leftover byte,
                or b"". Across a whole turn, the samples returned equal the turn's samples.
        Hint:   `pcm16_to_float32` in backend/audio.py does the conversion, and raises on an
                odd length, so give it only the complete pairs.
        """
        combined = turn.carry + data
        if len(combined) % 2 == 1:
            turn.carry = combined[-1:]
            combined = combined[:-1]
        else:
            turn.carry = b""
        if not combined:
            return np.array([], dtype=np.float32)
        return pcm16_to_float32(combined)

    async def _on_partial(
        self, turn: Turn, partial: PartialResult, audio_received_at: float
    ) -> None:
        """Send one `partial` message for `turn` (called by the turn's StreamingTranscription).

        Input:  audio_received_at - `self._clock()` when the last audio frame covered by this
                decode was pushed.
        Does:   nothing at all unless `turn` is still the current turn and still RECEIVING:
                a `partial` must never follow its `transcript` (SPEC_v1 6.2). Otherwise sends
                a `Partial` (turn id, committed, tentative), adds 1 to `turn.partial_count`
                and appends the lag in ms, `(self._clock() - audio_received_at) * 1000`, to
                `turn.partial_lags_ms`.
        """
        if self.turn is not turn or self.stage != TurnStage.RECEIVING:
            return
        await self._send(
            Partial(turn_id=turn.id, committed=partial.committed, tentative=partial.tentative)
        )
        turn.partial_count += 1
        turn.partial_lags_ms.append((self._clock() - audio_received_at) * 1000)

    def _streaming_fields(self, turn: Turn) -> dict:
        """The four streaming fields of `Timings` for this turn (SPEC_v1 6.3).

        Output: a dict with the keys `stream_strategy`, `partial_count`, `partial_lag_ms`,
                `final_ms`, to be unpacked into `Timings(...)`.
                partial_count is the number of partials sent (0 is a real value).
                partial_lag_ms is the MEDIAN of `turn.partial_lags_ms`, or None if there
                were no partials. final_ms is `turn.final_ms`.
        Hint:   `statistics.median`.
        """
        if turn.partial_lags_ms:
            median_lag = statistics.median(turn.partial_lags_ms)
        else:
            median_lag = None
        return {
            "stream_strategy": turn.stream_strategy,
            "partial_count": turn.partial_count,
            "partial_lag_ms": median_lag,
            "final_ms": turn.final_ms,
        }

    async def _close_stream(self, turn: Turn) -> None:
        """Abandon the turn's stream (rejected turn, disconnect): stops any decode we wait on."""
        if turn.streaming is not None:
            await turn.streaming.close()

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
        """Write the turn's one structured log line (T5.3). Later calls for the turn do nothing."""
        if turn.recorded:
            return
        turn.recorded = True
        turnlog.log_turn(
            session_id=self.id,
            turn_id=turn.id,
            asr_model=turn.asr_model,
            llm_model=self._llm.model,
            status=status,
            timings=timings,
            error=error,
        )

    def _finish_turn(self) -> None:
        if self.turn is not None:
            self.turn.audio.clear()
        self.turn = None
        self.stage = TurnStage.IDLE
