"""Message shapes for the /ws protocol (SPEC 6.3, PLAN T1.4).

Server messages are built as these models and sent with `model_dump(mode="json")`.
"""

from typing import Annotated, Literal

import pydantic
from pydantic import BaseModel, Field

# --- client -> server (JSON only; audio frames are binary and never go through here) ---


class StartTurn(BaseModel):
    type: Literal["start_turn"]
    asr_model: str
    review: bool = False


class EndTurn(BaseModel):
    type: Literal["end_turn"]


class ConfirmTurn(BaseModel):
    type: Literal["confirm_turn"]
    turn_id: str
    text: str  # the possibly edited transcript


class DiscardTurn(BaseModel):
    type: Literal["discard_turn"]
    turn_id: str


class Reset(BaseModel):
    type: Literal["reset"]


ClientMessage = Annotated[
    StartTurn | EndTurn | ConfirmTurn | DiscardTurn | Reset, Field(discriminator="type")
]

# --- server -> client ---


class SessionInfo(BaseModel):
    type: Literal["session"] = "session"
    session_id: str
    asr_models: list[str]
    default_asr_model: str
    max_turn_seconds: float  # the UI auto-stops here, so it never sends a turn the server rejects


class TurnStarted(BaseModel):
    type: Literal["turn_started"] = "turn_started"
    turn_id: str


class Transcript(BaseModel):
    type: Literal["transcript"] = "transcript"
    turn_id: str
    text: str
    asr_ms: float


class LlmDelta(BaseModel):
    type: Literal["llm_delta"] = "llm_delta"
    turn_id: str
    text: str


class Timings(BaseModel):
    audio_s: float
    asr_ms: float
    llm_ttft_ms: float | None = None
    llm_total_ms: float | None = None
    e2e_ms: float | None = None


class LlmDone(BaseModel):
    type: Literal["llm_done"] = "llm_done"
    turn_id: str
    text: str
    timings: Timings


class ErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    turn_id: str | None = None  # None for errors not tied to a turn
    message: str  # shown to the user, no internals


ServerMessage = SessionInfo | TurnStarted | Transcript | LlmDelta | LlmDone | ErrorMessage


class ProtocolError(ValueError):
    """The client sent something that is not a valid message. The text is safe to show."""


def parse_client_message(raw: str | bytes) -> ClientMessage:
    """Parse one JSON text frame from the browser into a typed message.

    Raises ProtocolError (never a pydantic or JSON error) so the session can turn it
    into an `error` message and keep the connection open.
    """
    try:
        return pydantic.TypeAdapter(ClientMessage).validate_json(raw)
    except pydantic.ValidationError as e:
        raise ProtocolError("invalid message: " + str(e)) from e
