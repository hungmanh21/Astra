"""One structured log line per turn, and the logging setup that makes it visible (PLAN T5.3).

The line is JSON so it can be parsed later (`grep astra.turns server.log`, then `jq`). It never
contains the transcript, the reply or any audio: those are the user's data (privacy NFR).
"""

import json
import logging

from backend.protocol import Timings

turn_log = logging.getLogger("astra.turns")

_HANDLER_MARK = "_astra_handler"
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# Every turn ends in exactly one of these.
STATUSES = ("ok", "empty", "asr_error", "llm_error", "rejected", "discarded", "dropped")


def log_turn(
    *,
    session_id: str,
    turn_id: str,
    asr_model: str,
    llm_model: str,
    status: str,
    timings: Timings | None = None,
    error: str | None = None,
) -> None:
    """Write the turn's single record to `turn_log`.

    Input:  status - one of STATUSES:
              ok         the reply was sent
              empty      the transcript was blank, no LLM call
              asr_error  transcription failed
              llm_error  the LLM request failed
              rejected   bad audio (too short, odd bytes, over the cap) or a blank confirm
              discarded  the user discarded the transcript in review mode
              dropped    the connection closed while the turn was in flight
            timings - the same five values as `llm_done`, or None when there are none.
            error - the short message the user was shown, or None.

    INFO for "ok", "empty" and "discarded", WARNING for the rest so failed turns stand out.
    The record never holds the transcript, the reply text or audio.
    """
    log_dict = {
        "event": "turn",
        "session_id": session_id,
        "turn_id": turn_id,
        "asr_model": asr_model,
        "llm_model": llm_model,
        "status": status,
        "timings": timings.model_dump() if timings is not None else None,
        "error": error,
    }

    if status in ("ok", "empty", "discarded"):
        turn_log.info(json.dumps(log_dict))

    if status in ("asr_error", "llm_error", "rejected", "dropped"):
        turn_log.warning(json.dumps(log_dict))


def configure_logging() -> None:
    """Make INFO logs visible on stderr, once, with quiet third-party loggers.

    Uvicorn only configures its own loggers, so ours would be dropped. Safe to call twice
    (create_app runs more than once in tests). httpx, httpcore and LiteLLM log one INFO line per
    HTTP request and would drown the turn lines.
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(getattr(h, _HANDLER_MARK, False) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        setattr(handler, _HANDLER_MARK, True)
        root.addHandler(handler)

    for name in ("httpx", "httpcore", "LiteLLM"):
        logging.getLogger(name).setLevel(logging.WARNING)
