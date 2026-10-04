"""Tests for backend/turnlog.py (PLAN T5.3): the record format and the logging setup."""

import json
import logging

import pytest

from backend.protocol import Timings
from backend.turnlog import STATUSES, configure_logging, log_turn

TIMINGS = Timings(audio_s=2.0, asr_ms=300.0, llm_ttft_ms=400.0, llm_total_ms=900.0, e2e_ms=1300.0)


def logged(caplog, **overrides):
    args = dict(
        session_id="s1",
        turn_id="t1",
        asr_model="whisper-large-v3",
        llm_model="gemini/x",
        status="ok",
        timings=TIMINGS,
        error=None,
    )
    args.update(overrides)
    with caplog.at_level(logging.INFO, logger="astra.turns"):
        log_turn(**args)
    return [r for r in caplog.records if r.name == "astra.turns"]


def test_record_is_one_json_line_with_exactly_the_documented_keys(caplog):
    (record,) = logged(caplog)
    message = record.getMessage()
    assert "\n" not in message
    assert json.loads(message) == {
        "event": "turn",
        "session_id": "s1",
        "turn_id": "t1",
        "asr_model": "whisper-large-v3",
        "llm_model": "gemini/x",
        "status": "ok",
        "timings": {
            "audio_s": 2.0,
            "asr_ms": 300.0,
            "llm_ttft_ms": 400.0,
            "llm_total_ms": 900.0,
            "e2e_ms": 1300.0,
        },
        "error": None,
    }


def test_missing_timings_and_an_error_are_written_as_null_and_a_string(caplog):
    (record,) = logged(caplog, status="asr_error", timings=None, error="transcription failed")
    data = json.loads(record.getMessage())
    assert data["timings"] is None
    assert data["error"] == "transcription failed"


@pytest.mark.parametrize(
    ("status", "level"),
    [
        ("ok", logging.INFO),
        ("empty", logging.INFO),
        ("discarded", logging.INFO),
        ("asr_error", logging.WARNING),
        ("llm_error", logging.WARNING),
        ("rejected", logging.WARNING),
        ("dropped", logging.WARNING),
    ],
)
def test_level_depends_on_the_status(caplog, status, level):
    (record,) = logged(caplog, status=status)
    assert record.levelno == level


def test_every_documented_status_is_covered_by_the_level_table():
    assert set(STATUSES) == {
        "ok",
        "empty",
        "discarded",
        "asr_error",
        "llm_error",
        "rejected",
        "dropped",
    }


@pytest.fixture
def clean_root_logging():
    root = logging.getLogger()
    saved = (root.handlers[:], root.level)
    # Start from a clean slate: earlier tests build apps, which already configured logging.
    root.handlers[:] = [h for h in root.handlers if type(h).__module__.startswith("_pytest")]
    noisy = {n: logging.getLogger(n).level for n in ("httpx", "httpcore", "LiteLLM")}
    yield root
    root.handlers[:] = saved[0]
    root.setLevel(saved[1])
    for name, level in noisy.items():
        logging.getLogger(name).setLevel(level)


def test_configure_logging_adds_one_stream_handler_and_sets_info(clean_root_logging):
    root = clean_root_logging
    before = len(root.handlers)
    configure_logging()
    assert root.level == logging.INFO
    assert len(root.handlers) == before + 1
    added = root.handlers[-1]
    assert isinstance(added, logging.StreamHandler)


def test_configure_logging_is_idempotent(clean_root_logging):
    root = clean_root_logging
    configure_logging()
    count = len(root.handlers)
    configure_logging()
    configure_logging()
    assert len(root.handlers) == count


def test_configure_logging_quiets_http_libraries(clean_root_logging):
    configure_logging()
    for name in ("httpx", "httpcore", "LiteLLM"):
        assert logging.getLogger(name).level == logging.WARNING


def test_configure_logging_formats_with_time_level_and_logger_name(clean_root_logging, capsys):
    configure_logging()
    logging.getLogger("astra.turns").info("hello")
    err = capsys.readouterr().err
    assert "INFO" in err and "astra.turns" in err and "hello" in err
