"""Tests for backend/asr/registry.py and backend/asr/service.py (PLAN T2.2, T2.6).

These use a fake adapter defined in this file so no GPU or model download is needed.
Point the ASRModelConfig.adapter at "tests.test_asr_registry.FakeASR" (add an empty
tests/__init__.py if the import does not resolve), and count load() calls on a class attribute.
"""

import pytest

pytestmark = pytest.mark.skip(reason="TODO: implement registry and service first")


class FakeASR:
    """TODO: satisfies ASRModel; load() increments a counter and can sleep briefly;
    transcribe() returns a TranscriptResult with a fixed text."""


def test_available_returns_config_keys_in_order():
    pass


def test_unknown_model_raises_unknown_asr_model():
    pass


def test_get_loads_lazily_and_caches():
    """No load() at construction; two get() calls -> one load(); second load_ms == 0.0."""


def test_concurrent_get_loads_once():
    """Several threads call get() on the same unloaded model: load() runs exactly once."""


def test_adapter_receives_name_model_id_and_options():
    """The adapter instance is built with name=<key>, model_id=..., and the options as kwargs."""


def test_service_runs_off_the_event_loop():
    """While a slow fake transcribe runs, another coroutine on the loop keeps making progress."""


def test_service_reports_load_time_separately():
    """First call: model_load_ms > 0 and asr_ms excludes it. Second call: model_load_ms == 0.0."""


def test_service_wraps_unexpected_errors_in_asr_error():
    """A fake raising ValueError surfaces as ASRError; UnknownASRModel passes through."""
