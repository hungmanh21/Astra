"""Shared pytest setup."""

import signal

import pytest

TEST_TIMEOUT_S = 20


@pytest.fixture(autouse=True)
def fail_hung_tests():
    """Fail a test that takes too long instead of hanging the whole run.

    The WebSocket tests block on `receive_json()` with no timeout, so a server that never
    answers would otherwise stall `make check` forever. SIGALRM is Unix-only, which matches
    the Linux/WSL target.
    """

    def on_alarm(signum, frame):
        raise TimeoutError(f"test ran longer than {TEST_TIMEOUT_S} s (hung waiting for a reply?)")

    previous = signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(TEST_TIMEOUT_S)
    yield
    signal.alarm(0)
    signal.signal(signal.SIGALRM, previous)
