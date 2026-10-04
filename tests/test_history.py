"""Tests for backend/history.py (PLAN T3.3).

Token counting is injected as `len` (one token per character) wherever exact numbers matter.
"""

from backend.history import build_messages, estimate_tokens, trim_history


def msg(role: str, size: int) -> dict[str, str]:
    return {"role": role, "content": "x" * size}


def sizes(history: list[dict[str, str]]) -> list[tuple[str, int]]:
    return [(m["role"], len(m["content"])) for m in history]


def test_estimate_tokens_is_zero_for_empty_and_grows_with_length():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a") >= 1
    assert estimate_tokens("hello world") >= 1
    assert estimate_tokens("x" * 400) > estimate_tokens("x" * 40) > estimate_tokens("x" * 4)
    assert 80 <= estimate_tokens("x" * 400) <= 120  # roughly 4 characters per token


def test_history_within_budget_is_left_alone():
    history = [msg("user", 10), msg("assistant", 10), msg("user", 10)]
    trim_history(history, 30, count_tokens=len)
    assert len(history) == 3


def test_oldest_messages_are_dropped_first():
    history = [msg("user", 10), msg("assistant", 20), msg("user", 30), msg("assistant", 40)]
    trim_history(history, 70, count_tokens=len)  # 100 total -> drop 10 and 20 -> 70
    assert sizes(history) == [("user", 30), ("assistant", 40)]


def test_trim_stops_as_soon_as_the_history_fits():
    history = [msg("user", 10), msg("assistant", 10), msg("user", 10), msg("assistant", 10)]
    trim_history(history, 30, count_tokens=len)  # 40 -> 30 after one drop, then an assistant
    # dropping one message leaves "assistant" first, which is then removed too
    assert sizes(history) == [("user", 10), ("assistant", 10)]


def test_history_never_starts_with_an_assistant_message():
    history = [msg("user", 50), msg("assistant", 10), msg("user", 10), msg("assistant", 10)]
    trim_history(history, 31, count_tokens=len)  # dropping the 50 leaves 30 and the assistant first
    assert history[0]["role"] == "user"
    assert sizes(history) == [("user", 10), ("assistant", 10)]


def test_the_newest_message_stays_even_when_it_alone_is_over_budget():
    history = [msg("user", 10), msg("assistant", 10), msg("user", 500)]
    trim_history(history, 100, count_tokens=len)
    assert sizes(history) == [("user", 500)]


def test_a_single_message_over_budget_is_kept():
    history = [msg("user", 500)]
    trim_history(history, 10, count_tokens=len)
    assert sizes(history) == [("user", 500)]


def test_consecutive_user_messages_are_trimmed_one_at_a_time():
    # a failed LLM turn leaves a user message without a reply
    history = [msg("user", 10), msg("user", 20), msg("assistant", 30), msg("user", 40)]
    trim_history(history, 90, count_tokens=len)  # 100 -> 90 after dropping the first user
    assert sizes(history) == [("user", 20), ("assistant", 30), ("user", 40)]


def test_empty_history_is_fine():
    history: list[dict[str, str]] = []
    trim_history(history, 10, count_tokens=len)
    assert history == []


def test_default_counter_is_the_estimate():
    history = [msg("user", 400), msg("assistant", 400), msg("user", 40)]
    trim_history(history, 150)  # ~100 + ~100 + ~10: the oldest goes, then the orphan reply
    assert sizes(history) == [("user", 40)]


def test_build_messages_puts_the_system_prompt_first():
    history = [msg("user", 3), msg("assistant", 4)]
    messages = build_messages("be brief", history)
    assert messages[0] == {"role": "system", "content": "be brief"}
    assert messages[1:] == history


def test_build_messages_with_empty_history_is_just_the_system_prompt():
    assert build_messages("be brief", []) == [{"role": "system", "content": "be brief"}]


def test_build_messages_returns_copies():
    history = [msg("user", 3)]
    messages = build_messages("be brief", history)
    messages[1]["content"] = "changed"
    history.append(msg("assistant", 2))
    assert history[0]["content"] == "xxx"
    assert len(messages) == 2
