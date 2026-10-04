"""Conversation history helpers (SPEC 6.5, FR-11, PLAN T3.3).

History is a plain list of `{"role": "user" | "assistant", "content": str}` that the session
owns. The system prompt is not stored in it: it is added in front when a request is built.
"""

from collections.abc import Callable

Message = dict[str, str]


def estimate_tokens(text: str) -> int:
    """Cheap token estimate, no tokenizer download: about 4 characters per token.

    0 for an empty string, at least 1 otherwise. An estimate is enough because the budget is a
    safety net, not an exact limit.
    """
    if not text:
        return 0

    return len(text) // 4 + 1


def trim_history(
    history: list[Message],
    max_tokens: int,
    count_tokens: Callable[[str], int] = estimate_tokens,
) -> None:
    """Drop the oldest messages, in place, until the history fits `max_tokens`.

    The newest message is never dropped, even if it alone is over the budget. If the first
    message left is an assistant one (its question was dropped), it goes too, so the history
    starts with a user message. A failed LLM turn leaves a user message with no reply, so two
    user messages in a row are normal; messages are dropped one at a time, not in pairs.
    """
    total_tokens = sum(count_tokens(m["content"]) for m in history)
    while total_tokens > max_tokens and len(history) > 1:
        total_tokens -= count_tokens(history[0]["content"])
        history.pop(0)

    if len(history) > 1 and history[0]["role"] == "assistant":
        total_tokens -= count_tokens(history[0]["content"])
        history.pop(0)


def build_messages(system_prompt: str, history: list[Message]) -> list[Message]:
    """The request for the LLM: the system prompt first, then the history, as new dicts.

    Copies, so a later change to `history` cannot alter a request that is still running.
    """
    return [{"role": "system", "content": system_prompt}, *[m.copy() for m in history]]
