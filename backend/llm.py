"""Streams a reply from the LLM through LiteLLM (SPEC 6.5, PLAN T3.2).

The session only calls `LLMClient.stream(messages)` and never imports litellm itself, so
tests can swap the client for a fake and the provider stays a config value.
"""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from backend.config import LLMSettings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """The LLM request failed, before or during the stream."""


class LLMClient:
    """Wraps `litellm.acompletion` as an async generator of text deltas.

    `completion` is for tests: any async callable with the signature of `litellm.acompletion`.
    When it is None the real litellm is used. Import litellm lazily (inside `stream`, not at
    module level): the first import takes many seconds and tests should not pay for it.
    """

    def __init__(
        self, settings: LLMSettings, completion: Callable[..., Awaitable[Any]] | None = None
    ) -> None:
        self._settings = settings
        self._completion = completion

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """Yield the reply text piece by piece.

        Input:  messages - the full chat as `[{"role": "system"|"user"|"assistant",
                "content": "..."}]`, system prompt first.
        Output: non-empty text deltas, in order.
        Raises: LLMError("LLM request failed") on any failure, including one in the middle of
                the stream (deltas already yielded stay valid; the caller decides what to do).
        """
        completion = self._completion
        if completion is None:
            import litellm  # slow first import, so only when a real request is made

            completion = litellm.acompletion

        kwargs: dict[str, Any] = {
            "model": self._settings.model,
            "messages": messages,
            "stream": True,
        }
        if self._settings.api_base is not None:
            kwargs["api_base"] = self._settings.api_base

        try:
            async for chunk in await completion(**kwargs):
                if not chunk.choices:  # some providers end with a usage-only chunk
                    continue
                text = chunk.choices[0].delta.content
                if text:
                    yield text
        except Exception as exc:
            # The message reaches the user; the original may hold internals, so only log it.
            # CancelledError is not an Exception and passes through, which cancels a stream.
            log.exception("LLM request failed")
            raise LLMError("LLM request failed") from exc
