"""Streams a reply from the LLM through LiteLLM (SPEC 6.5, PLAN T3.2).

The session only calls `LLMClient.stream(messages)` and never imports litellm itself, so
tests can swap the client for a fake and the provider stays a config value.
"""

import asyncio
import json
import logging
import urllib.request
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from backend.config import LLMSettings, is_auto_model

log = logging.getLogger(__name__)

# Lists the model ids an OpenAI-compatible endpoint serves: (api_base, api_key) -> ids.
ModelLister = Callable[[str, str | None], Awaitable[list[str]]]


class LLMError(RuntimeError):
    """The LLM request failed, before or during the stream."""


def _summary(exc: BaseException, limit: int = 300) -> str:
    """`ExcType: first part of the message` on one line, for logs."""
    text = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {text[:limit]}{'...' if len(text) > limit else ''}"


async def fetch_model_ids(
    api_base: str, api_key: str | None = None, timeout_s: float = 5.0
) -> list[str]:
    """Ask an OpenAI-compatible endpoint (vLLM) which models it serves.

    Input:  api_base - the same value as `llm.api_base`, e.g. "http://host:8000/v1".
            api_key - sent as a bearer token when not None.
    Output: the model ids, in the order the endpoint lists them.
    Raises: any exception on a connection error, a timeout, a non-200 status or a body that is
            not the expected JSON. The caller turns it into an `LLMError`.
    """

    def fetch() -> list[str]:
        request = urllib.request.Request(f"{api_base.rstrip('/')}/models")
        if api_key is not None:
            request.add_header("Authorization", f"Bearer {api_key}")
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = json.load(response)
        return [item["id"] for item in body["data"]]

    # urllib blocks, so it runs in a thread and the event loop (other sessions) keeps going.
    return await asyncio.to_thread(fetch)


class LLMClient:
    """Wraps `litellm.acompletion` as an async generator of text deltas.

    `completion` is for tests: any async callable with the signature of `litellm.acompletion`.
    When it is None the real litellm is used. Import litellm lazily (inside `stream`, not at
    module level): the first import takes many seconds and tests should not pay for it.
    `list_models` is for tests too: it defaults to `fetch_model_ids`.
    """

    def __init__(
        self,
        settings: LLMSettings,
        completion: Callable[..., Awaitable[Any]] | None = None,
        list_models: ModelLister = fetch_model_ids,
    ) -> None:
        self._settings = settings
        self._completion = completion
        self._list_models = list_models
        self._resolved: str | None = None  # the model name found by discovery, once known

    @property
    def model(self) -> str:
        """The model in use. Before discovery has run on an `auto` model, the configured string."""
        return self._resolved or self._settings.model

    async def _resolve_model(self) -> str:
        """The LiteLLM model string to send, running discovery first for an `auto` model.

        A failed lookup stores nothing, so the next turn asks again.
        """
        if self._resolved is not None:
            return self._resolved

        configured = self._settings.model
        if not is_auto_model(configured):
            return configured

        api_base = self._settings.api_base
        if api_base is None:  # load_settings rejects this; a hand-built setting could still do it
            raise RuntimeError("an auto model needs an api_base")

        ids = await self._list_models(api_base, self._settings.api_key)
        if not ids:
            raise RuntimeError("the endpoint serves no models")

        provider = configured.partition("/")[0]
        self._resolved = f"{provider}/{ids[0]}"
        log.info("using model %s (endpoint serves: %s)", self._resolved, ", ".join(ids))
        return self._resolved

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

            litellm.suppress_debug_info = True  # drops litellm's "Give Feedback / Get Help" banner
            completion = litellm.acompletion

        kwargs: dict[str, Any] = {"messages": messages, "stream": True}
        if self._settings.api_base is not None:
            kwargs["api_base"] = self._settings.api_base
        if self._settings.api_key is not None:
            kwargs["api_key"] = self._settings.api_key

        try:
            # Inside the try: a failed lookup is an LLMError like any other failure.
            kwargs["model"] = await self._resolve_model()
            async for chunk in await completion(**kwargs):
                if not chunk.choices:  # some providers end with a usage-only chunk
                    continue
                text = chunk.choices[0].delta.content
                if text:
                    yield text
        except Exception as exc:
            # The message reaches the user; the original may hold internals, so only log it.
            # One short line: provider errors chain several long tracebacks. The full one is
            # available at DEBUG. CancelledError is not an Exception and passes through.
            log.error("LLM request failed: %s", _summary(exc))
            log.debug("LLM request failure details", exc_info=True)
            raise LLMError("LLM request failed") from exc
