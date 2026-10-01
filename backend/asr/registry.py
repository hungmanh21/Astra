"""Maps a config key to an adapter and keeps loaded models around (SPEC 6.4, PLAN T2.2).

Adding a model = one adapter file + one config entry. The config entry names the
adapter class by dotted path, so this file never needs to change and heavy
libraries (transformers, NeMo) are only imported when a model is first used.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from backend.asr.base import ASRModel


@dataclass(frozen=True)
class ASRModelConfig:
    adapter: str  # dotted path, e.g. "backend.asr.whisper.WhisperASR"
    model_id: str  # checkpoint id, e.g. "openai/whisper-large-v3"
    options: dict[str, Any] = field(default_factory=dict)  # adapter-specific: device, dtype, ...


class ASRRegistry:
    """Adapter contract: `Adapter(name=<key>, model_id=<model_id>, **options)`."""

    def __init__(self, configs: Mapping[str, ASRModelConfig]) -> None:
        """TODO(you): store the config and set up empty caches.

        Input:  configs - registry key -> ASRModelConfig, in config order,
                e.g. {"whisper-large-v3": ASRModelConfig(...)}.
        Output: None.
        State to keep:
          - the configs (copy them),
          - a dict of loaded models: key -> ASRModel,
          - one threading.Lock per key (create them all here, so there is no race
            creating locks later).
        Don't build or load any adapter here.
        """
        raise NotImplementedError

    def available(self) -> list[str]:
        """TODO(you): the keys the UI can offer.

        Input:  none.
        Output: list[str] of registry keys, in config order. Sent in the `session` message.
        """
        raise NotImplementedError

    def is_loaded(self, name: str) -> bool:
        """TODO(you): whether this model is already resident.

        Input:  name - a registry key.
        Output: bool. False for a key that exists but is not loaded yet.
        Raises: nothing for unknown keys; just return False.
        """
        raise NotImplementedError

    def get(self, name: str) -> tuple[ASRModel, float]:
        """TODO(you): return a ready-to-use model, loading it on first use.

        Input:  name - a registry key.
        Output: (model, load_ms). load_ms is how long load() took in this call,
                or 0.0 if the model was already loaded.
        Raises: UnknownASRModel if `name` is not in the config.
                ASRError if _build() or load() fails (wrap other exceptions).

        Must be safe to call from several threads at once: two threads asking for the
        same unloaded model must cause ONE load() call.

        Steps:
          1. If `name` is unknown, raise UnknownASRModel.
          2. Fast path: already loaded -> (model, 0.0).
          3. Take the lock for this key; check again (another thread may have loaded it).
          4. _build(name), time model.load() with time.perf_counter(), cache it, return.
        """
        raise NotImplementedError

    def _build(self, name: str) -> ASRModel:
        """TODO(you): create the adapter instance without loading it.

        Input:  name - a registry key that exists in the config.
        Output: an ASRModel instance, built as Adapter(name=name, model_id=..., **options).
        Raises: ASRError if the dotted path cannot be imported or the class is missing.

        Steps:
          1. Split config.adapter on the last "." into module path and class name.
          2. importlib.import_module(module path), then getattr(module, class name).
          3. Instantiate with the keyword arguments from the contract above.
        """
        raise NotImplementedError
