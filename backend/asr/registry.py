"""Maps a config key to an adapter and keeps loaded models around (SPEC 6.4, PLAN T2.2).

Adding a model = one adapter file + one config entry. The config entry names the
adapter class by dotted path, so this file never needs to change and heavy
libraries (transformers, NeMo) are only imported when a model is first used.
"""

import importlib
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from backend.asr.base import ASRError, ASRModel, UnknownASRModel


@dataclass(frozen=True)
class ASRModelConfig:
    adapter: str  # dotted path, e.g. "backend.asr.whisper.WhisperASR"
    model_id: str  # checkpoint id, e.g. "openai/whisper-large-v3"
    options: dict[str, Any] = field(default_factory=dict)  # adapter-specific: device, dtype, ...


class ASRRegistry:
    """Adapter contract: `Adapter(name=<key>, model_id=<model_id>, **options)`."""

    def __init__(self, configs: Mapping[str, ASRModelConfig]) -> None:
        self._configs = dict(configs)
        self._models: dict[str, ASRModel] = {}
        # Created up front so two threads never race to create the same lock.
        self._locks = {name: threading.Lock() for name in self._configs}

    def available(self) -> list[str]:
        return list(self._configs)

    def is_loaded(self, name: str) -> bool:
        return name in self._models

    def get(self, name: str) -> tuple[ASRModel, float]:
        """Return a ready model and the load time of this call (0.0 if already loaded)."""
        if name not in self._configs:
            raise UnknownASRModel(f"unknown ASR model: {name!r}")
        model = self._models.get(name)
        if model is not None:
            return model, 0.0
        with self._locks[name]:
            model = self._models.get(name)
            if model is not None:
                return model, 0.0
            try:
                model = self._build(name)
                start = time.perf_counter()
                model.load()
                load_ms = (time.perf_counter() - start) * 1000.0
            except ASRError:
                raise
            except Exception as exc:
                raise ASRError(f"failed to load ASR model {name!r}: {exc}") from exc
            self._models[name] = model
            return model, load_ms

    def _build(self, name: str) -> ASRModel:
        config = self._configs[name]
        module_path, _, class_name = config.adapter.rpartition(".")
        try:
            module = importlib.import_module(module_path)
            adapter_cls = getattr(module, class_name)
        except (ImportError, AttributeError, ValueError) as exc:
            raise ASRError(f"cannot import adapter {config.adapter!r}: {exc}") from exc
        return adapter_cls(name=name, model_id=config.model_id, **config.options)
