"""Typed settings from config.yaml and .env (SPEC 6.2, PLAN T1.2).

config.yaml holds model names and defaults. .env holds secrets and the debug flag.
Nothing else in the backend reads either file or os.environ directly.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

from backend.asr.base import GENERIC_STRATEGIES
from backend.asr.registry import ASRModelConfig

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


class ConfigError(ValueError):
    """The config is missing something or contradicts itself. The message names the key."""


# `hosted_vllm/auto`: ask the endpoint which model it serves instead of naming one (PLAN T5.4).
AUTO_MODEL = "auto"


def is_auto_model(model: str) -> bool:
    """True when the part after the provider prefix is the AUTO_MODEL sentinel."""
    return model.partition("/")[2] == AUTO_MODEL


@dataclass(frozen=True)
class LLMSettings:
    model: str  # LiteLLM model string, e.g. "gemini/<name>" or "hosted_vllm/<name>"
    api_base: str | None
    system_prompt: str
    max_history_tokens: int
    api_key: str | None = None  # LLM_API_KEY, for an endpoint that wants one; None = not sent


@dataclass(frozen=True)
class StreamingSettings:
    """The `asr.streaming` section (SPEC_v1 6.5). A missing key keeps its default here."""

    default_strategy: str = "redecode"  # for models without native streaming
    partial_interval_ms: int = 500  # least new audio between two partial decodes
    partial_min_audio_ms: int = 1000  # no partial before this much audio
    chunk_ms: int = 2000  # chunk length for the `chunked` strategy


@dataclass(frozen=True)
class Settings:
    asr_models: dict[str, ASRModelConfig]  # in config order
    default_asr_model: str
    llm: LLMSettings
    max_turn_seconds: float
    debug_save_audio: bool
    debug_audio_dir: Path  # absolute, resolved against the repo root
    asr_streaming: StreamingSettings = StreamingSettings()
    release_tail_ms: int = 400  # the UI keeps recording this long after the button is released


def _release_tail_ms(value: object) -> int:
    """`limits.release_tail_ms`: missing keeps the default, otherwise 0 to 3000."""
    if value is None:
        return Settings.release_tail_ms
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 3000:
        raise ConfigError(
            f"limits.release_tail_ms must be a whole number from 0 to 3000, got {value!r}"
        )
    return value


def _parse_flag(name: str, value: str) -> bool:
    value = value.strip().lower()
    if value in ("1", "true", "yes"):
        return True
    if value in ("", "0", "false", "no"):
        return False
    raise ConfigError(f"{name} must be 1/true/yes or 0/false/no, got {value!r}")


def _env(environ: Mapping[str, str], name: str) -> str | None:
    """The stripped value of `name`, or None when it is missing or blank."""
    return environ.get(name, "").strip() or None


def _streaming_settings(section: Mapping | None) -> StreamingSettings:
    """Turn the `asr.streaming` YAML section into a StreamingSettings (PLAN_v1 S1.2).

    Input:  section - the parsed `asr.streaming` mapping, or None when the YAML has no
            such section (older configs, the test YAML). Any key may be missing.
    Output: StreamingSettings. A missing key keeps the dataclass default.
    Raises: ConfigError, naming the key (e.g. "asr.streaming.chunk_ms"), when
              - partial_interval_ms, partial_min_audio_ms or chunk_ms is not a
                positive int (0, -5, "fast" are all errors),
              - default_strategy is not in GENERIC_STRATEGIES. "native" and "trimmed" are
                refused too: they only exist for some models, and this default is for
                models that have neither capability.
    """
    defaults = StreamingSettings()
    section = section or {}

    durations = {}
    for key in ("partial_interval_ms", "partial_min_audio_ms", "chunk_ms"):
        value = section.get(key, getattr(defaults, key))
        # bool is an int subclass, so `chunk_ms: true` would otherwise pass.
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigError(f"asr.streaming.{key} must be a positive int, got {value!r}")
        durations[key] = value

    default_strategy = section.get("default_strategy", defaults.default_strategy)
    if default_strategy not in GENERIC_STRATEGIES:
        raise ConfigError(
            f"asr.streaming.default_strategy must be one of {GENERIC_STRATEGIES}, "
            f"got {default_strategy!r}"
        )

    return StreamingSettings(default_strategy=default_strategy, **durations)


def load_settings(
    config_path: Path = CONFIG_PATH, environ: Mapping[str, str] | None = None
) -> Settings:
    """Read config.yaml and the environment into a Settings.

    `environ=None` loads .env and uses os.environ; tests pass a plain dict instead.
    Raises ConfigError for a missing file, empty or inconsistent `asr` section, or a
    DEBUG_SAVE_AUDIO value that is not a clear yes/no.
    """
    if environ is None:
        load_dotenv()
        environ = os.environ

    try:
        with config_path.open(encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except OSError as e:
        raise ConfigError(f"cannot read {config_path}: {e}") from e

    models = config["asr"]["models"]
    if not models:
        raise ConfigError("asr.models is empty")

    for name, model in models.items():
        if "adapter" not in model or "model_id" not in model:
            raise ConfigError(f"asr.models.{name} needs adapter and model_id")

    default_model = config["asr"]["default_model"]
    if default_model not in models:
        raise ConfigError(f"asr.default_model {default_model!r} is not in asr.models")

    # now load the config of each model under the asr.models key into an ASRModelConfig
    asr_models = {
        name: ASRModelConfig(
            adapter=model["adapter"],
            model_id=model["model_id"],
            options=model.get("options") or {},
        )
        for name, model in config["asr"]["models"].items()
    }

    # An empty line in .env (`LLM_MODEL=`) means "not set", not "use an empty string".
    llm_model = _env(environ, "LLM_MODEL") or config["llm"]["model"]
    llm_api_base = _env(environ, "LLM_API_BASE") or config["llm"].get("api_base")

    if is_auto_model(llm_model) and llm_api_base is None:
        raise ConfigError(f"llm.model {llm_model!r} needs llm.api_base (or LLM_API_BASE) to ask")

    llm_settings = LLMSettings(
        model=llm_model,
        api_base=llm_api_base,
        api_key=_env(environ, "LLM_API_KEY"),
        system_prompt=config["llm"]["system_prompt"],
        max_history_tokens=config["llm"]["max_history_tokens"],
    )

    return Settings(
        asr_models=asr_models,
        default_asr_model=default_model,
        llm=llm_settings,
        max_turn_seconds=config["limits"]["max_turn_seconds"],
        debug_save_audio=_parse_flag("DEBUG_SAVE_AUDIO", environ.get("DEBUG_SAVE_AUDIO", "")),
        debug_audio_dir=CONFIG_PATH.parent / config["debug"]["audio_dir"],
        asr_streaming=_streaming_settings(config["asr"].get("streaming")),
        release_tail_ms=_release_tail_ms(config["limits"].get("release_tail_ms")),
    )
