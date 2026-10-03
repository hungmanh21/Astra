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

from backend.asr.registry import ASRModelConfig

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


class ConfigError(ValueError):
    """The config is missing something or contradicts itself. The message names the key."""


@dataclass(frozen=True)
class LLMSettings:
    model: str  # LiteLLM model string, e.g. "gemini/<name>" or "hosted_vllm/<name>"
    api_base: str | None
    system_prompt: str
    max_history_tokens: int


@dataclass(frozen=True)
class Settings:
    asr_models: dict[str, ASRModelConfig]  # in config order
    default_asr_model: str
    llm: LLMSettings
    max_turn_seconds: float
    debug_save_audio: bool
    debug_audio_dir: Path  # absolute, resolved against the repo root


def _parse_flag(name: str, value: str) -> bool:
    value = value.strip().lower()
    if value in ("1", "true", "yes"):
        return True
    if value in ("", "0", "false", "no"):
        return False
    raise ConfigError(f"{name} must be 1/true/yes or 0/false/no, got {value!r}")


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

    llm_model = environ.get("LLM_MODEL", config["llm"]["model"])
    llm_api_base = environ.get("LLM_API_BASE", config["llm"].get("api_base"))

    llm_settings = LLMSettings(
        model=llm_model,
        api_base=llm_api_base,
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
    )
