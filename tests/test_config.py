"""Tests for backend/config.py (PLAN T1.2).

Each test writes a small YAML to tmp_path and passes `environ={...}`, so the real .env
and os.environ are never read.
"""

from pathlib import Path

import pytest

from backend.config import CONFIG_PATH, ConfigError, load_settings

YAML = """
asr:
  default_model: model-b
  models:
    model-a:
      adapter: pkg.mod.AdapterA
      model_id: org/a
      options:
        device: cpu
        dtype: float32
    model-b:
      adapter: pkg.mod.AdapterB
      model_id: org/b
llm:
  model: gemini/test-model
  api_base: null
  system_prompt: be brief
  max_history_tokens: 123
limits:
  max_turn_seconds: 12
debug:
  audio_dir: out_audio
"""


def write(tmp_path: Path, text: str = YAML) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


def test_loads_every_field_from_a_sample_config(tmp_path):
    s = load_settings(write(tmp_path), environ={})

    assert list(s.asr_models) == ["model-a", "model-b"]
    assert s.asr_models["model-a"].adapter == "pkg.mod.AdapterA"
    assert s.asr_models["model-a"].model_id == "org/a"
    assert s.asr_models["model-a"].options == {"device": "cpu", "dtype": "float32"}
    assert s.asr_models["model-b"].options == {}
    assert s.default_asr_model == "model-b"
    assert s.llm.model == "gemini/test-model"
    assert s.llm.api_base is None
    assert s.llm.system_prompt.strip() == "be brief"
    assert s.llm.max_history_tokens == 123
    assert s.max_turn_seconds == 12
    assert s.debug_save_audio is False
    assert s.debug_audio_dir.is_absolute()
    assert s.debug_audio_dir == CONFIG_PATH.parent / "out_audio"


def test_env_overrides_llm_model_and_api_base(tmp_path):
    env = {"LLM_MODEL": "hosted_vllm/x", "LLM_API_BASE": "http://localhost:9000/v1"}
    s = load_settings(write(tmp_path), environ=env)
    assert s.llm.model == "hosted_vllm/x"
    assert s.llm.api_base == "http://localhost:9000/v1"


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", True), ("true", True), ("YES", True), ("0", False), ("false", False), ("", False)],
)
def test_debug_flag_parsing(tmp_path, value, expected):
    s = load_settings(write(tmp_path), environ={"DEBUG_SAVE_AUDIO": value})
    assert s.debug_save_audio is expected


def test_debug_flag_unset_is_off(tmp_path):
    assert load_settings(write(tmp_path), environ={}).debug_save_audio is False


def test_debug_flag_rejects_garbage(tmp_path):
    with pytest.raises(ConfigError, match="DEBUG_SAVE_AUDIO"):
        load_settings(write(tmp_path), environ={"DEBUG_SAVE_AUDIO": "banana"})


def test_default_model_must_exist(tmp_path):
    bad = YAML.replace("default_model: model-b", "default_model: nope")
    with pytest.raises(ConfigError, match="nope"):
        load_settings(write(tmp_path, bad), environ={})


@pytest.mark.parametrize("missing", ["adapter: pkg.mod.AdapterB", "model_id: org/b"])
def test_model_entry_needs_adapter_and_model_id(tmp_path, missing):
    bad = YAML.replace(f"      {missing}\n", "")
    with pytest.raises(ConfigError, match="model-b"):
        load_settings(write(tmp_path, bad), environ={})


def test_missing_file_and_empty_models_raise(tmp_path):
    with pytest.raises(ConfigError):
        load_settings(tmp_path / "nope.yaml", environ={})
    empty = "asr:\n  default_model: x\n  models: {}\n"
    with pytest.raises(ConfigError):
        load_settings(write(tmp_path, empty), environ={})


def test_the_real_config_yaml_loads():
    s = load_settings(environ={})
    assert s.default_asr_model in s.asr_models
    assert s.max_turn_seconds == 30
