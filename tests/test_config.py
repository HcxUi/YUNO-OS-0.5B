"""
Tests for YunoConfig
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from yuno_llm.config import YunoConfig, YunoGenerationConfig


def test_default_config():
    cfg = YunoConfig()
    assert cfg.base_model == "Qwen/Qwen3-0.6B"
    assert cfg.identity.name == "YUNO"
    assert cfg.generation.temperature == 0.7
    assert cfg.hardware.device == "auto"


def test_from_yaml():
    cfg = YunoConfig.from_yaml(str(ROOT / "config" / "yuno_config.yaml"))
    assert cfg.identity.name == "YUNO"
    assert isinstance(cfg.generation.temperature, float)
    assert 0 < cfg.generation.temperature <= 2.0


def test_generation_config_defaults():
    gen = YunoGenerationConfig()
    assert gen.max_new_tokens == 512
    assert gen.do_sample is True
    assert gen.top_p == 0.9


def test_to_dict():
    cfg = YunoConfig()
    d = cfg.to_dict()
    assert "identity" in d
    assert "generation" in d
    assert "hardware" in d
    assert d["identity"]["name"] == "YUNO"


def test_repr():
    cfg = YunoConfig()
    r = repr(cfg)
    assert "YunoConfig" in r
    assert "YUNO" in r
