"""
Tests for YunoIdentity and YunoGenerator (no model load required)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from yuno_llm.config import YunoConfig
from yuno_llm.identity import YunoIdentity


def test_identity_default():
    identity = YunoIdentity()
    assert identity.name == "YUNO"
    assert len(identity.system_prompt) > 10


def test_identity_from_config():
    cfg = YunoConfig()
    cfg.identity.name = "YUNO-TEST"
    identity = YunoIdentity(cfg)
    assert identity.name == "YUNO-TEST"


def test_apply_system_prompt_prepends():
    identity = YunoIdentity()
    messages = [{"role": "user", "content": "Hello"}]
    result = identity.apply_system_prompt(messages)
    assert result[0]["role"] == "system"
    assert result[1]["role"] == "user"
    assert result[1]["content"] == "Hello"


def test_apply_system_prompt_keeps_existing():
    identity = YunoIdentity()
    messages = [
        {"role": "system", "content": "Custom system"},
        {"role": "user", "content": "Hi"},
    ]
    result = identity.apply_system_prompt(messages)
    assert result[0]["content"] == "Custom system"  # Kept unchanged


def test_apply_system_prompt_override():
    identity = YunoIdentity()
    messages = [
        {"role": "system", "content": "Old system"},
        {"role": "user", "content": "Hi"},
    ]
    result = identity.apply_system_prompt(messages, override_system="New system")
    assert result[0]["content"] == "New system"


def test_greeting():
    identity = YunoIdentity()
    greeting = identity.greeting()
    assert "YUNO" in greeting
