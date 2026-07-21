"""
Unit tests for YunoUpdater (Self-System / Lalam)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from yuno_llm.updater import YunoUpdater
from yuno_llm.config import YunoConfig


def test_updater_config_defaults():
    updater = YunoUpdater()
    assert updater.enabled is True
    assert updater.auto_git_pull is True
    assert updater.check_model_updates is False


def test_updater_disabled():
    cfg = YunoConfig()
    # Mock config properties for updater
    class DummyUpdaterConfig:
        enabled = False
        auto_git_pull = False
        check_model_updates = False
        timeout_seconds = 2

    cfg.updater = DummyUpdaterConfig()
    
    updater = YunoUpdater(cfg)
    assert updater.enabled is False
    
    # Run auto update, it should return early with empty summary
    res = updater.run_auto_update()
    assert res["internet_available"] is False
    assert res["code_updated"] is False


def test_check_internet_connection_mocked():
    updater = YunoUpdater()
    
    # Mock check_internet_connection to simulate offline
    updater.check_internet_connection = lambda: False
    res = updater.run_auto_update()
    assert res["internet_available"] is False

    # Mock check_internet_connection to simulate online
    updater.check_internet_connection = lambda: True
    updater.update_code = lambda: True
    updater.update_model = lambda: True
    res = updater.run_auto_update()
    assert res["internet_available"] is True
    assert res["code_updated"] is True
    assert res["model_checked"] is True
