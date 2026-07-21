"""
Unit tests for YunoMemory
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from yuno_llm.memory import YunoMemory


def test_short_term_memory():
    mem = YunoMemory()
    mem.add_chat_turn("user", "Hello YUNO!")
    mem.add_chat_turn("assistant", "Hi human!")

    history = mem.get_chat_history()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hello YUNO!"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "Hi human!"

    mem.clear_short_term()
    assert len(mem.get_chat_history()) == 0


def test_long_term_memory_facts():
    mem = YunoMemory()
    mem.remember_personal_fact("user_name", "Piyush")
    assert mem.get_personal_fact("user_name") == "Piyush"

    # Delete fact
    assert mem.forget_personal_fact("user_name") is True
    assert mem.get_personal_fact("user_name") is None


def test_project_memory_facts():
    mem = YunoMemory()
    mem.update_project_fact("active_branch", "vision-upgrade")
    assert mem.get_project_fact("active_branch") == "vision-upgrade"


def test_compile_memory_context():
    mem = YunoMemory()
    mem.remember_personal_fact("user_name", "Piyush")
    mem.update_project_fact("active_branch", "vision-upgrade")

    ctx = mem.compile_memory_context()
    assert "[Long-Term Memory Facts]" in ctx
    assert "- user_name: Piyush" in ctx
    assert "[Project Memory Facts]" in ctx
    assert "- active_branch: vision-upgrade" in ctx

    # Clean up
    mem.forget_personal_fact("user_name")
