"""
Unit tests for YunoToolRegistry and SafeToolExecutor
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from yuno_llm.tools import YunoToolRegistry, YunoTool


class DummyTool(YunoTool):
    def __init__(self):
        super().__init__(
            name="dummy_tool",
            description="Dummy tool description",
            requires_write_permission=False
        )

    def execute(self, val: str) -> str:
        return f"received {val}"


class WriteDummyTool(YunoTool):
    def __init__(self):
        super().__init__(
            name="write_dummy",
            description="Modifies state dummy",
            requires_write_permission=True
        )

    def execute(self, val: str) -> str:
        return f"wrote {val}"


def test_tool_registration():
    registry = YunoToolRegistry()
    registry.register_tool(DummyTool())

    tools = registry.list_tools()
    tool_names = [t[0] for t in tools]
    assert "dummy_tool" in tool_names


def test_tool_execute_no_permission_required():
    registry = YunoToolRegistry()
    registry.register_tool(DummyTool())

    res = registry.run_tool("dummy_tool", val="hello")
    assert res == "received hello"


def test_tool_execute_permission_approved():
    # Mock user input to approve execution
    def mock_approve(prompt):
        return "yes"

    registry = YunoToolRegistry(console_input_fn=mock_approve)
    registry.register_tool(WriteDummyTool())

    res = registry.run_tool("write_dummy", val="hello")
    assert res == "wrote hello"


def test_tool_execute_permission_rejected():
    # Mock user input to reject execution
    def mock_reject(prompt):
        return "no"

    registry = YunoToolRegistry(console_input_fn=mock_reject)
    registry.register_tool(WriteDummyTool())

    res = registry.run_tool("write_dummy", val="hello")
    assert "[ERROR] Action rejected by user permission policy." in res
