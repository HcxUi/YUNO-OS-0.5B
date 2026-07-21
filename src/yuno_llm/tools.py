"""
YUNO Safe Tool Use Module
==========================
Defines local Tools and the Human-In-The-Loop Safe Tool Executor.
"""

import os
import logging
from typing import Dict, List, Any, Callable, Tuple

logger = logging.getLogger("yuno_llm.tools")


class YunoTool:
    """
    Base class representing a system tool executable by YUNO-LLM OS.
    """

    def __init__(self, name: str, description: str, requires_write_permission: bool = False):
        self.name = name
        self.description = description
        self.requires_write_permission = requires_write_permission

    def execute(self, **kwargs) -> Any:
        raise NotImplementedError("Each tool must implement its own execute logic.")


# ── 1. Read File Tool (Auto-Approve Safe) ────────────────────────────────────

class ReadFileTool(YunoTool):
    def __init__(self):
        super().__init__(
            name="read_file",
            description="Reads the contents of a local file in the workspace.",
            requires_write_permission=False
        )

    def execute(self, file_path: str) -> str:
        try:
            if not os.path.exists(file_path):
                return f"[ERROR] File not found: {file_path}"
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception as e:
            return f"[ERROR] Failed to read file {file_path}: {e}"


# ── 2. Write File Tool (Explicit Permission Required) ───────────────────────

class WriteFileTool(YunoTool):
    def __init__(self):
        super().__init__(
            name="write_file",
            description="Writes content or modifications to a local file in the workspace.",
            requires_write_permission=True
        )

    def execute(self, file_path: str, content: str) -> str:
        try:
            # Ensure parent directories exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"[SUCCESS] File written successfully: {file_path}"
        except Exception as e:
            return f"[ERROR] Failed to write file {file_path}: {e}"


# ── 3. Tool Registry & Executor Loop ──────────────────────────────────────────

class YunoToolRegistry:
    """
    Manages all available tools and enforces human-in-the-loop permission queries
    prior to executing state-modifying actions.
    """

    def __init__(self, config=None, console_input_fn: Callable[[str], str] = input):
        self.config = config
        self.console_input_fn = console_input_fn
        self._tools: Dict[str, YunoTool] = {}

        # Register default tools
        self.register_tool(ReadFileTool())
        self.register_tool(WriteFileTool())

    def register_tool(self, tool: YunoTool) -> None:
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def list_tools(self) -> List[Tuple[str, str]]:
        return [(t.name, t.description) for t in self._tools.values()]

    def run_tool(self, tool_name: str, **kwargs) -> str:
        """
        Executes a registered tool. Prompts user for approval if the tool
        requires write permission and config requires permission check.
        """
        if tool_name not in self._tools:
            return f"[ERROR] Tool '{tool_name}' is not registered."

        tool = self._tools[tool_name]

        # Enforce HITL (Human-In-The-Loop) check if tool modifies system state
        if tool.requires_write_permission:
            require_perm = True
            if self.config:
                require_perm = getattr(self.config.tools, "require_permission", True)

            if require_perm:
                print(f"\n  [ALERT] YUNO is requesting permission to execute: '{tool_name}'")
                print(f"  Description: {tool.description}")
                print(f"  Arguments:   {kwargs}")
                
                # Demand explicit console approval
                approval = self.console_input_fn("  Approve action? (yes/no): ").strip().lower()
                if approval not in ("yes", "y"):
                    print("  [BLOCKED] User rejected action execution.")
                    return "[ERROR] Action rejected by user permission policy."

        logger.info(f"Executing tool {tool_name} with args: {kwargs}")
        return str(tool.execute(**kwargs))
