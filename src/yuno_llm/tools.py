"""
YUNO Safe Tool Use Module
==========================
Defines the full YUNO tool registry with Human-In-The-Loop (HITL) permission
enforcement and a complete audit log of every tool execution.

Permission Levels
-----------------
AUTO      → Executes immediately, no user prompt required (read-only, safe)
CONFIRM   → Displays what will happen and waits for a simple yes/no
EXPLICIT  → Requires the user to type "yes" explicitly (state-modifying)

Available Tools
---------------
  read_file      (AUTO)     — Read text content of a local file
  list_dir       (AUTO)     — List files and subdirectories
  search_files   (AUTO)     — Search text pattern across files
  summarize_file (AUTO)     — Read and summarize a document
  write_file     (EXPLICIT) — Write or overwrite a local file
  create_note    (EXPLICIT) — Create a new Markdown note
  run_script     (EXPLICIT) — Execute a Python script in a subprocess

Usage
-----
    registry = YunoToolRegistry(config)
    result = registry.run_tool("read_file", file_path="config.yaml")
    result = registry.run_tool("write_file", file_path="out.txt", content="Hello")
"""

import os
import re
import json
import logging
import subprocess
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("yuno_llm.tools")


# ── Permission Level ──────────────────────────────────────────────────────────

class ToolPermissionLevel(str, Enum):
    AUTO     = "AUTO"     # Read-only: no user prompt required
    CONFIRM  = "CONFIRM"  # Low-risk: show plan, simple y/n
    EXPLICIT = "EXPLICIT" # State-modifying: require typed "yes"


# ── Base Tool ─────────────────────────────────────────────────────────────────

class YunoTool(ABC):
    """
    Abstract base class for all YUNO system tools.

    Subclasses must implement:
        name        — unique string identifier
        description — human-readable description shown during approval prompt
        permission  — ToolPermissionLevel enum value
        execute()   — actual tool logic
    """

    name: str = ""
    description: str = ""
    permission: ToolPermissionLevel = ToolPermissionLevel.EXPLICIT

    @abstractmethod
    def execute(self, **kwargs) -> str:
        ...

    # Backwards-compat alias used by the old API
    @property
    def requires_write_permission(self) -> bool:
        return self.permission == ToolPermissionLevel.EXPLICIT


# ── Tool Implementations ──────────────────────────────────────────────────────

class ReadFileTool(YunoTool):
    """Read the text content of a local file."""
    name = "read_file"
    description = "Read the text content of a file in the local workspace."
    permission = ToolPermissionLevel.AUTO

    def execute(self, file_path: str, max_chars: int = 8000) -> str:
        path = Path(file_path)
        if not path.exists():
            return f"[ERROR] File not found: {file_path}"
        if not path.is_file():
            return f"[ERROR] Path is not a file: {file_path}"
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            if len(content) > max_chars:
                content = content[:max_chars] + f"\n\n[... truncated at {max_chars} chars ...]"
            return content
        except Exception as e:
            return f"[ERROR] Could not read {file_path}: {e}"


class ListDirTool(YunoTool):
    """List files and subdirectories in a local directory."""
    name = "list_dir"
    description = "List the contents (files and subdirectories) of a local directory."
    permission = ToolPermissionLevel.AUTO

    def execute(self, dir_path: str, show_hidden: bool = False) -> str:
        path = Path(dir_path)
        if not path.exists():
            return f"[ERROR] Directory not found: {dir_path}"
        if not path.is_dir():
            return f"[ERROR] Path is not a directory: {dir_path}"
        try:
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            lines = [f"Contents of: {path.resolve()}\n"]
            for entry in entries:
                if not show_hidden and entry.name.startswith("."):
                    continue
                if entry.is_dir():
                    lines.append(f"  📁 {entry.name}/")
                else:
                    size = entry.stat().st_size
                    size_str = f"{size:,} B" if size < 1024 else f"{size/1024:.1f} KB"
                    lines.append(f"  📄 {entry.name}  ({size_str})")
            return "\n".join(lines)
        except PermissionError:
            return f"[ERROR] Permission denied: {dir_path}"
        except Exception as e:
            return f"[ERROR] Could not list directory: {e}"


class SearchFilesTool(YunoTool):
    """Search for a text pattern across files in a directory."""
    name = "search_files"
    description = "Search for a keyword or pattern across files in a local directory."
    permission = ToolPermissionLevel.AUTO

    def execute(
        self,
        query: str,
        dir_path: str = ".",
        extensions: Optional[List[str]] = None,
        max_results: int = 20,
    ) -> str:
        search_dir = Path(dir_path)
        if not search_dir.exists():
            return f"[ERROR] Directory not found: {dir_path}"

        extensions = extensions or [".py", ".txt", ".md", ".yaml", ".yml", ".json"]
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        matches: List[str] = []

        try:
            for file_path in search_dir.rglob("*"):
                if not file_path.is_file():
                    continue
                if file_path.suffix not in extensions:
                    continue
                try:
                    text = file_path.read_text(encoding="utf-8", errors="ignore")
                    for i, line in enumerate(text.splitlines(), 1):
                        if pattern.search(line):
                            rel = file_path.relative_to(search_dir)
                            matches.append(f"  {rel}:{i}: {line.strip()}")
                            if len(matches) >= max_results:
                                break
                except Exception:
                    continue
                if len(matches) >= max_results:
                    break
        except Exception as e:
            return f"[ERROR] Search failed: {e}"

        if not matches:
            return f"[No results found for '{query}' in {dir_path}]"
        header = f"Search results for '{query}' in {dir_path}:\n"
        return header + "\n".join(matches)


class SummarizeFileTool(YunoTool):
    """Read a file and return a compact summary (first N lines + size)."""
    name = "summarize_file"
    description = "Read and summarize the key contents of a local file."
    permission = ToolPermissionLevel.AUTO

    def execute(self, file_path: str, preview_lines: int = 30) -> str:
        path = Path(file_path)
        if not path.exists():
            return f"[ERROR] File not found: {file_path}"
        if not path.is_file():
            return f"[ERROR] Not a file: {file_path}"
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            total_lines = len(lines)
            size_kb = path.stat().st_size / 1024
            preview = "\n".join(lines[:preview_lines])
            suffix = (
                f"\n\n[... {total_lines - preview_lines} more lines ...]"
                if total_lines > preview_lines
                else ""
            )
            return (
                f"File: {file_path}\n"
                f"Size: {size_kb:.1f} KB | Lines: {total_lines}\n"
                f"{'─' * 50}\n"
                f"{preview}{suffix}"
            )
        except Exception as e:
            return f"[ERROR] Could not summarize {file_path}: {e}"


class WriteFileTool(YunoTool):
    """Write content to a local file (requires explicit user approval)."""
    name = "write_file"
    description = "Write or overwrite the content of a file in the local workspace."
    permission = ToolPermissionLevel.EXPLICIT

    def execute(self, file_path: str, content: str) -> str:
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return f"[SUCCESS] File written: {file_path} ({len(content)} chars)"
        except Exception as e:
            return f"[ERROR] Could not write {file_path}: {e}"


class CreateNoteTool(YunoTool):
    """Create a new Markdown note file (requires explicit user approval)."""
    name = "create_note"
    description = "Create a new Markdown (.md) note file with the given title and content."
    permission = ToolPermissionLevel.EXPLICIT

    def execute(self, title: str, content: str, notes_dir: str = "notes") -> str:
        try:
            safe_name = re.sub(r"[^\w\-_]", "_", title.strip()) + ".md"
            notes_path = Path(notes_dir)
            notes_path.mkdir(parents=True, exist_ok=True)
            note_file = notes_path / safe_name
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            full_content = f"# {title}\n\n*Created: {timestamp}*\n\n{content}\n"
            note_file.write_text(full_content, encoding="utf-8")
            return f"[SUCCESS] Note created: {note_file.resolve()}"
        except Exception as e:
            return f"[ERROR] Could not create note: {e}"


class RunScriptTool(YunoTool):
    """Execute a Python script in a sandboxed subprocess (requires explicit approval)."""
    name = "run_script"
    description = "Execute a Python (.py) script in a sandboxed subprocess."
    permission = ToolPermissionLevel.EXPLICIT

    def __init__(self, timeout: int = 30, sandbox_dir: str = "workspace"):
        self._timeout = timeout
        self._sandbox_dir = Path(sandbox_dir)
        self._sandbox_dir.mkdir(parents=True, exist_ok=True)

    def execute(self, script_path: str, args: Optional[List[str]] = None) -> str:
        path = Path(script_path)
        if not path.exists():
            return f"[ERROR] Script not found: {script_path}"
        if path.suffix != ".py":
            return f"[ERROR] Only .py scripts are supported. Got: {path.suffix}"
        try:
            cmd = ["python", str(path.resolve())] + (args or [])
            result = subprocess.run(
                cmd,
                cwd=str(self._sandbox_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self._timeout,
            )
            output_parts = []
            if result.stdout:
                output_parts.append(f"[stdout]\n{result.stdout.strip()}")
            if result.stderr:
                output_parts.append(f"[stderr]\n{result.stderr.strip()}")
            status = "SUCCESS" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
            return f"[{status}]\n" + "\n".join(output_parts) if output_parts else f"[{status}] No output."
        except subprocess.TimeoutExpired:
            return f"[ERROR] Script timed out after {self._timeout}s: {script_path}"
        except Exception as e:
            return f"[ERROR] Script execution failed: {e}"


# ── Vision Tools ──────────────────────────────────────────────────────────────

class AnalyzeImageTool(YunoTool):
    """Analyze an image file for metadata, structure, and OCR text."""
    name = "analyze_image"
    description = "Analyze a local image file (.png, .jpg, .webp) to extract metadata and OCR text."
    permission = ToolPermissionLevel.AUTO

    def __init__(self, vision=None):
        self.vision = vision

    def execute(self, image_path: str) -> str:
        from .vision import YunoVision
        v = self.vision or YunoVision()
        try:
            result = v.analyze_image(image_path)
            return result.summary()
        except Exception as e:
            return f"[ERROR] Image analysis failed: {e}"


class AnalyzeScreenshotTool(YunoTool):
    """Capture and analyze the active desktop display."""
    name = "analyze_screenshot"
    description = "Capture a real-time desktop screenshot and extract screen text via OCR."
    permission = ToolPermissionLevel.AUTO

    def __init__(self, vision=None):
        self.vision = vision

    def execute(self) -> str:
        from .vision import YunoVision
        v = self.vision or YunoVision()
        try:
            result = v.analyze_screenshot(save=True)
            return result.summary()
        except Exception as e:
            return f"[ERROR] Screenshot analysis failed: {e}"


class ExtractDocumentOCRTool(YunoTool):
    """Extract text from PDF or image documents via OCR."""
    name = "extract_document_ocr"
    description = "Extract text from PDF documents or multi-page image files via OCR."
    permission = ToolPermissionLevel.AUTO

    def __init__(self, vision=None):
        self.vision = vision

    def execute(self, document_path: str, max_pages: int = 10) -> str:
        from .vision import YunoVision
        v = self.vision or YunoVision()
        try:
            return v.extract_document_ocr(document_path, max_pages=max_pages)
        except Exception as e:
            return f"[ERROR] Document OCR failed: {e}"


# ── Voice Tools ───────────────────────────────────────────────────────────────

class SpeakTextTool(YunoTool):
    """Speak text aloud using Text-To-Speech (TTS)."""
    name = "speak_text"
    description = "Synthesize and speak text aloud using local Text-To-Speech."
    permission = ToolPermissionLevel.AUTO

    def __init__(self, voice=None):
        self.voice = voice

    def execute(self, text: str) -> str:
        from .voice import YunoVoice
        v = self.voice or YunoVoice()
        try:
            ok = v.speak(text)
            return f"[SUCCESS] Spoke: '{text[:80]}...'" if ok else "[ERROR] Speech synthesis failed."
        except Exception as e:
            return f"[ERROR] Speak text failed: {e}"


class ListenSpeechTool(YunoTool):
    """Listen to microphone speech or transcribe an audio file."""
    name = "listen_speech"
    description = "Listen to microphone audio or transcribe an audio file (.wav, .flac) to text."
    permission = ToolPermissionLevel.AUTO

    def __init__(self, voice=None):
        self.voice = voice

    def execute(self, audio_path: Optional[str] = None) -> str:
        from .voice import YunoVoice
        v = self.voice or YunoVoice()
        try:
            if audio_path:
                return v.transcribe_audio_file(audio_path)
            return v.listen()
        except Exception as e:
            return f"[ERROR] Listen speech failed: {e}"


# ── Automation Tools ──────────────────────────────────────────────────────────

class OrganizeFilesTool(YunoTool):
    """Sort messy directory files into categorized subfolders."""
    name = "organize_files"
    description = "Organize loose files in a directory into extension subfolders (Images, Docs, Code, etc)."
    permission = ToolPermissionLevel.CONFIRM

    def __init__(self, automation=None):
        self.automation = automation

    def execute(self, target_dir: Optional[str] = None) -> str:
        from .automation import YunoAutomation
        a = self.automation or YunoAutomation()
        try:
            return a.organize_directory(target_dir)
        except Exception as e:
            return f"[ERROR] Organize files failed: {e}"


class ScheduleReminderTool(YunoTool):
    """Create and log a reminder task entry."""
    name = "schedule_reminder"
    description = "Log and schedule a background task reminder entry."
    permission = ToolPermissionLevel.AUTO

    def __init__(self, automation=None):
        self.automation = automation

    def execute(self, task: str, time_str: str = "now") -> str:
        from .automation import YunoAutomation
        a = self.automation or YunoAutomation()
        try:
            return a.create_reminder(task, time_str)
        except Exception as e:
            return f"[ERROR] Schedule reminder failed: {e}"


class InitProjectTool(YunoTool):
    """Scaffold a new project boilerplate (python, web, cpp)."""
    name = "init_project"
    description = "Scaffold a new project directory with standard boilerplate files and structure."
    permission = ToolPermissionLevel.EXPLICIT

    def __init__(self, automation=None):
        self.automation = automation

    def execute(self, project_name: str, project_type: str = "python", target_parent: str = "workspace") -> str:
        from .automation import YunoAutomation
        a = self.automation or YunoAutomation()
        try:
            return a.scaffold_project(project_name, project_type, target_parent)
        except Exception as e:
            return f"[ERROR] Init project failed: {e}"


# ── Tool Registry ─────────────────────────────────────────────────────────────

class YunoToolRegistry:
    """
    Manages all available YUNO tools and enforces Human-In-The-Loop permission
    checks before executing any state-modifying actions.

    Tools are organized by ToolPermissionLevel:
    - AUTO      → Run immediately, log quietly
    - CONFIRM   → Show plan, ask y/n
    - EXPLICIT  → Show plan, require typed "yes"

    Every tool execution is appended to the audit log.
    """

    def __init__(self, config=None, console_input_fn: Callable[[str], str] = input):
        self.config = config
        self.console_input_fn = console_input_fn
        self._tools: Dict[str, YunoTool] = {}

        # Config-driven settings
        self._require_permission: bool = True
        self._auto_approve_reads: bool = True
        self._script_timeout: int = 30
        self._sandbox_dir: str = "workspace"
        self._audit_log: Optional[Path] = None

        if config and hasattr(config, "tools"):
            t = config.tools
            self._require_permission = getattr(t, "require_permission", True)
            self._auto_approve_reads = getattr(t, "auto_approve_reads", True)
            self._script_timeout = getattr(t, "script_timeout_seconds", 30)
            self._sandbox_dir = getattr(t, "sandbox_dir", "workspace")
            log_file = getattr(t, "audit_log_file", "logs/tool_audit.log")
            self._audit_log = Path(log_file)
            self._audit_log.parent.mkdir(parents=True, exist_ok=True)

        # Vision, Voice, and Automation engines reference
        from .vision import YunoVision
        from .voice import YunoVoice
        from .automation import YunoAutomation
        self.vision = YunoVision(config)
        self.voice = YunoVoice(config)
        self.automation = YunoAutomation(config)

        # Register all built-in tools
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register all built-in YUNO tools."""
        self.register(ReadFileTool())
        self.register(ListDirTool())
        self.register(SearchFilesTool())
        self.register(SummarizeFileTool())
        self.register(WriteFileTool())
        self.register(CreateNoteTool())
        self.register(RunScriptTool(
            timeout=self._script_timeout,
            sandbox_dir=self._sandbox_dir,
        ))
        self.register(AnalyzeImageTool(vision=self.vision))
        self.register(AnalyzeScreenshotTool(vision=self.vision))
        self.register(ExtractDocumentOCRTool(vision=self.vision))
        self.register(SpeakTextTool(voice=self.voice))
        self.register(ListenSpeechTool(voice=self.voice))
        self.register(OrganizeFilesTool(automation=self.automation))
        self.register(ScheduleReminderTool(automation=self.automation))
        self.register(InitProjectTool(automation=self.automation))

    def register(self, tool: YunoTool) -> None:
        """Register a tool in the registry."""
        self._tools[tool.name] = tool
        logger.info(f"[Tools] Registered: {tool.name} [{tool.permission.value}]")

    def list_tools(self) -> List[Tuple[str, str, str]]:
        """Return list of (name, permission, description) for all tools."""
        return [
            (t.name, t.permission.value, t.description)
            for t in self._tools.values()
        ]

    def run_tool(self, tool_name: str, **kwargs) -> str:
        """
        Execute a registered tool with HITL enforcement.

        Workflow:
          1. Check tool exists
          2. Apply permission level (AUTO / CONFIRM / EXPLICIT)
          3. Execute
          4. Audit log the result

        Args:
            tool_name: Name of the registered tool to execute.
            **kwargs:  Arguments passed directly to tool.execute().

        Returns:
            Tool output string (success or error message).
        """
        if tool_name not in self._tools:
            return f"[ERROR] Tool not found: '{tool_name}'. Available: {list(self._tools)}"

        tool = self._tools[tool_name]

        # ── Permission Gate ────────────────────────────────────────────────
        if not self._check_permission(tool, kwargs):
            self._audit(tool_name, kwargs, result="REJECTED_BY_USER")
            return "[BLOCKED] Action rejected by user permission policy."

        # ── Execute ───────────────────────────────────────────────────────
        try:
            logger.info(f"[Tools] Executing {tool_name} | args={kwargs}")
            result = str(tool.execute(**kwargs))
            self._audit(tool_name, kwargs, result="SUCCESS")
            return result
        except TypeError as e:
            msg = f"[ERROR] Invalid arguments for tool '{tool_name}': {e}"
            self._audit(tool_name, kwargs, result="ARG_ERROR")
            return msg
        except Exception as e:
            msg = f"[ERROR] Tool '{tool_name}' execution error: {e}"
            logger.error(msg)
            self._audit(tool_name, kwargs, result="ERROR")
            return msg

    def _check_permission(self, tool: YunoTool, kwargs: Dict) -> bool:
        """
        Enforce the correct permission check for the given tool.
        Returns True if execution should proceed.
        """
        lvl = tool.permission

        # AUTO: always allowed (read-only operations)
        if lvl == ToolPermissionLevel.AUTO:
            if self._auto_approve_reads:
                return True

        # CONFIRM: show plan, simple y/n
        if lvl in (ToolPermissionLevel.CONFIRM, ToolPermissionLevel.EXPLICIT):
            if not self._require_permission:
                return True

            print()
            print(f"  ┌─ YUNO Permission Request {'─' * 35}")
            print(f"  │  Tool       : {tool.name}")
            print(f"  │  Permission : {tool.permission.value}")
            print(f"  │  Description: {tool.description}")
            if kwargs:
                print(f"  │  Arguments  :")
                for k, v in kwargs.items():
                    val_preview = str(v)[:80] + ("..." if len(str(v)) > 80 else "")
                    print(f"  │    {k}: {val_preview}")
            print(f"  └{'─' * 54}")

            if lvl == ToolPermissionLevel.EXPLICIT:
                prompt = "  Approve this action? Type 'yes' to confirm: "
                response = self.console_input_fn(prompt).strip().lower()
                allowed = response == "yes"
            else:
                prompt = "  Proceed? (y/n): "
                response = self.console_input_fn(prompt).strip().lower()
                allowed = response in ("y", "yes")

            if not allowed:
                print("  [BLOCKED] Action rejected by user.")
            return allowed

        # AUTO with reads disabled — still ask
        print(f"\n  [YUNO] Tool '{tool.name}' requires approval.")
        response = self.console_input_fn("  Approve? (y/n): ").strip().lower()
        return response in ("y", "yes")

    def _audit(
        self,
        tool_name: str,
        kwargs: Dict,
        result: str,
    ) -> None:
        """Append a structured entry to the tool audit log."""
        if not self._audit_log:
            return
        try:
            entry = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "tool": tool_name,
                "args": {k: str(v)[:200] for k, v in kwargs.items()},
                "result": result,
            }
            with open(self._audit_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"[Tools] Audit log write failed: {e}")
