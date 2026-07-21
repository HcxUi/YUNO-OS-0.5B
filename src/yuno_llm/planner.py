"""
YUNO Planner & Intent Classifier
==================================
Classifies user intent, extracts structured plans from LLM thinking blocks,
and decides whether to route to a tool, a memory query, or a regular chat turn.

Intent Types
------------
CHAT          → Normal conversational reply (no tool needed)
TOOL_CALL     → Execute a specific registered tool
MEMORY_QUERY  → Search episodic/long-term memory for recall
PLAN          → Multi-step agentic task requiring step-by-step execution

Usage
-----
    planner = YunoPlanner(config)
    intent, tool_name, tool_args = planner.parse_intent(user_input)
    if intent == IntentType.TOOL_CALL:
        result = tool_registry.run_tool(tool_name, **tool_args)
"""

import re
import json
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional, List

logger = logging.getLogger("yuno_llm.planner")


# ── Intent Types ──────────────────────────────────────────────────────────────

class IntentType(str, Enum):
    CHAT         = "CHAT"          # Regular conversational reply
    TOOL_CALL    = "TOOL_CALL"     # Execute a tool from the registry
    MEMORY_QUERY = "MEMORY_QUERY"  # Retrieve from episodic/long-term memory
    PLAN         = "PLAN"          # Multi-step agentic task


# ── Plan Step ─────────────────────────────────────────────────────────────────

@dataclass
class PlanStep:
    """A single step in a multi-step YUNO plan."""
    index: int
    description: str
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = field(default_factory=dict)
    done: bool = False


@dataclass
class YunoPlan:
    """A sequence of steps to complete an agentic task."""
    goal: str
    steps: List[PlanStep] = field(default_factory=list)

    def pending_steps(self) -> List[PlanStep]:
        return [s for s in self.steps if not s.done]

    def mark_done(self, index: int) -> None:
        for s in self.steps:
            if s.index == index:
                s.done = True
                return


# ── Keyword Pattern Tables ────────────────────────────────────────────────────

# Each entry: (regex_pattern, tool_name, arg_extractor_fn)
_FILE_PATH_RE = r"([a-zA-Z]:[\\\/][^\s]+|[^\s]+\.[a-z]{1,5})"

_READ_PATTERNS = [
    r"^(?:read|show|cat|open|print|display|dekho|dikhao)\s+" + _FILE_PATH_RE,
    r"(?:read|open|show)\s+(?:the\s+)?(?:file|content[s]?)\s+(?:of\s+|at\s+|from\s+)?" + _FILE_PATH_RE,
    r"(?:file\s+)?" + _FILE_PATH_RE + r"\s+(?:padho|padhna|parhna|read kar)",
]

_WRITE_PATTERNS = [
    r"^(?:write|create|save|banao|likhna|likhdo)\s+" + _FILE_PATH_RE
    + r"\s+with\s+content\s+['\"](.+)['\"]",
    r"^(?:write|create)\s+(?:a\s+)?(?:file\s+)?" + _FILE_PATH_RE
    + r"\s+(?:containing|with)\s+['\"](.+)['\"]",
]

_LIST_DIR_PATTERNS = [
    r"^(?:list|ls|dir|show\s+files?)\s+(?:in\s+)?" + _FILE_PATH_RE,
    r"^(?:files?\s+in|folder\s+contents?\s+of)\s+" + _FILE_PATH_RE,
    r"^(?:kya\s+hai|kya\s+files?)\s+(?:in\s+)?" + _FILE_PATH_RE,
]

_SEARCH_PATTERNS = [
    r"^(?:search|grep|find|dhundo|khojo)\s+['\"](.+)['\"]\s+in\s+" + _FILE_PATH_RE,
    r"^(?:search|find)\s+(?:for\s+)?['\"](.+)['\"]\s+(?:inside|in)\s+" + _FILE_PATH_RE,
]

_MEMORY_QUERY_KEYWORDS = [
    "remember", "recall", "yaad", "bata", "did i tell", "mujhe batao",
    "what did i say", "do you know my", "what is my", "mera", "meri",
    "what have we discussed", "last time", "pehle", "previously", "history",
]

_PLAN_KEYWORDS = [
    "plan", "help me", "step by step", "steps", "how to", "kaise karo",
    "organize", "automate", "workflow", "task", "create a report",
    "setup", "configure", "install", "karo", "kar do",
]

_IMAGE_PATTERNS = [
    r"^(?:analyze|ocr|inspect|image|photo)\s+(?:image\s+)?([a-zA-Z]:[\\\/][^\s]+\.(?:png|jpg|jpeg|webp|bmp)|[^\s]+\.(?:png|jpg|jpeg|webp|bmp))",
    r"^(?:read|scan)\s+(?:text\s+from\s+)?([a-zA-Z]:[\\\/][^\s]+\.(?:png|jpg|jpeg|webp|bmp)|[^\s]+\.(?:png|jpg|jpeg|webp|bmp))",
]

_SCREENSHOT_PATTERNS = [
    r"^(?:take\s+)?(?:screenshot|screen|snapshot)\s*(?:capture|analyze|dekho)?",
    r"^(?:screen\s+par\s+kya\s+hai|screen\s+dekho|desktop\s+capture)",
]

_PDF_OCR_PATTERNS = [
    r"^(?:ocr|extract\s+pdf|pdf\s+ocr|scan\s+pdf)\s+([a-zA-Z]:[\\\/][^\s]+\.pdf|[^\s]+\.pdf)",
]

_SPEAK_PATTERNS = [
    r"^(?:speak|say|bolo|sunao|read\s+aloud)\s+['\"]?(.+)['\"]?",
    r"^(?:speak\s+text|say\s+out\s+loud)\s+['\"]?(.+)['\"]?",
]

_LISTEN_PATTERNS = [
    r"^(?:listen|listen\s+speech|transcribe\s+audio|audio\s+transcribe)\s*(?:([a-zA-Z]:[\\\/][^\s]+\.(?:wav|flac|mp3)|[^\s]+\.(?:wav|flac|mp3)))?",
]

_ORGANIZE_PATTERNS = [
    r"^(?:organize|sort|clean|arrange)\s+(?:files?\s+in\s+)?([a-zA-Z]:[\\\/][^\s]+|[^\s]+)?",
    r"^(?:organize\s+workspace|organize\s+directory|files?\s+organize\s+karo)",
]

_REMINDER_PATTERNS = [
    r"^(?:remind|reminder|set\s+reminder|yaad\s+dilaana)\s+['\"]?(.+?)['\"]?\s+(?:at|in|on)\s+(.+)",
    r"^(?:remind\s+me\s+to|set\s+a\s+reminder\s+to)\s+['\"]?(.+)['\"]?",
]

_INIT_PROJECT_PATTERNS = [
    r"^(?:init|create|scaffold|banao)\s+(python|web|cpp|html)\s+project\s+([a-zA-Z0-9_\-]+)",
    r"^(?:init\s+project|create\s+project)\s+([a-zA-Z0-9_\-]+)\s*(python|web|cpp)?",
]

_SCRIPT_PATTERNS = [
    r"^(?:run|execute|chalao|chala)\s+" + _FILE_PATH_RE,
    r"^(?:run|execute)\s+(?:the\s+)?(?:script|code)\s+" + _FILE_PATH_RE,
]


# ── Planner Class ─────────────────────────────────────────────────────────────

class YunoPlanner:
    """
    Classifies user intent and extracts tool calls or plans.

    Routing logic (in priority order):
      1. Keyword/pattern matching (fast, deterministic)
      2. Memory query detection (keyword heuristics)
      3. Plan detection (multi-step task keywords)
      4. Default → CHAT

    For ambiguous inputs, the LLM's <think> blocks are parsed to extract
    any tool calls or plan steps that the model reasoned about.
    """

    def __init__(self, config=None):
        self.config = config

    # ── Primary Intent Classifier ─────────────────────────────────────────────

    def parse_intent(
        self, user_input: str
    ) -> Tuple[IntentType, Optional[str], Dict[str, Any]]:
        """
        Classify the user input and extract tool name + args if applicable.

        Returns:
            (IntentType, tool_name_or_None, tool_args_dict)

        Examples:
            "read F:\\llm\\config.yaml"
                → (IntentType.TOOL_CALL, "read_file", {"file_path": "..."})
            "Mujhe yaad dilao main kya bol raha tha"
                → (IntentType.MEMORY_QUERY, None, {"query": "..."})
            "Chat karte hain"
                → (IntentType.CHAT, None, {})
        """
        stripped = user_input.strip()

        # 1. Image analysis
        img_path = self._match_image(stripped)
        if img_path:
            return IntentType.TOOL_CALL, "analyze_image", {"image_path": img_path}

        # 2. Screenshot analysis
        if self._match_screenshot(stripped):
            return IntentType.TOOL_CALL, "analyze_screenshot", {}

        # 3. PDF OCR document analysis
        pdf_path = self._match_pdf_ocr(stripped)
        if pdf_path:
            return IntentType.TOOL_CALL, "extract_document_ocr", {"document_path": pdf_path}

        # 4. Speak text
        speak_text = self._match_speak(stripped)
        if speak_text:
            return IntentType.TOOL_CALL, "speak_text", {"text": speak_text}

        # 5. Listen speech / transcribe audio
        audio_path, is_listen = self._match_listen(stripped)
        if is_listen:
            args = {"audio_path": audio_path} if audio_path else {}
            return IntentType.TOOL_CALL, "listen_speech", args

        # 6. Organize files
        target_dir = self._match_organize(stripped)
        if target_dir is not None:
            args = {"target_dir": target_dir} if target_dir else {}
            return IntentType.TOOL_CALL, "organize_files", args

        # 7. Schedule reminder
        task_str, time_val = self._match_reminder(stripped)
        if task_str:
            return IntentType.TOOL_CALL, "schedule_reminder", {"task": task_str, "time_str": time_val}

        # 8. Init project
        p_name, p_type = self._match_init_project(stripped)
        if p_name:
            return IntentType.TOOL_CALL, "init_project", {"project_name": p_name, "project_type": p_type}

        # 9. Read file
        path = self._match_read(stripped)
        if path:
            return IntentType.TOOL_CALL, "read_file", {"file_path": path}

        # 5. Write file
        path, content = self._match_write(stripped)
        if path:
            return IntentType.TOOL_CALL, "write_file", {"file_path": path, "content": content}

        # 6. List directory
        path = self._match_list_dir(stripped)
        if path:
            return IntentType.TOOL_CALL, "list_dir", {"dir_path": path}

        # 7. Search in files
        query, path = self._match_search(stripped)
        if query and path:
            return IntentType.TOOL_CALL, "search_files", {"query": query, "dir_path": path}

        # 8. Run script
        path = self._match_script(stripped)
        if path:
            return IntentType.TOOL_CALL, "run_script", {"script_path": path}

        # 6. Memory query
        if self._is_memory_query(stripped):
            return IntentType.MEMORY_QUERY, None, {"query": stripped}

        # 7. Multi-step plan request
        if self._is_plan_request(stripped):
            return IntentType.PLAN, None, {"goal": stripped}

        # 8. Default — normal chat
        return IntentType.CHAT, None, {}

    # ── Thought Block Parser ──────────────────────────────────────────────────

    def extract_thought_steps(self, raw_llm_response: str) -> Tuple[str, str]:
        """
        Splits an LLM response into internal <think>...</think> reasoning
        and the final visible response.

        Args:
            raw_llm_response: Raw model output, possibly containing <think> tags.

        Returns:
            (thought_block, clean_response)
        """
        think_match = re.search(r"<think>(.*?)</think>", raw_llm_response, re.DOTALL)
        if think_match:
            thought = think_match.group(1).strip()
            response = re.sub(
                r"<think>.*?</think>", "", raw_llm_response, flags=re.DOTALL
            ).strip()
            return thought, response
        return "", raw_llm_response.strip()

    def extract_tool_call_from_thought(
        self, thought: str
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Scans a <think> block for JSON-formatted tool call instructions
        that the model may have reasoned about.

        Expected format inside <think>:
            TOOL: read_file
            ARGS: {"file_path": "config.yaml"}

        Returns:
            (tool_name, tool_args) or (None, {})
        """
        tool_match = re.search(r"TOOL:\s*(\w+)", thought, re.IGNORECASE)
        args_match = re.search(r"ARGS:\s*(\{.*?\})", thought, re.IGNORECASE | re.DOTALL)

        if tool_match:
            tool_name = tool_match.group(1).strip()
            tool_args = {}
            if args_match:
                try:
                    tool_args = json.loads(args_match.group(1))
                except json.JSONDecodeError:
                    logger.warning("[Planner] Could not parse tool args JSON from thought block.")
            return tool_name, tool_args

        return None, {}

    def build_plan_from_goal(self, goal: str, llm_response: str) -> YunoPlan:
        """
        Constructs a YunoPlan by parsing numbered steps from the LLM response.

        Supports formats like:
            1. Read the config file
            2. Summarize its contents
            3. Write a report

        Args:
            goal: The user's original goal string.
            llm_response: LLM output describing the plan.

        Returns:
            YunoPlan with extracted steps.
        """
        plan = YunoPlan(goal=goal)
        # Match lines starting with: "1.", "Step 1:", "1)", etc.
        step_pattern = re.compile(
            r"^\s*(?:step\s*)?(\d+)[.):\-]\s+(.+)", re.IGNORECASE | re.MULTILINE
        )
        for match in step_pattern.finditer(llm_response):
            idx = int(match.group(1))
            desc = match.group(2).strip()
            step = PlanStep(index=idx, description=desc)

            # Try to detect embedded tool intent in the step description
            intent, tool_name, tool_args = self.parse_intent(desc)
            if intent == IntentType.TOOL_CALL and tool_name:
                step.tool_name = tool_name
                step.tool_args = tool_args

            plan.steps.append(step)

        if not plan.steps:
            logger.debug("[Planner] No numbered steps found in plan response.")

        return plan

    # ── Private Pattern Matchers ──────────────────────────────────────────────

    def _match_read(self, text: str) -> Optional[str]:
        for pattern in _READ_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return m.group(1).strip().rstrip(".,;'\"")
        return None

    def _match_write(self, text: str) -> Tuple[Optional[str], str]:
        for pattern in _WRITE_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return m.group(1).strip(), m.group(2).strip()
        return None, ""

    def _match_list_dir(self, text: str) -> Optional[str]:
        for pattern in _LIST_DIR_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return m.group(1).strip().rstrip(".,;'\"")
        return None

    def _match_search(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        for pattern in _SEARCH_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return m.group(1).strip(), m.group(2).strip()
        return None, None

    def _match_script(self, text: str) -> Optional[str]:
        for pattern in _SCRIPT_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return m.group(1).strip().rstrip(".,;'\"")
        return None

    def _match_image(self, text: str) -> Optional[str]:
        for pattern in _IMAGE_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return m.group(1).strip().rstrip(".,;'\"")
        return None

    def _match_screenshot(self, text: str) -> bool:
        for pattern in _SCREENSHOT_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _match_pdf_ocr(self, text: str) -> Optional[str]:
        for pattern in _PDF_OCR_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return m.group(1).strip().rstrip(".,;'\"")
        return None

    def _match_speak(self, text: str) -> Optional[str]:
        for pattern in _SPEAK_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return m.group(1).strip().rstrip(".,;'\"")
        return None

    def _match_listen(self, text: str) -> Tuple[Optional[str], bool]:
        for pattern in _LISTEN_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                audio_path = m.group(1).strip().rstrip(".,;'\"") if m.group(1) else None
                return audio_path, True
        return None, False

    def _match_organize(self, text: str) -> Optional[str]:
        for pattern in _ORGANIZE_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return m.group(1).strip().rstrip(".,;'\"") if m.lastindex and m.group(1) else ""
        return None

    def _match_reminder(self, text: str) -> Tuple[Optional[str], str]:
        for pattern in _REMINDER_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                task = m.group(1).strip()
                time_val = m.group(2).strip() if m.lastindex >= 2 and m.group(2) else "now"
                return task, time_val
        return None, "now"

    def _match_init_project(self, text: str) -> Tuple[Optional[str], str]:
        for pattern in _INIT_PROJECT_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                g1 = m.group(1).strip()
                g2 = m.group(2).strip() if m.lastindex >= 2 and m.group(2) else "python"
                if g1 in ("python", "web", "cpp", "html"):
                    return g2, g1
                return g1, g2
        return None, "python"

    def _is_memory_query(self, text: str) -> bool:
        lower = text.lower()
        return any(kw in lower for kw in _MEMORY_QUERY_KEYWORDS)

    def _is_plan_request(self, text: str) -> bool:
        lower = text.lower()
        # Must contain a plan keyword AND be asking for action (not just info)
        has_plan_keyword = any(kw in lower for kw in _PLAN_KEYWORDS)
        is_question_or_command = (
            lower.endswith("?")
            or any(lower.startswith(w) for w in ["help", "how", "kaise", "plan", "create", "make"])
        )
        return has_plan_keyword and is_question_or_command
