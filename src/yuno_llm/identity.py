"""
YUNO-LLM Identity Module
=========================
Manages YUNO's system persona and dynamically builds the system prompt
by injecting current context: date/time, user name, active project, language
mode, and a summary of relevant memory.

The system prompt is the single most important control surface for YUNO's
behavior — everything from language (Hinglish/Hindi/English) to HITL safety
rules is encoded here.

Usage:
    identity = YunoIdentity(config)
    messages = identity.apply_system_prompt(
        conversation_history,
        memory=memory_instance,
        last_user_message="Main kya bol raha tha?"
    )
"""

from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from .memory import YunoMemory
    from .config import YunoConfig


# ── Default System Prompt Template ────────────────────────────────────────────

_DEFAULT_SYSTEM_PROMPT = """\
You are YUNO, a personal AI operating system built on the YUNO-LLM architecture.

Identity & Mission:
- Name: YUNO | Version: {version}
- Role: Local Personal AI OS — private, offline-first, always on your side.
- Architecture: Decoder-only Transformer with memory, planning, and safe tool modules.

Language Instructions:
- Default: Hinglish (natural mix of Hindi + English).
- Pure Hindi input → respond in Hindi. Pure English input → respond in English.
- Keep technical terms (API, token, embedding, file path, code) in English.
- Example Hinglish: "Main aapki kaise madad kar sakta hoon?"

Reasoning & Honesty:
- Think step-by-step inside <think>...</think> before answering.
- Self-check before presenting your answer.
- Never hallucinate. If uncertain, say: "Mujhe pakka nahi pata, lekin..."

Tool Safety (Human-in-the-Loop):
- Read-only tools (read file, list directory, search) run automatically.
- ANY write, delete, execute, or external API action REQUIRES explicit user approval.
- Always describe what you are about to do. Ask: "Kya main yeh kar sakta hoon?"

Memory:
- Use your Short-Term, Long-Term, and Episodic memories to personalize answers.
- If you recall a relevant fact, use it naturally — don't just list facts.
"""

_CONTEXT_SEPARATOR = "\n" + "─" * 60 + "\n"


# ── YunoIdentity Class ────────────────────────────────────────────────────────

class YunoIdentity:
    """
    Manages YUNO's system identity and dynamic prompt construction.

    Responsibilities:
    - Building the base system prompt from config
    - Injecting dynamic context: timestamp, user name, active project
    - Injecting relevant memory summaries
    - Prepending the system prompt to message lists

    Example:
        identity = YunoIdentity(config)
        messages = identity.apply_system_prompt(
            history, memory=memory, last_user_message=user_input
        )
    """

    def __init__(self, config: Optional["YunoConfig"] = None):
        if config and hasattr(config, "identity"):
            self.name = config.identity.name
            self.version = config.identity.version
            self._base_prompt = config.identity.system_prompt
        else:
            self.name = "YUNO"
            self.version = "0.2"
            self._base_prompt = _DEFAULT_SYSTEM_PROMPT.format(version=self.version)

        # Language mode from config
        self._language_mode = "hinglish"
        if config and hasattr(config, "language"):
            self._language_mode = getattr(config.language, "default_mode", "hinglish")

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def system_prompt(self) -> str:
        """Return the base system prompt (no dynamic context)."""
        return self._base_prompt

    def apply_system_prompt(
        self,
        messages: List[Dict[str, str]],
        override_system: Optional[str] = None,
        memory: Optional["YunoMemory"] = None,
        last_user_message: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Prepend a system message to a message list, with optional dynamic context.

        If messages already start with a system role message, it is replaced.

        Args:
            messages:          Chat history (list of {role, content} dicts).
            override_system:   If provided, replaces the base prompt entirely.
            memory:            YunoMemory instance for context injection.
            last_user_message: Used to search episodic memory for relevance.

        Returns:
            Messages with system prompt prepended/replaced.
        """
        if override_system:
            system_content = override_system
        else:
            system_content = self.build_dynamic_system_prompt(
                memory=memory,
                search_query=last_user_message,
            )

        # Replace or prepend system message
        if messages and messages[0].get("role") == "system":
            return [{"role": "system", "content": system_content}] + messages[1:]
        return [{"role": "system", "content": system_content}] + messages

    def build_dynamic_system_prompt(
        self,
        memory: Optional["YunoMemory"] = None,
        search_query: Optional[str] = None,
    ) -> str:
        """
        Constructs the full system prompt with dynamic context injected.

        Context injected:
        - Current date & time (Asia/Kolkata — IST)
        - User's name (from long-term memory)
        - Active project name (from project memory)
        - Language mode reminder
        - Relevant episodic memories (FTS5 search)

        Args:
            memory:       YunoMemory instance (optional).
            search_query: If provided, episodic memory is searched for relevance.

        Returns:
            Complete system prompt string.
        """
        parts: List[str] = [self._base_prompt.rstrip()]

        # ── 1. Date / Time context ────────────────────────────────────────
        now = datetime.now()
        date_str = now.strftime("%A, %d %B %Y — %I:%M %p")
        parts.append(f"\n[System Time]\nCurrent date and time: {date_str} (IST)")

        # ── 2. Language mode ──────────────────────────────────────────────
        mode_labels = {
            "hinglish": "Hinglish (Hindi + English mix)",
            "hindi": "Hindi",
            "english": "English",
        }
        mode_label = mode_labels.get(self._language_mode, self._language_mode)
        parts.append(f"[Language Mode]\nDefault response language: {mode_label}")

        # ── 3. Memory context ─────────────────────────────────────────────
        if memory:
            mem_context = memory.compile_memory_context(search_query=search_query)
            if mem_context:
                parts.append(f"[Active Memory Context]\n{mem_context}")

        return _CONTEXT_SEPARATOR.join(parts)

    # ── Greeting ──────────────────────────────────────────────────────────────

    def greeting(self, memory: Optional["YunoMemory"] = None) -> str:
        """
        Return a personalized greeting for YUNO.

        Pulls the user's name from long-term memory if available.
        """
        user_name = ""
        if memory:
            user_name = memory.get_personal_fact("user_name", "")
            if not user_name:
                user_name = memory.get_personal_fact("name", "")

        now = datetime.now()
        hour = now.hour
        if 5 <= hour < 12:
            time_of_day = "Good morning"
            hindi_greeting = "Suprabhat"
        elif 12 <= hour < 17:
            time_of_day = "Good afternoon"
            hindi_greeting = "Namaskar"
        elif 17 <= hour < 21:
            time_of_day = "Good evening"
            hindi_greeting = "Shubh Sandhya"
        else:
            time_of_day = "Hello"
            hindi_greeting = "Namaste"

        name_part = f", {user_name}" if user_name else ""
        return (
            f"{hindi_greeting}{name_part}! 🙏 {time_of_day}!\n"
            f"Main hoon {self.name} v{self.version} — aapka personal AI assistant.\n"
            f"Aaj main aapki kaise madad kar sakta hoon?"
        )

    # ── Language Adaptation Hint ──────────────────────────────────────────────

    def detect_language_hint(self, user_input: str) -> str:
        """
        Returns a language instruction string based on the detected input language.
        Used to inject per-turn language hints into the context.

        Returns:
            One of: "hinglish", "hindi", "english"
        """
        # Simple heuristic: check for Devanagari script
        has_devanagari = any("\u0900" <= ch <= "\u097F" for ch in user_input)
        if has_devanagari:
            return "hindi"

        # Check for common Hindi romanized words
        hindi_markers = [
            "hai", "hain", "karo", "kya", "aur", "nahi", "mujhe",
            "agar", "lekin", "yeh", "woh", "tum", "aap", "main",
        ]
        lower = user_input.lower().split()
        hindi_word_count = sum(1 for w in lower if w in hindi_markers)
        if hindi_word_count >= 2:
            return "hinglish"

        return "english"

    def __repr__(self) -> str:
        return f"YunoIdentity(name={self.name!r}, version={self.version!r})"
