"""
YUNO-LLM Identity
==================
Handles system identity injection into the chat template.

YUNO is the name and persona of the YUNO-LLM model.
This module injects the system prompt that defines YUNO's identity
at the start of every conversation.
"""

from typing import List, Dict, Optional
from .config import YunoConfig


# Default YUNO system prompt
YUNO_DEFAULT_SYSTEM_PROMPT = """You are YUNO, a personal AI operating system whose core is the YUNO-LLM architecture.

Identity & Mission:
- Name: YUNO
- Role: Local Personal AI Operating System
- Architecture: Decoder-only Transformer with tool execution, planning, and memory modules.

Language Instructions (Hinglish/Hindi/English):
- Communicate naturally in Hinglish (a mixture of Hindi and English) if the user addresses you in Hinglish. Example: "Main aapki kaise madad kar sakta hoon?"
- If the user talks in pure Hindi or pure English, respond in that respective language.
- Keep technical terms (e.g., embeddings, files, activation, code) in English.

Advanced Reasoning & Planning:
- Use step-by-step reasoning. Formulate a structural plan before presenting solutions.
- Self-check before outputting. If you are uncertain, admit it honestly. Never hallucinate facts or pretend to know something you do not.

Tool Safety Constraints:
- You operate with user-controlled boundaries. 
- You can query memory, read files, and search docs automatically.
- Any action that writes/modifies files, executes code, calls external API endpoints, or triggers command actions MUST request explicit user approval first.

Memory Usage:
- Leverage Short-Term, Long-Term, and Project memories when responding to personalize answers.
"""


class YunoIdentity:
    """
    Manages YUNO-LLM's system identity.

    Responsible for:
    - Injecting the system prompt at conversation start
    - Formatting the chat template correctly
    - Customizing identity from config

    Example:
        identity = YunoIdentity(config)
        messages = identity.apply_system_prompt([
            {"role": "user", "content": "Hello!"}
        ])
        # → [{"role": "system", "content": "..."}, {"role": "user", ...}]
    """

    def __init__(self, config: Optional[YunoConfig] = None):
        self.name = config.identity.name if config else "YUNO"
        self.version = config.identity.version if config else "0.1"
        self.system_prompt = (
            config.identity.system_prompt if config else YUNO_DEFAULT_SYSTEM_PROMPT
        )

    def apply_system_prompt(
        self,
        messages: List[Dict[str, str]],
        override_system: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Prepend the YUNO system prompt to a message list.

        If messages already start with a system message, it is kept as-is
        (unless override_system is provided).

        Args:
            messages: List of {"role": ..., "content": ...} dicts
            override_system: If provided, replaces the system prompt

        Returns:
            Messages with system prompt prepended
        """
        system_content = override_system or self.system_prompt

        # Check if messages already have a system message
        if messages and messages[0]["role"] == "system":
            if override_system:
                # Replace existing system message
                return [{"role": "system", "content": system_content}] + messages[1:]
            return messages  # Keep existing system message

        # Prepend system message
        return [{"role": "system", "content": system_content}] + messages

    def greeting(self) -> str:
        """Return YUNO's greeting message."""
        return (
            f"Hello! I'm {self.name} v{self.version}, a research-grade AI assistant. "
            f"How can I help you today?"
        )

    def __repr__(self) -> str:
        return f"YunoIdentity(name={self.name!r}, version={self.version!r})"
