"""
YUNO Memory Module
===================
Implements Short-Term, Long-Term, and Project memories for the Personal AI OS.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger("yuno_llm.memory")


class YunoMemory:
    """
    Manages the Personal AI OS memory stack.
    """

    def __init__(self, config=None):
        self.config = config
        self.datasets_dir = Path("datasets")
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

        # Resolve file paths from config
        self.long_term_file = self.datasets_dir / "long_term_memory.json"
        self.project_file = self.datasets_dir / "project_memory.json"

        # Initialize databases
        self.long_term_db = self._load_json(self.long_term_file)
        self.project_db = self._load_json(self.project_file)
        self.short_term_turns: List[Dict[str, str]] = []

    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        """Loads a JSON database file, returning empty dict if missing."""
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading memory file {file_path}: {e}")
        return {}

    def _save_json(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Saves the dict to a JSON database file."""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving memory file {file_path}: {e}")

    # ── 1. Long Term Memory ──────────────────────────────────────────────────

    def remember_personal_fact(self, key: str, value: Any) -> None:
        """Saves a personal user fact to long-term memory."""
        self.long_term_db[key] = value
        self._save_json(self.long_term_file, self.long_term_db)
        logger.info(f"Remembered personal fact: {key} = {value}")

    def forget_personal_fact(self, key: str) -> bool:
        """Removes a personal user fact from long-term memory."""
        if key in self.long_term_db:
            del self.long_term_db[key]
            self._save_json(self.long_term_file, self.long_term_db)
            logger.info(f"Forgot personal fact: {key}")
            return True
        return False

    def get_personal_fact(self, key: str, default: Any = None) -> Any:
        """Retrieves a personal fact from long-term memory."""
        return self.long_term_db.get(key, default)

    # ── 2. Project Memory ────────────────────────────────────────────────────

    def update_project_fact(self, key: str, value: Any) -> None:
        """Saves project workspace facts."""
        self.project_db[key] = value
        self._save_json(self.project_file, self.project_db)
        logger.info(f"Updated project fact: {key}")

    def get_project_fact(self, key: str, default: Any = None) -> Any:
        """Retrieves project workspace facts."""
        return self.project_db.get(key, default)

    # ── 3. Short Term Context ────────────────────────────────────────────────

    def add_chat_turn(self, role: str, content: str) -> None:
        """Appends a turn to the conversational short term context window."""
        self.short_term_turns.append({"role": role, "content": content})
        # Truncate oldest turns if exceeding limit
        max_turns = 30
        if self.config:
            max_turns = getattr(self.config.memory, "max_short_term_turns", 15) * 2
        if len(self.short_term_turns) > max_turns:
            self.short_term_turns = self.short_term_turns[-max_turns:]

    def get_chat_history(self) -> List[Dict[str, str]]:
        """Returns the current conversation context window."""
        return self.short_term_turns

    def clear_short_term(self) -> None:
        """Resets the active conversational history."""
        self.short_term_turns = []

    def compile_memory_context(self) -> str:
        """
        Compiles long-term and project memory profiles into a formatted context block
        suitable for system prompt injection.
        """
        context = []
        if self.long_term_db:
            context.append("[Long-Term Memory Facts]")
            for k, v in self.long_term_db.items():
                context.append(f"- {k}: {v}")
        if self.project_db:
            context.append("[Project Memory Facts]")
            for k, v in self.project_db.items():
                context.append(f"- {k}: {v}")
        return "\n".join(context) if context else ""
