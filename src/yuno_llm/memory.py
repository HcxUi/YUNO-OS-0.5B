"""
YUNO Memory Module
===================
Implements the full YUNO Personal AI OS memory stack:

  ┌─────────────────────────────────────────────────────────┐
  │  Short-Term Memory   → Active conversation turns        │
  │  Long-Term Memory    → Personal user facts (JSON)       │
  │  Project Memory      → Workspace file/code facts (JSON) │
  │  Episodic Memory     → Searchable events (SQLite FTS5)  │
  └─────────────────────────────────────────────────────────┘

Episodic Memory uses SQLite's built-in FTS5 (full-text search) engine,
which requires zero additional dependencies and works completely offline.

Usage:
    memory = YunoMemory(config)

    # Store and recall facts
    memory.remember_personal_fact("user_name", "Piyush")
    name = memory.get_personal_fact("user_name")

    # Add searchable episodic events
    memory.add_episodic_entry("Discussed YUNO-LLM architecture and memory design.")
    results = memory.search_memory("memory design")

    # Short-term conversation context
    memory.add_chat_turn("user", "Hello!")
    history = memory.get_chat_history()
"""

import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger("yuno_llm.memory")


# ── Episodic Memory Entry ─────────────────────────────────────────────────────

class EpisodicEntry:
    """Represents a single searchable memory event."""

    def __init__(
        self,
        rowid: int,
        content: str,
        tags: str,
        timestamp: str,
        source: str,
    ):
        self.rowid = rowid
        self.content = content
        self.tags = tags
        self.timestamp = timestamp
        self.source = source

    def __repr__(self) -> str:
        return f"EpisodicEntry(ts={self.timestamp!r}, content={self.content[:60]!r}...)"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.rowid,
            "content": self.content,
            "tags": self.tags,
            "timestamp": self.timestamp,
            "source": self.source,
        }


# ── Episodic Memory (SQLite FTS5) ─────────────────────────────────────────────

class EpisodicMemory:
    """
    Searchable episodic memory using SQLite FTS5.

    Events are stored as full-text searchable rows, indexed by content
    and tags. Queries use SQLite MATCH syntax (prefix, phrase, boolean).

    No extra dependencies required — SQLite is part of Python's stdlib.
    """

    def __init__(self, db_path: str = "datasets/episodic_memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        """Create the FTS5 virtual table if it doesn't exist."""
        self._conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS episodic_fts
            USING fts5(
                content,
                tags,
                timestamp UNINDEXED,
                source UNINDEXED,
                tokenize = 'porter ascii'
            )
        """)
        self._conn.commit()

    def add(
        self,
        content: str,
        tags: str = "",
        source: str = "conversation",
    ) -> int:
        """
        Store a new episodic memory entry.

        Args:
            content: The text content to store and index.
            tags:    Space-separated tags (e.g., "project yuno memory").
            source:  Origin of the memory ("conversation", "tool", "user").

        Returns:
            Row ID of the inserted entry.
        """
        timestamp = datetime.now().isoformat(timespec="seconds")
        cursor = self._conn.execute(
            "INSERT INTO episodic_fts(content, tags, timestamp, source) VALUES (?, ?, ?, ?)",
            (content, tags, timestamp, source),
        )
        self._conn.commit()
        logger.info(f"[EpisodicMemory] Added entry #{cursor.lastrowid}: {content[:60]!r}")
        return cursor.lastrowid

    def search(self, query: str, top_k: int = 5) -> List[EpisodicEntry]:
        """
        Full-text search over episodic memory using SQLite FTS5 MATCH.

        Automatically adds wildcard prefix matching for better recall.

        Args:
            query: Natural language or keyword search string.
            top_k: Maximum number of results to return.

        Returns:
            List of EpisodicEntry objects, most relevant first.
        """
        if not query.strip():
            return []

        # Build a safer FTS5 query with prefix matching
        fts_query = self._build_fts_query(query)

        try:
            cursor = self._conn.execute(
                """
                SELECT rowid, content, tags, timestamp, source,
                       rank
                FROM episodic_fts
                WHERE episodic_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, top_k),
            )
            rows = cursor.fetchall()
            return [
                EpisodicEntry(
                    rowid=row[0],
                    content=row[1],
                    tags=row[2],
                    timestamp=row[3],
                    source=row[4],
                )
                for row in rows
            ]
        except sqlite3.OperationalError as e:
            # FTS5 query syntax errors should not crash the app
            logger.warning(f"[EpisodicMemory] FTS5 search error for query {query!r}: {e}")
            return []

    def recent(self, limit: int = 10) -> List[EpisodicEntry]:
        """Return the most recently added episodic entries."""
        cursor = self._conn.execute(
            """
            SELECT rowid, content, tags, timestamp, source
            FROM episodic_fts
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        return [
            EpisodicEntry(rowid=r[0], content=r[1], tags=r[2], timestamp=r[3], source=r[4])
            for r in rows
        ]

    def delete(self, rowid: int) -> bool:
        """Delete an episodic entry by row ID (user-controlled memory deletion)."""
        try:
            self._conn.execute("DELETE FROM episodic_fts WHERE rowid = ?", (rowid,))
            self._conn.commit()
            logger.info(f"[EpisodicMemory] Deleted entry #{rowid}")
            return True
        except Exception as e:
            logger.error(f"[EpisodicMemory] Delete error: {e}")
            return False

    def clear_all(self) -> None:
        """Wipe all episodic memories (user-controlled)."""
        self._conn.execute("DELETE FROM episodic_fts")
        self._conn.commit()
        logger.info("[EpisodicMemory] All episodic memories cleared.")

    def count(self) -> int:
        """Return total number of stored episodic entries."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM episodic_fts"
        ).fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()

    @staticmethod
    def _build_fts_query(query: str) -> str:
        """
        Converts a natural language query into an FTS5-safe MATCH expression.
        Strips special FTS5 characters and adds prefix wildcards.
        """
        # Remove FTS5 special chars except quotes
        cleaned = re.sub(r'[^\w\s"\']+', " ", query)
        # Split into tokens, add prefix wildcard to each
        tokens = [f"{t}*" for t in cleaned.split() if len(t) >= 2]
        return " OR ".join(tokens) if tokens else query

# Import re here for the staticmethod above
import re  # noqa: E402 (intentional late import for clarity)


# ── Main Memory Manager ───────────────────────────────────────────────────────

class YunoMemory:
    """
    The YUNO Personal AI OS memory stack.

    Manages all four memory layers:
    - Short-term: Active conversation turns (in-memory list)
    - Long-term: Personal user facts (JSON file)
    - Project: Workspace/code facts (JSON file)
    - Episodic: Searchable events and discussions (SQLite FTS5)
    """

    def __init__(self, config=None):
        self.config = config

        # ── Resolve paths ────────────────────────────────────────────────────
        datasets_dir = Path("datasets")
        datasets_dir.mkdir(parents=True, exist_ok=True)

        lt_file = "datasets/long_term_memory.json"
        proj_file = "datasets/project_memory.json"
        ep_db = "datasets/episodic_memory.db"
        self._max_short_term = 15
        self._max_episodic_results = 5

        if config and hasattr(config, "memory"):
            m = config.memory
            lt_file = getattr(m, "long_term_file", lt_file)
            proj_file = getattr(m, "project_file", proj_file)
            ep_db = getattr(m, "episodic_db_file", ep_db)
            self._max_short_term = getattr(m, "max_short_term_turns", 15)
            self._max_episodic_results = getattr(m, "max_episodic_results", 5)

        self.long_term_file = Path(lt_file)
        self.project_file = Path(proj_file)

        # ── Initialize stores ────────────────────────────────────────────────
        self.long_term_db: Dict[str, Any] = self._load_json(self.long_term_file)
        self.project_db: Dict[str, Any] = self._load_json(self.project_file)
        self.short_term_turns: List[Dict[str, str]] = []
        self.episodic = EpisodicMemory(db_path=ep_db)

    # ── JSON Helpers ──────────────────────────────────────────────────────────

    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        """Loads a JSON database file, returning empty dict if missing."""
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"[Memory] Error loading {file_path}: {e}")
        return {}

    def _save_json(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Saves a dict to a JSON database file."""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[Memory] Error saving {file_path}: {e}")

    # ── 1. Long-Term Personal Memory ──────────────────────────────────────────

    def remember_personal_fact(self, key: str, value: Any) -> None:
        """Save a personal user fact to long-term memory (e.g., name, preferences)."""
        self.long_term_db[key] = value
        self._save_json(self.long_term_file, self.long_term_db)
        logger.info(f"[Memory] Long-term fact stored: {key} = {value}")

    def forget_personal_fact(self, key: str) -> bool:
        """Remove a personal user fact from long-term memory."""
        if key in self.long_term_db:
            del self.long_term_db[key]
            self._save_json(self.long_term_file, self.long_term_db)
            logger.info(f"[Memory] Long-term fact removed: {key}")
            return True
        return False

    def get_personal_fact(self, key: str, default: Any = None) -> Any:
        """Retrieve a personal fact by exact key."""
        return self.long_term_db.get(key, default)

    def list_personal_facts(self) -> Dict[str, Any]:
        """Return all stored personal facts."""
        return dict(self.long_term_db)

    # ── 2. Project Memory ─────────────────────────────────────────────────────

    def update_project_fact(self, key: str, value: Any) -> None:
        """Save a project workspace fact (file paths, code context, etc.)."""
        self.project_db[key] = value
        self._save_json(self.project_file, self.project_db)
        logger.info(f"[Memory] Project fact stored: {key}")

    def get_project_fact(self, key: str, default: Any = None) -> Any:
        """Retrieve a project fact by exact key."""
        return self.project_db.get(key, default)

    # ── 3. Short-Term Conversation Context ────────────────────────────────────

    def add_chat_turn(self, role: str, content: str) -> None:
        """Append a turn to the short-term conversation window."""
        self.short_term_turns.append({"role": role, "content": content})
        # Keep within the configured window size
        max_messages = self._max_short_term * 2  # user + assistant pairs
        if len(self.short_term_turns) > max_messages:
            self.short_term_turns = self.short_term_turns[-max_messages:]

    def get_chat_history(self) -> List[Dict[str, str]]:
        """Return the current short-term conversation context."""
        return self.short_term_turns

    def clear_short_term(self) -> None:
        """Reset the active conversational history."""
        self.short_term_turns = []
        logger.info("[Memory] Short-term memory cleared.")

    # ── 4. Episodic Memory ────────────────────────────────────────────────────

    def add_episodic_entry(
        self,
        content: str,
        tags: str = "",
        source: str = "conversation",
    ) -> int:
        """
        Store a searchable episodic memory entry.

        Args:
            content: Text description of the event/discussion.
            tags:    Space-separated topic tags for better retrieval.
            source:  Origin — "conversation", "user", "tool".

        Returns:
            Row ID of the stored entry.
        """
        return self.episodic.add(content=content, tags=tags, source=source)

    def search_memory(
        self, query: str, top_k: Optional[int] = None
    ) -> List[EpisodicEntry]:
        """
        Full-text search across all episodic memories.

        Args:
            query: Natural language search string.
            top_k: Max results (defaults to config.memory.max_episodic_results).

        Returns:
            List of EpisodicEntry objects, ranked by relevance.
        """
        k = top_k or self._max_episodic_results
        results = self.episodic.search(query, top_k=k)
        logger.debug(f"[Memory] Episodic search '{query}' → {len(results)} results")
        return results

    def forget_episodic(self, rowid: int) -> bool:
        """Delete a specific episodic memory by ID (user-controlled)."""
        return self.episodic.delete(rowid)

    def clear_all_episodic(self) -> None:
        """Wipe all episodic memories (user-controlled)."""
        self.episodic.clear_all()

    # ── 5. Compiled System Prompt Context ─────────────────────────────────────

    def compile_memory_context(self, search_query: Optional[str] = None) -> str:
        """
        Compiles long-term, project, and relevant episodic memory into a
        formatted string block for injection into the system prompt.

        Args:
            search_query: If provided, performs an episodic search and includes
                          relevant past events in the context.

        Returns:
            Formatted memory context string, or empty string if no memories exist.
        """
        context: List[str] = []

        # Long-term personal facts
        if self.long_term_db:
            context.append("[Long-Term Memory]")
            for k, v in self.long_term_db.items():
                context.append(f"  - {k}: {v}")

        # Project facts
        if self.project_db:
            context.append("[Project Memory]")
            for k, v in self.project_db.items():
                context.append(f"  - {k}: {v}")

        # Relevant episodic memories
        if search_query:
            hits = self.search_memory(search_query)
            if hits:
                context.append("[Relevant Past Events]")
                for hit in hits:
                    context.append(f"  [{hit.timestamp}] {hit.content}")

        return "\n".join(context) if context else ""

    def get_stats(self) -> Dict[str, int]:
        """Return counts for each memory layer."""
        return {
            "long_term_facts": len(self.long_term_db),
            "project_facts": len(self.project_db),
            "short_term_turns": len(self.short_term_turns),
            "episodic_entries": self.episodic.count(),
        }
