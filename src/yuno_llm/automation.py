"""
YUNO Automation Module
=======================
Provides file organization workflows, reminder task logging/scheduling,
and project scaffolding for YUNO OS.

Usage:
    from yuno_llm.automation import YunoAutomation
    auto = YunoAutomation(config)

    # Organize workspace directory files into subfolders
    summary = auto.organize_directory("workspace")

    # Schedule a reminder
    auto.create_reminder("Check server logs", "10m")

    # Scaffold a Python project
    auto.scaffold_project("my_awesome_app", "python")
"""

import os
import json
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger("yuno_llm.automation")


# Category mapping for file organizer
CATEGORY_EXTENSIONS: Dict[str, List[str]] = {
    "Images": [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg", ".ico"],
    "Documents": [".pdf", ".docx", ".doc", ".xlsx", ".pptx", ".txt", ".csv", ".epub"],
    "Code": [".py", ".js", ".ts", ".html", ".css", ".rs", ".cpp", ".c", ".h", ".java", ".go", ".php", ".sql", ".sh", ".ps1"],
    "Archives": [".zip", ".tar", ".gz", ".7z", ".rar", ".bz2"],
    "Media": [".mp3", ".wav", ".mp4", ".mkv", ".flac", ".avi", ".mov"],
    "Data": [".json", ".yaml", ".yml", ".xml", ".db", ".sqlite"],
}


class YunoAutomation:
    """
    Core YUNO OS Automation engine.

    Handles desktop & workspace file organization, reminder logging,
    and project scaffolding recipes.
    """

    def __init__(self, config=None):
        self.config = config
        self.enabled: bool = True
        self.default_organize_dir: Path = Path("workspace")
        self.reminders_file: Path = Path("workspace/reminders.json")

        if config and hasattr(config, "automation"):
            a = config.automation
            self.enabled = getattr(a, "enabled", True)
            self.default_organize_dir = Path(getattr(a, "default_organize_dir", "workspace"))
            self.reminders_file = Path(getattr(a, "reminders_file", "workspace/reminders.json"))

        self.reminders_file.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. File Organizer Workflow ─────────────────────────────────────────────

    def organize_directory(self, target_dir: Optional[str] = None) -> str:
        """
        Sort files in target_dir into category subfolders based on extension.

        Args:
            target_dir: Directory path to organize (defaults to config default_organize_dir).

        Returns:
            Human-readable summary of organized files.
        """
        dir_path = Path(target_dir) if target_dir else self.default_organize_dir
        if not dir_path.exists():
            return f"[ERROR] Directory not found: {dir_path}"
        if not dir_path.is_dir():
            return f"[ERROR] Path is not a directory: {dir_path}"

        moved_counts: Dict[str, int] = {cat: 0 for cat in CATEGORY_EXTENSIONS}
        moved_counts["Other"] = 0
        total_files = 0

        try:
            for item in list(dir_path.iterdir()):
                # Only move top-level files, skip existing subdirectories
                if item.is_dir():
                    continue

                ext = item.suffix.lower()
                target_cat = "Other"
                for cat, exts in CATEGORY_EXTENSIONS.items():
                    if ext in exts:
                        target_cat = cat
                        break

                dest_dir = dir_path / target_cat
                dest_dir.mkdir(exist_ok=True)
                dest_file = dest_dir / item.name

                # Avoid overwriting existing files
                if dest_file.exists():
                    timestamp = datetime.now().strftime("%H%M%S")
                    dest_file = dest_dir / f"{item.stem}_{timestamp}{item.suffix}"

                shutil.move(str(item), str(dest_file))
                moved_counts[target_cat] += 1
                total_files += 1

            # Format result summary
            summary_lines = [f"[File Organizer Summary — {dir_path.resolve()}]"]
            summary_lines.append(f"  Total files organized: {total_files}")
            for cat, count in moved_counts.items():
                if count > 0:
                    summary_lines.append(f"    - {cat:<10} : {count} files")

            if total_files == 0:
                return f"[File Organizer] No unorganized loose files found in {dir_path}."

            return "\n".join(summary_lines)

        except Exception as e:
            logger.error(f"[Automation] File organization failed for {dir_path}: {e}")
            return f"[ERROR organizing directory: {e}]"

    # ── 2. Reminder Task Logger ───────────────────────────────────────────────

    def create_reminder(self, task: str, time_str: str = "now") -> str:
        """
        Create and log a reminder entry into reminders.json.

        Args:
            task: Reminder description text.
            time_str: Time specification (e.g., "10m", "1h", "tomorrow", "17:00").

        Returns:
            Confirmation message string.
        """
        if not task or not task.strip():
            return "[ERROR] Reminder task text cannot be empty."

        reminders = self._load_reminders()
        timestamp = datetime.now().isoformat(timespec="seconds")
        reminder_entry = {
            "id": len(reminders) + 1,
            "task": task.strip(),
            "time": time_str,
            "created_at": timestamp,
            "status": "pending",
        }
        reminders.append(reminder_entry)
        self._save_reminders(reminders)

        logger.info(f"[Automation] Reminder created: '{task}' @ {time_str}")
        return f"[SUCCESS] Reminder #{reminder_entry['id']} logged: '{task}' (Scheduled: {time_str})"

    def list_reminders(self) -> str:
        """List all pending reminders."""
        reminders = self._load_reminders()
        pending = [r for r in reminders if r.get("status") == "pending"]
        if not pending:
            return "[YUNO] No pending reminders logged."

        lines = ["[Pending Reminders]"]
        for r in pending:
            lines.append(f"  #{r['id']} | {r['task']} (Scheduled: {r['time']})")
        return "\n".join(lines)

    # ── 3. Project Scaffolding Workflow ───────────────────────────────────────

    def scaffold_project(self, project_name: str, project_type: str = "python", target_parent: str = "workspace") -> str:
        """
        Scaffold a new project directory with boilerplate structure.

        Args:
            project_name: Name of the project folder.
            project_type: Type of project ("python", "web", "cpp").
            target_parent: Parent directory path (defaults to "workspace").

        Returns:
            Summary of created files and directories.
        """
        safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in project_name)
        proj_dir = Path(target_parent) / safe_name

        if proj_dir.exists():
            return f"[ERROR] Project directory already exists: {proj_dir.resolve()}"

        try:
            proj_dir.mkdir(parents=True, exist_ok=True)
            created_files = []

            p_type = project_type.lower().strip()
            if p_type == "python":
                src_dir = proj_dir / "src" / safe_name
                src_dir.mkdir(parents=True, exist_ok=True)
                (proj_dir / "tests").mkdir(exist_ok=True)

                main_py = src_dir / "main.py"
                main_py.write_text(f'"""\n{safe_name} — Python Application\n"""\n\ndef main():\n    print("Hello from {safe_name}!")\n\nif __name__ == "__main__":\n    main()\n', encoding="utf-8")
                created_files.append("src/main.py")

                init_py = src_dir / "__init__.py"
                init_py.write_text(f'__version__ = "0.1.0"\n', encoding="utf-8")
                created_files.append("src/__init__.py")

                req_txt = proj_dir / "requirements.txt"
                req_txt.write_text("# Project Dependencies\n", encoding="utf-8")
                created_files.append("requirements.txt")

            elif p_type in ("web", "html", "js"):
                (proj_dir / "css").mkdir(exist_ok=True)
                (proj_dir / "js").mkdir(exist_ok=True)

                index_html = proj_dir / "index.html"
                index_html.write_text(f'<!DOCTYPE html>\n<html lang="en">\n<head>\n  <meta charset="UTF-8">\n  <title>{safe_name}</title>\n  <link rel="stylesheet" href="css/style.css">\n</head>\n<body>\n  <h1>{safe_name}</h1>\n  <script src="js/main.js"></script>\n</body>\n</html>\n', encoding="utf-8")
                created_files.append("index.html")

                style_css = proj_dir / "css" / "style.css"
                style_css.write_text("/* Custom Styles */\nbody {\n  font-family: system-ui, sans-serif;\n  background: #0f172a;\n  color: #f8fafc;\n  padding: 2rem;\n}\n", encoding="utf-8")
                created_files.append("css/style.css")

                main_js = proj_dir / "js" / "main.js"
                main_js.write_text(f'console.log("{safe_name} initialized.");\n', encoding="utf-8")
                created_files.append("js/main.js")

            else:  # Generic / C++ fallback
                (proj_dir / "src").mkdir(exist_ok=True)
                main_cpp = proj_dir / "src" / "main.cpp"
                main_cpp.write_text(f'#include <iostream>\n\nint main() {{\n    std::cout << "Hello from {safe_name}!" << std::endl;\n    return 0;\n}}\n', encoding="utf-8")
                created_files.append("src/main.cpp")

            # Common files for all project types
            readme = proj_dir / "README.md"
            readme.write_text(f"# {safe_name}\n\nProject created with YUNO OS Automation.\n\n## Overview\nProject type: `{p_type}`\n", encoding="utf-8")
            created_files.append("README.md")

            gitignore = proj_dir / ".gitignore"
            gitignore.write_text("__pycache__/\n*.pyc\n.venv/\nnode_modules/\ndist/\nbuild/\n.DS_Store\n", encoding="utf-8")
            created_files.append(".gitignore")

            return (
                f"[SUCCESS] Scaffolding complete for project '{safe_name}' ({p_type})\n"
                f"  Location: {proj_dir.resolve()}\n"
                f"  Created files:\n" + "\n".join(f"    - {f}" for f in created_files)
            )

        except Exception as e:
            logger.error(f"[Automation] Project scaffolding failed for {project_name}: {e}")
            return f"[ERROR scaffolding project: {e}]"

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_reminders(self) -> List[dict]:
        if not self.reminders_file.exists():
            return []
        try:
            return json.loads(self.reminders_file.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _save_reminders(self, reminders: List[dict]) -> None:
        try:
            self.reminders_file.write_text(json.dumps(reminders, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"[Automation] Save reminders failed: {e}")

    def __repr__(self) -> str:
        return f"YunoAutomation(enabled={self.enabled}, dir={self.default_organize_dir!r})"
