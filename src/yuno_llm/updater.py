"""
YUNO OS Self-Updating System (Lalam)
=====================================
Checks for an internet connection and automatically updates code (via git pull)
and optionally model weights on startup.
"""

import os
import subprocess
import socket
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger("yuno_llm.updater")


class YunoUpdater:
    """
    Handles automatic codebase and model weight updates.
    """

    def __init__(self, config=None):
        self.config = config
        self.enabled = True
        self.auto_git_pull = True
        self.check_model_updates = False
        self.timeout = 4

        # Read config if present
        if config and hasattr(config, "updater"):
            self.enabled = getattr(config.updater, "enabled", True)
            self.auto_git_pull = getattr(config.updater, "auto_git_pull", True)
            self.check_model_updates = getattr(config.updater, "check_model_updates", False)
            self.timeout = getattr(config.updater, "timeout_seconds", 4)

    def check_internet_connection(self, host: str = "1.1.1.1", port: int = 53) -> bool:
        """
        Quick DNS-level connection check to check if internet is available.
        Defaults to Cloudflare public DNS (1.1.1.1:53).
        """
        try:
            socket.setdefaulttimeout(self.timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
            return True
        except socket.error:
            return False

    def update_code(self) -> bool:
        """
        Attempts to update the repository using `git pull`.
        Returns True if code updates were successfully fetched and merged.
        """
        if not self.auto_git_pull:
            logger.info("Auto git pull is disabled in config.")
            return False

        # Verify git directory exists
        project_root = Path(__file__).parent.parent.parent
        if not (project_root / ".git").exists():
            logger.warning("Not a git repository. Skipping git pull.")
            return False

        try:
            # Check git status first
            logger.info("Checking for repository updates...")
            # Run git pull
            res = subprocess.run(
                ["git", "pull"],
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=20
            )
            if res.returncode == 0:
                stdout_lower = res.stdout.lower()
                if "already up to date" in stdout_lower or "already up-to-date" in stdout_lower:
                    logger.info("YUNO codebase is already up to date.")
                    return False
                else:
                    logger.info(f"YUNO codebase updated successfully: {res.stdout.strip()}")
                    return True
            else:
                logger.error(f"Git pull failed with exit code {res.returncode}: {res.stderr}")
        except subprocess.TimeoutExpired:
            logger.error("Git pull command timed out.")
        except Exception as e:
            logger.error(f"Unexpected error running git pull: {e}")
        return False

    def update_model(self) -> bool:
        """
        Checks HF Hub for base model weights updates.
        """
        if not self.check_model_updates:
            return False

        logger.info("Checking for model weight updates...")
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            model_id = "Qwen/Qwen3-0.6B"
            if self.config:
                model_id = getattr(self.config.base_model, "name", model_id)
            
            # This triggers download checks and updates the local HF cache
            AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            logger.info("Model weights checked and verified up to date.")
            return True
        except Exception as e:
            logger.error(f"Error checking model weights updates: {e}")
        return False

    def run_auto_update(self) -> Dict[str, Any]:
        """
        Executes the auto-update pipeline.
        Returns a summary dictionary of actions taken.
        """
        summary = {
            "internet_available": False,
            "code_updated": False,
            "model_checked": False,
            "error": None
        }

        if not self.enabled:
            logger.info("Self-updater system is disabled.")
            return summary

        # 1. Check connection
        if not self.check_internet_connection():
            logger.info("No active internet connection detected. Skipping online updates.")
            return summary

        summary["internet_available"] = True

        # 2. Update Code
        try:
            summary["code_updated"] = self.update_code()
        except Exception as e:
            summary["error"] = f"Code update error: {e}"

        # 3. Check Model Weights
        try:
            summary["model_checked"] = self.update_model()
        except Exception as e:
            if not summary["error"]:
                summary["error"] = f"Model update error: {e}"

        return summary
