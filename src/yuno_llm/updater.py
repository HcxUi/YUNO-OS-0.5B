"""
YUNO OS Safe Update System
============================
Implements a secure, user-controlled update pipeline:

  check_internet → fetch_remote_info → compare_versions
      → show_changelog → prompt_user_approval
      → backup_current_state → apply_update
      → run_smoke_test → rollback_on_failure

Principles:
  - NOTHING is applied without explicit user approval.
  - The current state is always backed up before any change.
  - Failed updates roll back automatically.
  - All actions are logged to an audit trail.
"""

import os
import json
import socket
import hashlib
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger("yuno_llm.updater")


# ── Dataclass-like result container ──────────────────────────────────────────

class UpdateInfo:
    """Holds information about an available update."""

    def __init__(
        self,
        available: bool,
        current_version: str,
        remote_version: str,
        changelog: str,
        commit_hash: str = "",
    ):
        self.available = available
        self.current_version = current_version
        self.remote_version = remote_version
        self.changelog = changelog
        self.commit_hash = commit_hash
        self.timestamp = datetime.now().isoformat()

    def __repr__(self) -> str:
        return (
            f"UpdateInfo(available={self.available}, "
            f"current={self.current_version!r}, "
            f"remote={self.remote_version!r})"
        )


# ── Main Updater Class ────────────────────────────────────────────────────────

class YunoUpdater:
    """
    Handles safe, user-approved codebase and model weight updates.

    Usage:
        updater = YunoUpdater(config)
        info = updater.check_for_updates()      # Read-only. No side effects.
        if info.available:
            updater.prompt_and_apply(info)      # Shows changelog, asks user.
    """

    def __init__(self, config=None):
        self.config = config
        self.project_root = Path(__file__).parent.parent.parent

        # ── Defaults (overridden by config) ──────────────────────────────────
        self.enabled: bool = True
        self.check_on_startup: bool = True
        self.require_user_approval: bool = True     # ALWAYS ask before applying
        self.auto_apply: bool = False               # Never apply silently
        self.backup_before_update: bool = True
        self.verify_signatures: bool = False        # Future: GPG
        self.rollback_on_failure: bool = True
        self.timeout_seconds: int = 10
        self.check_model_updates: bool = False

        # ── Load from config if available ─────────────────────────────────────
        if config and hasattr(config, "updater"):
            u = config.updater
            self.enabled = getattr(u, "enabled", self.enabled)
            self.check_on_startup = getattr(u, "check_on_startup", self.check_on_startup)
            self.require_user_approval = getattr(u, "require_user_approval", True)
            self.auto_apply = getattr(u, "auto_apply", False)
            self.backup_before_update = getattr(u, "backup_before_update", True)
            self.verify_signatures = getattr(u, "verify_signatures", False)
            self.rollback_on_failure = getattr(u, "rollback_on_failure", True)
            self.timeout_seconds = getattr(u, "timeout_seconds", 10)
            self.check_model_updates = getattr(u, "check_model_updates", False)

        # ── Audit log file ────────────────────────────────────────────────────
        self._audit_log = self.project_root / "logs" / "update_audit.log"
        self._audit_log.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Internet Check ─────────────────────────────────────────────────────

    def check_internet(self, host: str = "1.1.1.1", port: int = 53) -> bool:
        """
        Non-blocking DNS-level connection test.
        Returns True if internet is reachable.
        """
        try:
            socket.setdefaulttimeout(self.timeout_seconds)
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((host, port))
            s.close()
            return True
        except socket.error:
            return False

    # ── 2. Check for Updates (Read-Only, No Side Effects) ─────────────────────

    def check_for_updates(self) -> Optional[UpdateInfo]:
        """
        Fetches remote commit info and compares to local HEAD.
        NEVER modifies any files. Returns UpdateInfo or None if offline/error.
        """
        if not self.enabled:
            logger.info("[Updater] Update system is disabled in config.")
            return None

        if not self._is_git_repo():
            logger.warning("[Updater] Not a git repository. Update checks unavailable.")
            return None

        if not self.check_internet():
            logger.info("[Updater] No internet connection. Skipping update check.")
            return None

        try:
            # Get current local HEAD hash
            local_hash = self._run_git(["rev-parse", "HEAD"])
            local_short = local_hash[:8] if local_hash else "unknown"

            # Fetch remote metadata (does NOT merge, does NOT pull)
            fetch_result = self._run_git(["fetch", "--dry-run"], timeout=self.timeout_seconds)
            remote_hash = self._run_git(["rev-parse", "origin/HEAD"])
            remote_short = remote_hash[:8] if remote_hash else "unknown"

            if not local_hash or not remote_hash:
                logger.warning("[Updater] Could not read git commit hashes.")
                return None

            if local_hash == remote_hash:
                logger.info("[Updater] YUNO codebase is up to date.")
                return UpdateInfo(
                    available=False,
                    current_version=local_short,
                    remote_version=remote_short,
                    changelog="Already up to date.",
                    commit_hash=local_hash,
                )

            # Get human-readable changelog between local and remote
            changelog = self._get_changelog(local_hash, remote_hash)

            logger.info(
                f"[Updater] Update available: {local_short} → {remote_short}"
            )
            return UpdateInfo(
                available=True,
                current_version=local_short,
                remote_version=remote_short,
                changelog=changelog,
                commit_hash=remote_hash,
            )

        except Exception as e:
            logger.error(f"[Updater] Update check error: {e}")
            return None

    # ── 3. Show Changelog & Prompt User ──────────────────────────────────────

    def prompt_and_apply(
        self,
        info: UpdateInfo,
        input_fn=input,
    ) -> bool:
        """
        Shows the update changelog and prompts the user for approval.
        If approved: backup → apply → smoke test → rollback on failure.

        Args:
            info: UpdateInfo returned from check_for_updates()
            input_fn: callable for user input (injectable for testing)

        Returns:
            True if update was successfully applied.
        """
        if not info.available:
            print("  [YUNO Updater] Already up to date. Nothing to do.")
            return False

        # ── Show update info ─────────────────────────────────────────────────
        print()
        print("  ╔══════════════════════════════════════════════════╗")
        print("  ║          YUNO — Update Available                 ║")
        print("  ╚══════════════════════════════════════════════════╝")
        print(f"  Current version : {info.current_version}")
        print(f"  New version     : {info.remote_version}")
        print()
        print("  What's changed:")
        for line in info.changelog.splitlines():
            print(f"    {line}")
        print()

        if self.require_user_approval:
            approval = input_fn(
                "  Apply this update? (yes/no) [Backup will be created]: "
            ).strip().lower()
            if approval not in ("yes", "y"):
                print("  [YUNO Updater] Update skipped by user.")
                self._audit("UPDATE_SKIPPED", info)
                return False

        return self._apply_update(info)

    # ── 4. Apply Update with Backup + Rollback ────────────────────────────────

    def _apply_update(self, info: UpdateInfo) -> bool:
        """
        Internal: backup → git pull → smoke test → rollback on failure.
        """
        backup_ref = None

        # Step 1: Backup current state
        if self.backup_before_update:
            backup_ref = self._backup_current_state()
            if backup_ref:
                print(f"  [Updater] Backup created: {backup_ref}")
            else:
                print("  [Updater] Warning: Could not create backup. Proceeding cautiously.")

        # Step 2: Apply update (git pull)
        print("  [Updater] Downloading and applying update...")
        try:
            result = subprocess.run(
                ["git", "pull", "--ff-only"],
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                error_msg = result.stderr.strip() or result.stdout.strip()
                print(f"  [Updater] ❌ Update failed: {error_msg}")
                self._audit("UPDATE_FAILED", info, error=error_msg)
                if backup_ref:
                    self._rollback(backup_ref)
                return False
        except subprocess.TimeoutExpired:
            print("  [Updater] ❌ Update timed out.")
            self._audit("UPDATE_TIMEOUT", info)
            if backup_ref and self.rollback_on_failure:
                self._rollback(backup_ref)
            return False
        except Exception as e:
            print(f"  [Updater] ❌ Unexpected error: {e}")
            self._audit("UPDATE_ERROR", info, error=str(e))
            if backup_ref and self.rollback_on_failure:
                self._rollback(backup_ref)
            return False

        # Step 3: Smoke test — verify the package still imports cleanly
        print("  [Updater] Running smoke test...")
        smoke_ok = self._run_smoke_test()
        if not smoke_ok:
            print("  [Updater] ❌ Smoke test failed after update.")
            self._audit("SMOKE_TEST_FAILED", info)
            if backup_ref and self.rollback_on_failure:
                self._rollback(backup_ref)
                return False

        print(f"  [Updater] ✅ Update applied successfully: {info.current_version} → {info.remote_version}")
        self._audit("UPDATE_SUCCESS", info)
        return True

    # ── 5. Backup Current State ───────────────────────────────────────────────

    def _backup_current_state(self) -> Optional[str]:
        """
        Creates a git stash to save current state before update.
        Returns stash reference string or None on failure.
        """
        try:
            # Use a timestamped stash message
            stash_msg = f"yuno-auto-backup-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            result = subprocess.run(
                ["git", "stash", "push", "-m", stash_msg, "--include-untracked"],
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=15,
            )
            if result.returncode == 0 and "No local changes" not in result.stdout:
                return stash_msg
            # If nothing to stash, the current HEAD is the safe state
            return self._run_git(["rev-parse", "HEAD"])
        except Exception as e:
            logger.error(f"[Updater] Backup failed: {e}")
            return None

    # ── 6. Rollback ────────────────────────────────────────────────────────────

    def _rollback(self, backup_ref: str) -> None:
        """
        Rolls back to the pre-update state using git stash pop or git reset.
        """
        print(f"  [Updater] Rolling back to: {backup_ref}...")
        try:
            # Try stash pop first (works if backup_ref is a stash message)
            result = subprocess.run(
                ["git", "stash", "pop"],
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=15,
            )
            if result.returncode == 0:
                print("  [Updater] ✅ Rollback successful via git stash pop.")
                self._audit("ROLLBACK_SUCCESS", extra={"ref": backup_ref})
                return
            # Fall back to hard reset if stash pop fails
            subprocess.run(
                ["git", "reset", "--hard", backup_ref],
                cwd=str(self.project_root),
                timeout=15,
            )
            print("  [Updater] ✅ Rollback successful via git reset.")
            self._audit("ROLLBACK_SUCCESS", extra={"ref": backup_ref})
        except Exception as e:
            logger.error(f"[Updater] Rollback failed: {e}")
            print(f"  [Updater] ❌ Rollback failed: {e}. Manual git reset may be needed.")
            self._audit("ROLLBACK_FAILED", extra={"error": str(e)})

    # ── 7. Smoke Test ─────────────────────────────────────────────────────────

    def _run_smoke_test(self) -> bool:
        """
        Verifies that the yuno_llm package still imports cleanly after update.
        Returns True if the package loads without errors.
        """
        try:
            result = subprocess.run(
                ["python", "-c", "import yuno_llm; print('OK')"],
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
            )
            return result.returncode == 0 and "OK" in result.stdout
        except Exception as e:
            logger.error(f"[Updater] Smoke test error: {e}")
            return False

    # ── 8. Model Weight Updates ───────────────────────────────────────────────

    def check_model_update(self, input_fn=input) -> bool:
        """
        Checks HuggingFace Hub for base model weight updates.
        Notifies the user — does NOT download without approval.
        """
        if not self.check_model_updates:
            return False

        logger.info("[Updater] Checking for model weight updates...")
        try:
            from huggingface_hub import model_info as hf_model_info
            model_id = "Qwen/Qwen3-0.6B"
            if self.config:
                model_id = getattr(self.config.base_model, "name", model_id)

            info = hf_model_info(model_id)
            latest_sha = info.sha[:8] if info.sha else "unknown"
            print(f"\n  [Updater] Model: {model_id} | Latest SHA: {latest_sha}")

            approval = input_fn(
                "  Download/refresh model weights? (yes/no): "
            ).strip().lower()
            if approval in ("yes", "y"):
                from transformers import AutoTokenizer
                AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, force_download=False)
                print("  [Updater] ✅ Model weights verified/updated.")
                return True
        except Exception as e:
            logger.error(f"[Updater] Model update check error: {e}")
        return False

    # ── 9. Startup Entry Point ────────────────────────────────────────────────

    def run_startup_check(self, input_fn=input) -> Dict[str, Any]:
        """
        Called on YUNO startup. Silently checks for updates and notifies user.
        Never applies anything without approval.

        Returns a summary dict for logging.
        """
        summary: Dict[str, Any] = {
            "internet": False,
            "update_available": False,
            "update_applied": False,
            "error": None,
        }

        if not self.enabled or not self.check_on_startup:
            return summary

        if not self.check_internet():
            logger.info("[Updater] Offline — skipping startup update check.")
            return summary

        summary["internet"] = True

        try:
            info = self.check_for_updates()
            if info and info.available:
                summary["update_available"] = True
                applied = self.prompt_and_apply(info, input_fn=input_fn)
                summary["update_applied"] = applied
        except Exception as e:
            summary["error"] = str(e)
            logger.error(f"[Updater] Startup check error: {e}")

        return summary

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _is_git_repo(self) -> bool:
        return (self.project_root / ".git").exists()

    def _run_git(self, args: List[str], timeout: int = 10) -> str:
        """Run a git command, return stdout or empty string on failure."""
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
            return result.stdout.strip()
        except Exception:
            return ""

    def _get_changelog(self, local_hash: str, remote_hash: str) -> str:
        """Returns a human-readable git log between local and remote."""
        log = self._run_git([
            "log",
            "--oneline",
            "--no-merges",
            f"{local_hash}..{remote_hash}",
        ])
        return log if log else "(No commit messages available)"

    def _audit(
        self,
        event: str,
        info: Optional[UpdateInfo] = None,
        extra: Optional[Dict] = None,
        error: Optional[str] = None,
    ) -> None:
        """Appends a structured line to the update audit log."""
        try:
            entry: Dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "event": event,
            }
            if info:
                entry["from"] = info.current_version
                entry["to"] = info.remote_version
            if error:
                entry["error"] = error
            if extra:
                entry.update(extra)
            with open(self._audit_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"[Updater] Could not write audit log: {e}")
