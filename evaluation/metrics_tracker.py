"""
YUNO-LLM: Metrics Tracker
===========================
Logs evaluation results to JSON files in experiments/.
Enables tracking progress across versions and training runs.

Usage:
    tracker = MetricsTracker()
    tracker.log({
        "model": "yuno-v0.1",
        "perplexity": 12.3,
        "hellaswag_acc": 0.47,
        "gsm8k_acc": 0.08,
    })
    tracker.summary()
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List


class MetricsTracker:
    """
    Tracks evaluation metrics across experiments.

    All results are written to `experiments/` as JSON files.
    One file per experiment run, named by timestamp.
    """

    def __init__(self, experiments_dir: str = "experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        metrics: Dict[str, Any],
        run_name: Optional[str] = None,
        checkpoint: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Path:
        """
        Log a set of metrics for a single evaluation run.

        Args:
            metrics: Dict of metric_name → value
            run_name: Human-readable run name (e.g., "yuno-v0.1-lora-r16")
            checkpoint: Path to checkpoint being evaluated
            notes: Free-form notes about this run

        Returns:
            Path to the saved JSON file
        """
        ts = datetime.now().isoformat()
        ts_safe = ts.replace(":", "-").replace(".", "-")

        record = {
            "timestamp": ts,
            "run_name": run_name or f"run_{ts_safe}",
            "checkpoint": checkpoint,
            "notes": notes,
            "metrics": metrics,
        }

        filename = self.experiments_dir / f"{ts_safe}_{run_name or 'eval'}.json"
        with open(filename, "w") as f:
            json.dump(record, f, indent=2)

        print(f"  [SUCCESS] Metrics saved -> {filename}")
        return filename

    def load_all(self) -> List[Dict]:
        """Load all experiment records, sorted by timestamp."""
        records = []
        for path in sorted(self.experiments_dir.glob("*.json")):
            with open(path) as f:
                records.append(json.load(f))
        return records

    def summary(self) -> None:
        """Print a table of all experiments."""
        records = self.load_all()
        if not records:
            print("  No experiments logged yet.")
            return

        print(f"\n  YUNO-LLM Experiment History")
        print(f"  {'='*70}")
        print(f"  {'Run':<30} {'PPL':>6} {'HellaSwag':>10} {'GSM8K':>8} {'HumanEval':>10}")
        print(f"  {'-'*70}")
        for r in records:
            m = r.get("metrics", {})
            print(
                f"  {r.get('run_name', 'unknown'):<30} "
                f"{m.get('perplexity', '-'):>6} "
                f"{m.get('hellaswag_acc', '-'):>10} "
                f"{m.get('gsm8k_acc', '-'):>8} "
                f"{m.get('humaneval_pass_at_1', '-'):>10}"
            )
        print()
