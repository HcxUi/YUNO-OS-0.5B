"""
YUNO-LLM: Download Base Model
==============================
Downloads the Qwen3 base model weights from HuggingFace Hub.
Run once before any other scripts.

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --model Qwen/Qwen3-1.7B
    python scripts/download_model.py --model Qwen/Qwen3-0.6B --cache-dir models/base
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def download_model(model_name: str, save_dir: str, token: str = None) -> None:
    """Download model weights and tokenizer from HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    save_path = ROOT / save_dir / model_name.replace("/", "--")
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"")
    print(f"  YUNO-LLM — Base Model Download")
    print(f"  {'='*40}")
    print(f"  Model:      {model_name}")
    print(f"  Save path:  {save_path}")
    print(f"")

    try:
        local_dir = snapshot_download(
            repo_id=model_name,
            local_dir=str(save_path),
            token=token,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
        )
        print(f"\n  [SUCCESS] Download complete -> {local_dir}")
        print(f"")
        _print_downloaded_files(Path(local_dir))
    except Exception as e:
        print(f"\n  [ERROR] Download failed: {e}")
        print(f"")
        print(f"  Troubleshooting:")
        print(f"  1. Check your internet connection")
        print(f"  2. For gated models, set HF_TOKEN environment variable")
        print(f"     export HF_TOKEN=hf_your_token_here")
        print(f"  3. Try: huggingface-cli login")
        sys.exit(1)


def _print_downloaded_files(path: Path) -> None:
    """Print the downloaded model files with sizes."""
    print(f"  Downloaded files:")
    total_size = 0
    for f in sorted(path.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            total_size += f.stat().st_size
            print(f"    {f.name:<50} {size_mb:>8.1f} MB")
    print(f"  {'─'*60}")
    print(f"  {'Total:':<50} {total_size / (1024*1024):>8.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download YUNO-LLM base model")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace model ID (default: Qwen/Qwen3-0.6B)"
    )
    parser.add_argument(
        "--save-dir",
        default="models/base",
        help="Directory to save model (relative to project root)"
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace API token (or set HF_TOKEN env var)"
    )
    args = parser.parse_args()
    download_model(args.model, args.save_dir, args.token)
