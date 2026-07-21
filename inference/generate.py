"""
YUNO-LLM: CLI Inference
========================
Command-line interface for running YUNO-LLM interactively.

Usage:
    # Interactive chat (REPL):
    python inference/generate.py

    # Single prompt:
    python inference/generate.py --prompt "What is machine learning?"

    # Load a LoRA adapter:
    python inference/generate.py --adapter checkpoints/sft_run_001/final

    # Non-streaming, greedy:
    python inference/generate.py --no-stream --greedy
"""

import sys
import numpy as np
np.complex = complex  # Monkeypatch for legacy librosa compatibility with modern numpy
import argparse
import logging
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)  # Keep inference clean


YUNO_BANNER = r"""
  __  __ _   _ _   _  ___    _     _     __  __
 |  \/  | | | | \ | |/ _ \  | |   | |   |  \/  |
 | |\/| | | | |  \| | | | | | |   | |   | |\/| |
 | |  | | |_| | |\  | |_| | | |___| |___| |  | |
 |_|  |_|\___/|_| \_|\___/  |_____|_____|_|  |_|

  Research-Grade Open-Source LLM  •  v0.1.0
"""


def load_model(model_id: str, adapter_path: str = None):
    """Load model, tokenizer, and optionally a LoRA adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n  Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()

    # Load LoRA adapter if provided
    if adapter_path:
        from peft import PeftModel
        print(f"  Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()  # Merge for faster inference
        print(f"  [SUCCESS] Adapter merged.")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  [SUCCESS] Model ready  ({total_params:,} parameters)\n")
    return model, tokenizer


def chat_loop(model, tokenizer, args):
    """Interactive multi-turn chat REPL."""
    import torch
    from transformers import TextStreamer

    print(YUNO_BANNER)

    # Load YunoConfig
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from yuno_llm.config import YunoConfig
        from yuno_llm.generation import YunoGenerator
        config = YunoConfig.from_yaml(str(ROOT / "config" / "yuno_config.yaml"))
        generator = YunoGenerator(model, tokenizer, config)
        identity_name = config.identity.name
        print(f"  Identity: {identity_name} v{config.identity.version}")
    except Exception as e:
        generator = None
        identity_name = "YUNO"

    print(f"  Type your message and press Enter.")
    print(f"  Commands: /reset (clear history), /quit or /exit")
    print(f"  {'-'*50}\n")

    while True:
        try:
            user_input = input(f"  You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n  Goodbye!\n")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            print(f"\n  Goodbye!\n")
            break
        if user_input.lower() == "/reset":
            if generator:
                generator.reset_history()
            print(f"  [History cleared]\n")
            continue

        # Generate response
        print(f"\n  {identity_name}: ", end="", flush=True)

        if generator and not args.no_stream:
            for chunk in generator.stream(user_input):
                # Clean up unicode emojis that break Windows terminals
                chunk_safe = chunk.encode('cp1252', errors='replace').decode('cp1252')
                print(chunk_safe, end="", flush=True)
        elif generator:
            response = generator.chat(user_input)
            response_safe = response.encode('cp1252', errors='replace').decode('cp1252')
            print(response_safe)
        else:
            # Fallback without YunoGenerator
            messages = [{"role": "user", "content": user_input}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=not args.greedy,
                    temperature=args.temperature if not args.greedy else 1.0,
                    top_p=args.top_p,
                )
            new_tokens = output[0][inputs.input_ids.shape[1]:]
            print(tokenizer.decode(new_tokens, skip_special_tokens=True))

        print(f"\n")


def single_prompt(model, tokenizer, args):
    """Generate a single response and exit."""
    import torch

    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=not args.greedy,
            temperature=args.temperature if not args.greedy else 1.0,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
    new_tokens = output[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    response_safe = response.encode('cp1252', errors='replace').decode('cp1252')
    print(response_safe)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YUNO-LLM Inference")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Model ID or local path")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter checkpoint")
    parser.add_argument("--prompt", default=None, help="Single prompt (non-interactive)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming")
    args = parser.parse_args()

    # Load YunoConfig to check updater settings
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from yuno_llm.config import YunoConfig
        from yuno_llm.updater import YunoUpdater
        config = YunoConfig.from_yaml(str(ROOT / "config" / "yuno_config.yaml"))
        print("\n  [SYSTEM] Checking for Lalam self-system updates...")
        updater = YunoUpdater(config)
        summary = updater.run_auto_update()
        if summary["internet_available"]:
            if summary["code_updated"]:
                print("  [SUCCESS] YUNO codebase has been updated. Please restart for changes to take effect.")
            else:
                print("  [SUCCESS] YUNO codebase is up to date.")
        else:
            print("  [INFO] Offline mode: Skipping updates.")
    except Exception as e:
        print(f"  [WARNING] Self-updater failed: {e}")

    model, tokenizer = load_model(args.model, args.adapter)

    if args.prompt:
        single_prompt(model, tokenizer, args)
    else:
        chat_loop(model, tokenizer, args)
