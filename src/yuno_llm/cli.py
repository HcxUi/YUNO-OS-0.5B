"""
YUNO-LLM Command Line Interface (CLI)
=====================================
Executable entry point for the `yuno` command.

Usage:
    yuno                         # Launch interactive OS chat session
    yuno --test                  # Run full E2E OS test suite
    yuno --demo                  # Run live OS workflow demonstration
    yuno --version               # Display YUNO OS version
"""

import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="yuno",
        description="YUNO-LLM Personal AI Operating System (v0.5.0)",
    )
    parser.add_argument("--version", action="version", version="YUNO-LLM v0.5.0")
    parser.add_argument("--config", default="config/yuno_config.yaml", help="Path to config YAML")
    parser.add_argument("--test", action="store_true", help="Run full E2E system test suite")
    parser.add_argument("--demo", action="store_true", help="Run live OS workflow demonstration")

    args = parser.parse_args()

    if args.test:
        from yuno_llm import YunoConfig
        print("Running YUNO OS E2E Verification Test...")
        import subprocess
        subprocess.run([sys.executable, "tests/test_e2e_full_os.py"])
        return

    if args.demo:
        import subprocess
        subprocess.run([sys.executable, "scripts/demo_os_cli.py"])
        return

    # Default: Start interactive CLI session
    from yuno_llm import (
        YunoConfig,
        YunoIdentity,
        YunoMemory,
        YunoToolRegistry,
        YunoPlanner,
        YunoAutomation,
    )

    cfg_path = Path(args.config)
    cfg = YunoConfig.from_yaml(str(cfg_path)) if cfg_path.exists() else YunoConfig()
    identity = YunoIdentity(cfg)
    memory = YunoMemory(cfg)
    tools = YunoToolRegistry(cfg)
    planner = YunoPlanner(cfg)
    automation = YunoAutomation(cfg)

    print("=" * 65)
    print("  YUNO-LLM v0.5.0 PERSONAL AI OPERATING SYSTEM")
    print("=" * 65)
    print(f"  Persona:             {identity.name}")
    print(f"  Short-Term Window:   {cfg.memory.max_short_term_turns} turns")
    print(f"  Episodic Database:   SQLite FTS5")
    print(f"  Registered Tools:    {len(tools.list_tools())} tools")
    print("=" * 65)
    print("  Type 'exit' or 'quit' to exit.\n")

    user_name = memory.get_personal_fact("user_name", "User")
    while True:
        try:
            prompt = input(f"{user_name} > ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("exit", "quit"):
                print("YUNO: Alvida! Phir milenge.")
                break

            intent, tool_name, tool_args = planner.parse_intent(prompt)
            if intent.name == "TOOL_CALL" and tool_name:
                print(f"[YUNO Planning] Tool Call Detected: {tool_name} with args {tool_args}")
                result = tools.run_tool(tool_name, **tool_args)
                safe_out = result.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8")
                print(f"[YUNO Tool Output]\n{safe_out}")
            else:
                print(f"YUNO: Main aapki baat samajh gaya. How can I help you further with '{prompt}'?")
        except (KeyboardInterrupt, EOFError):
            print("\nYUNO: Session ended.")
            break


if __name__ == "__main__":
    main()
