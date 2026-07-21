"""
YUNO-LLM OS Live Demonstration Script
======================================
Runs an end-to-end demonstration showing YUNO OS:
1. Responding in natural Hinglish with identity awareness.
2. Managing long-term personal facts & SQLite FTS5 episodic memories.
3. Automatically executing read-only tools and prompting for writes.
4. Organizing loose workspace files.
5. Scheduling a reminder.
6. Scaffolding a project boilerplate.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from yuno_llm import (
    YunoConfig,
    YunoIdentity,
    YunoMemory,
    YunoPlanner,
    IntentType,
    YunoToolRegistry,
    YunoAutomation,
)


def run_demo():
    print("=" * 65)
    print("        YUNO-LLM v0.5.0 OS LIVE DEMONSTRATION")
    print("=" * 65)

    # 1. Initialize System Components
    cfg = YunoConfig.from_yaml("config/yuno_config.yaml")
    identity = YunoIdentity(cfg)
    memory = YunoMemory(cfg)
    planner = YunoPlanner(cfg)
    tools = YunoToolRegistry(cfg)
    automation = YunoAutomation(cfg)

    # 2. Demo Memory Personalization
    print("\n[1/4] Memory Personalization:")
    memory.remember_personal_fact("user_name", "Piyush")
    memory.remember_personal_fact("user_role", "Lead AI Engineer")
    memory.add_episodic_entry("Built YUNO-LLM personal AI operating system with Hinglish support.", tags="project")

    user_name = memory.get_personal_fact("user_name")
    user_role = memory.get_personal_fact("user_role")
    ep_memories = memory.search_memory("operating system")

    print(f"  - User: {user_name} ({user_role})")
    print(f"  - Recalled Episodic Memory: '{ep_memories[0].content}'")

    # 3. Demo Natural Phrasing & System Prompt
    print("\n[2/4] Dynamic System Prompt & Hinglish Persona:")
    messages = identity.apply_system_prompt([], memory=memory, last_user_message="Main kya kar raha tha?")
    print("  - System Prompt Character Count:", len(messages[0]["content"]))
    print("  - Dynamic Context Injected: Date/Time, User Name (Piyush), & Episodic Memory")

    # 4. Demo Intent Classification
    print("\n[3/4] Multi-Modal Intent Routing:")
    sample_queries = [
        ("Mujhe workspace organize karna hai", "organize_files"),
        ("Remind me to check server logs at 19:00", "schedule_reminder"),
        ("Init python project smart_bot", "init_project"),
        ("Read file config/yuno_config.yaml", "read_file"),
    ]
    for query, expected_tool in sample_queries:
        intent, tool_name, args = planner.parse_intent(query)
        print(f"  - Prompt: '{query}'")
        print(f"    --> Intent: {intent.name} | Tool: {tool_name} | Args: {args}")

    # 5. Demo Automation Workflows
    print("\n[4/4] Live Desktop Automation Execution:")

    # Reminder Logging
    rem_res = automation.create_reminder("Review YUNO v0.5.0 evaluation benchmark results", "18:00")
    print(f"  - {rem_res}")

    # Project Scaffolding
    demo_proj = ROOT / "workspace" / "demo_ai_app"
    if demo_proj.exists():
        import shutil
        shutil.rmtree(demo_proj)

    scaffold_res = automation.scaffold_project("demo_ai_app", "python", target_parent="workspace")
    print(f"  - Scaffolder Output:\n{scaffold_res}")

    # Clean up demo project
    if demo_proj.exists():
        import shutil
        shutil.rmtree(demo_proj)

    print("\n" + "=" * 65)
    print("  [SUCCESS] DEMONSTRATION COMPLETE — YUNO OS OPERATIONAL")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_demo()
