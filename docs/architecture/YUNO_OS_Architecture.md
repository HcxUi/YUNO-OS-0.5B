# YUNO Operating System Architecture

This document describes the design and specification of the **YUNO-LLM Vision** system architecture. The system acts as a local-first Personal AI Operating System, integrating raw LLM weights with external structured memory, safe tool environments, and structural planning loops.

---

## 1. Core Component Breakdown

### 1.1 YunoMemory
Memory is split into three scopes to provide context without overloading the token window:
1. **Short-Term Conversation Memory:** A sliding window or summary context of the current chat turn history.
2. **Long-Term Personal Memory:** A local JSON flat-file storage (`datasets/long_term_memory.json`) storing persistent key-value profiles about the user (e.g., user name, recurring tasks, preferences).
3. **Project Memory:** Tracks files, file-types, and recent codebase modifications in the active workspace.

### 1.2 YunoTools & Safe Executor
Tools allow YUNO to interact with the environment. To ensure predictability, the execution is divided by risk level:
- **Read-Only / Side-Effect Free (Auto-Approve):** Includes reading workspace files, searching local documentation, or listing directory files.
- **State-Modifying (Explicit Permission Required):** Includes writing/modifying files, running shell commands, executing scripts, or making state-modifying API calls. The `SafeToolExecutor` prompts the user in the CLI before running these actions.

### 1.3 YunoPlanner & Reasoning Engine
Interprets prompts using a Chain-of-Thought (CoT) format. 
1. **Analyze:** Parse intent to identify if tools are required.
2. **Draft Plan:** Break complex tasks into steps.
3. **Execute:** Execute read tools to gather info, request permissions for write tools, and track progress.
4. **Self-Check:** Compare outputs against the system prompt rules (e.g., verifying Hinglish code translation comments) and return the final answer.

---

## 2. Dynamic Execution Flow

```
   User Input
       │
       ▼
   YunoPlanner ──[Query]──> YunoMemory (Retrieve context / user facts)
       │
       ▼
   Assemble System Prompt (Identity + Context + Memory)
       │
       ▼
   YUNO-LLM (Forward Pass)
       │
       ├─> [If Plan Requires Tool] ──> YunoTools Registry
       │                                     │
       │                                     ▼
       │                            Requires write/execute?
       │                            ├─> Yes ──> HITL Permission Dialog (Console)
       │                            └─> No ───> SafeToolExecutor Run
       │                                              │
       │                                              ▼
       │                                     Append Output to Context
       │                                              │
       │                                              ▼
       │                                        Loop Back to LLM
       │
       └─> [If Text Generation Complete] ──> Self-Checking/Validation
                                                   │
                                                   ▼
                                              User Output
```

---

## 3. Hinglish & Language Alignment

To facilitate Hinglish (mixed Hindi/English) conversation, the prompt system uses semantic triggers. The system prompt instructs the model to balance vocabulary dynamically:
- Respond in Hinglish naturally if the user starts with mixed Hindi-English phrasing.
- Fallback to clean English or clean Hindi if the user adopts single-language queries.
- Retain technical terminology (e.g., "compiler", "VRAM", "residual connection") in English script to avoid forced translation.
