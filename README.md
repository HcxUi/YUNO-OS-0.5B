# YUNO-LLM Vision (Personal AI OS)

> A personal AI operating system whose core is YUNO-LLM, designed to communicate naturally in Hinglish, reason through complex tasks, use tools, remember important information, and act only with the user's permission.

---

## Architecture Design

```
                 USER
                   │
          Voice / Text / Image
                   │
                   ▼
             YUNO Interface
                   │
                   ▼
         YUNO Reasoning Engine
                   │
         ┌─────────┼─────────┐
         │         │         │
         ▼         ▼         ▼
     YunoMemory YunoTools YunoPlanner
         │         │         │
         └─────────┼─────────┘
                   │
                   ▼
               YUNO LLM
```

The LLM is the brain, while memory, planning, and tools provide capabilities that neural network weights alone cannot.

---

## Core Capabilities

### 1. Human-like Conversation
- **Natural Hinglish & Language Switching:** Seamlessly switch between English, Hindi, and mixed Hinglish.
- **Context & Emotion Aware:** Responses adjust to the conversational context and user state.
- **Long-context retention:** Track conversational state across multiple turns.

### 2. Advanced Reasoning & Planning
- **Step-by-Step Planning:** Break down prompts and execute plans logically.
- **Self-Checking:** Reflects on its answers before outputting.
- **Uncertainty & Consistency:** Direct and honest; explicitly states when it does not know the answer.

### 3. Memory Stack
- **Short-Term Conversation Memory:** Retain current context.
- **Long-Term Personal Memory:** Remember user facts (e.g., name, preferences) locally in a JSON database.
- **Project Memory:** Tracks files, folders, and workspace code files.
- **User-Controlled:** Fully managed; you can delete or update stored memories at any time.

### 4. Safe Tool Execution (HITL)
- **User Permission-First:** Actions affecting the local system or network (write, delete, execute, API calls) require explicit confirmation.
- **Supported Tools:** File summarization, code generation, document search, and workflow automation.

---

## Project Structure

```
YUNO-LLM/
│
├── docs/                    # Theory, architecture & OS documentation
│   ├── theory/              # Concept-by-concept LLM theory
│   ├── architecture/        # OS & model layer-by-layer details
│   └── YUNO_OS_Architecture.md
│
├── datasets/                # Local data & memory JSONs
├── config/
│   ├── yuno_config.yaml     # OS & model configurations
│   └── training_config.yaml # LoRA hyperparameters
│
├── src/
│   └── yuno_llm/            # YUNO-LLM OS Core package
│       ├── config.py        # Config Loader
│       ├── identity.py      # Hinglish & HITL prompt identity
│       ├── model.py         # CausalLM base loader
│       ├── tokenizer.py     # Special token handler
│       ├── memory.py        # Short, Long & Project memory stores
│       ├── tools.py         # HITL Tool registry and SafeExecutor
│       ├── planner.py       # Plan parsing & reasoning execution
│       └── generation.py    # Main generator orchestrator
│
└── inference/
    └── generate.py          # Interactive chat REPL
```

---

## Setup

### 1. Create Python environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/macOS
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download base weights & verify
```bash
python scripts/download_model.py
python scripts/verify_model.py --local models/base/Qwen--Qwen3-0.6B
```

### 4. Start YUNO CLI OS
```bash
python inference/generate.py --model models/base/Qwen--Qwen3-0.6B
```

---

## OS Philosophy & Control

> **User Control Principle:** YUNO is powerful but completely controlled by the user. It explains what it is about to do, requests approval for state-modifying actions, and lets you revoke permissions at any point.

---

## Roadmap

- **Version 0.1:** Open-source foundation, Custom OS config and system identity prompt.
- **Version 0.5:** Fine-tune for Hinglish/instruction following, introduce advanced local memory.
- **Version 1.0:** Voice/Vision, Desktop integrations, Tool-use local APIs.
- **Version 2.0:** Multi-agent collaboration, planning & reflection OS.
