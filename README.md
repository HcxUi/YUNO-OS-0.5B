# YUNO-OS-0.5B-Instruct-

> **Personal AI Operating System — Private, Local, Offline-First, Natural Hinglish & Agentic Automation**

[![Release](https://img.shields.io/badge/Release-v0.5.0-blue.svg)](https://github.com/HcxUi/YUNO-OS-0.5B-Instruct-)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Transformers-4.40+-FFD21E.svg?logo=huggingface&logoColor=black)](https://huggingface.co/)

---

## 🌟 Overview & Mission

**YUNO-OS-0.5B-Instruct-** (YUNO-LLM v0.5.0) is an end-to-end, workstation-grade AI Operating System engineered from the ground up to run 100% locally and offline on personal desktop hardware.

It combines a fine-tuned **0.5B / 0.6B language backbone** optimized for natural trilingual interaction (English, Hindi, and natural **Hinglish**), SQLite FTS5 episodic memory, vision OCR document inspection, offline voice interaction, and multi-step desktop automation routines — while keeping the user in full control through a strict **Human-In-The-Loop (HITL)** security model.

---

## 🚀 Key Features

* 🔒 **100% Local & Privacy-First**: Operates completely offline. No external API calls, zero telemetry, and total user ownership over conversation history and personal facts.
* 🛡️ **Human-In-The-Loop (HITL) Safety Enforcement**: 3-tiered permission architecture (`AUTO`, `CONFIRM`, `EXPLICIT`). Modifying filesystem or script execution actions strictly require explicit user consent.
* 🗣️ **Native Hinglish Conversational AI**: Custom-tuned for mixed Hindi-English phrasing, technical vocabulary, and natural daily conversation.
* 🧠 **SQLite FTS5 Episodic & Personal Memory**: Zero-cloud long-term memory store supporting full-text search across past chats, project context, and user preferences.
* 👁️ **Native OCR & Vision Processing**: Pytesseract OCR, PyMuPDF PDF parsing, and real-time screen inspection.
* 🎙️ **Offline Voice Synthesis & STT**: Local Text-to-Speech (`pyttsx3`) and Speech-to-Text (`SpeechRecognition`).
* ⚡ **End-to-End Fine-Tuning & Model Export Pipeline**: Includes full dataset generation, LoRA SFT training, evaluation benchmarks, and single-file model export scripts.

---

## 🏗️ System Architecture

```
                                  USER INTERFACE
                          (CLI / Spoken / Screenshots)
                                       │
                                       ▼
                             YunoGenerator (Orchestrator)
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
      YunoPlanner                YunoIdentity                 YunoMemory
(Intent Classification)      (Dynamic Prompting)         (Short/Long/Episodic)
            │                          │                          │
            └──────────────────────────┼──────────────────────────┘
                                       │
                                       ▼
                               YunoToolRegistry
                                (15 HITL Tools)
        ┌──────────────┬───────────────┼───────────────┬──────────────┐
        ▼              ▼               ▼               ▼              ▼
    File I/O        Scripts         Vision           Voice        Automation
   (Read/Write)   (Sandboxed)    (OCR/Screenshot)   (TTS/STT)   (Sorter/Init)
```

---

## 📦 Core OS Modules

| Module | Source Location | Description |
| :--- | :--- | :--- |
| **Config Engine** | [`src/yuno_llm/config.py`](file:///f:/llm90/src/yuno_llm/config.py) | Master YAML configuration loader (`config/yuno_config.yaml`). |
| **Identity Engine** | [`src/yuno_llm/identity.py`](file:///f:/llm90/src/yuno_llm/identity.py) | Dynamic system prompt builder injecting time, persona, and memory context. |
| **Memory Engine** | [`src/yuno_llm/memory.py`](file:///f:/llm90/src/yuno_llm/memory.py) | Short-term turns, long-term user facts, project facts, & SQLite FTS5 episodic storage. |
| **Planner Engine** | [`src/yuno_llm/planner.py`](file:///f:/llm90/src/yuno_llm/planner.py) | High-speed intent classification (`CHAT`, `TOOL_CALL`, `MEMORY_QUERY`, `PLAN`). |
| **Tool Registry** | [`src/yuno_llm/tools.py`](file:///f:/llm90/src/yuno_llm/tools.py) | 15 native tools with 3-tier HITL safety matrix (`AUTO`, `CONFIRM`, `EXPLICIT`). |
| **Updater System** | [`src/yuno_llm/updater.py`](file:///f:/llm90/src/yuno_llm/updater.py) | Safe 6-step update pipeline (check → prompt → backup → apply → test → rollback). |
| **Vision Engine** | [`src/yuno_llm/vision.py`](file:///f:/llm90/src/yuno_llm/vision.py) | Image inspection, Pytesseract OCR, desktop screenshot analysis, PyMuPDF PDF OCR. |
| **Voice Engine** | [`src/yuno_llm/voice.py`](file:///f:/llm90/src/yuno_llm/voice.py) | Offline Text-To-Speech (`pyttsx3`) and Speech-To-Text (`SpeechRecognition`). |
| **Automation** | [`src/yuno_llm/automation.py`](file:///f:/llm90/src/yuno_llm/automation.py) | File sorting, task reminder scheduling, and project scaffolding. |
| **Generator Loop** | [`src/yuno_llm/generation.py`](file:///f:/llm90/src/yuno_llm/generation.py) | Core orchestration loop integrating streaming generation, memory, and tool execution. |

---

## 🛠️ Tool Safety & Permission Matrix (15 HITL Tools)

| Tool Name | Category | Permission Level | Description |
| :--- | :--- | :--- | :--- |
| `read_file` | File I/O | `AUTO` | Reads contents of local text or code files. |
| `list_dir` | File I/O | `AUTO` | Lists files and subdirectories. |
| `search_files` | File I/O | `AUTO` | Performs regex/string searches across workspace files. |
| `summarize_file` | File I/O | `AUTO` | Generates a condensed summary of a document. |
| `analyze_image` | Vision | `AUTO` | Extracts image metadata and performs OCR text extraction. |
| `analyze_screenshot` | Vision | `AUTO` | Captures primary screen and analyzes text content. |
| `extract_document_ocr` | Vision | `AUTO` | Extracts text and OCR from multi-page PDF documents. |
| `speak_text` | Voice | `AUTO` | Synthesizes spoken audio output via offline TTS engine. |
| `listen_speech` | Voice | `AUTO` | Transcribes microphone input or audio files locally. |
| `schedule_reminder` | Automation | `AUTO` | Logs a structured task reminder to `reminders.json`. |
| `organize_files` | Automation | `CONFIRM` | Sorts loose files into structured subfolders (y/n prompt). |
| `write_file` | File I/O | `EXPLICIT` | Writes or overwrites a file (requires explicit typed confirmation). |
| `create_note` | File I/O | `EXPLICIT` | Creates a new structured Markdown note file. |
| `run_script` | Execution | `EXPLICIT` | Executes a Python script inside an isolated sandbox. |
| `init_project` | Automation | `EXPLICIT` | Scaffolds standard project boilerplate (Python, Web, C++). |

---

## 💻 Quickstart & Installation

### Prerequisites
* **Python**: 3.11 or higher
* **Tesseract OCR** (Optional, for Vision module): System binary installed and added to `PATH`.

### Installation

```bash
# Clone the repository
git clone https://github.com/HcxUi/YUNO-OS-0.5B-Instruct-.git
cd YUNO-OS-0.5B-Instruct-

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies and package in editable mode
pip install -r requirements.txt
pip install -e .
```

### Running System Tests & Interactive CLI

```bash
# 1. Run full System End-to-End integration test suite (10/10 modules)
python tests/test_e2e_full_os.py

# 2. Run live demonstration CLI workflow
python scripts/demo_os_cli.py

# 3. Launch interactive YUNO OS CLI Session
python src/yuno_llm/generation.py
```

---

## 🎓 Training, Fine-Tuning & Model Export

YUNO OS comes with a complete dataset preparation, LoRA SFT fine-tuning, and export suite:

```bash
# Step 1: Generate 500-sample Hinglish instruction dataset
python scripts/generate_dataset.py

# Step 2: Run LoRA SFT Fine-Tuning
python training/train_sft.py \
  --config config/training_config.yaml \
  --train datasets/train.jsonl \
  --eval datasets/eval.jsonl

# Step 3: Run Evaluation Benchmarks (Latency, Hinglish accuracy, Perplexity)
python evaluation/run_evals.py \
  --model Qwen/Qwen3-0.6B \
  --benchmarks latency yuno_hinglish

# Step 4: Export Hybrid Model / Weights
python scripts/export_hybrid_model.py
```

---

## 📊 Verification & System Report

* **E2E Integration Test**: [`tests/test_e2e_full_os.py`](file:///f:/llm90/tests/test_e2e_full_os.py) — **PASSED (10/10 module checks)**.
* **Live Demo Test**: [`scripts/demo_os_cli.py`](file:///f:/llm90/scripts/demo_os_cli.py) — **PASSED (4/4 workflow scenarios)**.
* **System Release Report**: Detailed release metrics available in [`yuno_v0.5.0_final_report.md`](file:///f:/llm90/yuno_v0.5.0_final_report.md).

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for details.
