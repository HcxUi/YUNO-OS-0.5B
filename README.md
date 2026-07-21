# YUNO-OS-0.5B-Instruct-

> **Personal AI Operating System — Private, Local, Offline-First, Natural Hinglish**

[![Model Card](https://img.shields.io/badge/Model%20Card-YUNO--OS--0.5B--Instruct-brightgreen.svg)](MODEL_CARD.md)
[![Release Notes](https://img.shields.io/badge/Release%20Notes-v0.5.0-orange.svg)](RELEASE_NOTES.md)

---


## 🌟 Overview & Mission

**YUNO-OS-0.5B-Instruct** is an intelligent, offline-first personal AI operating system designed for desktop automation, multi-modal reasoning (Vision & Voice), high-performance computing, and natural conversation in **English**, **Hindi**, and **Hinglish**.

Developed from the ground up by **[@hcxui](https://github.com/HcxUi)**, YUNO OS operates completely locally on personal workstation hardware while ensuring complete user privacy and total action safety through a 3-tier Human-In-The-Loop (HITL) permission architecture.

---

## 👑 Creator & Developer Attribution

> **"I was created and developed by @hcxui."**  
> **"मुझे @hcxui द्वारा बनाया और विकसित किया गया है। 😊"**

- **Developer & Architect:** [@hcxui](https://github.com/HcxUi)
- **Model Architecture:** `YunoForCausalLM` (`YUNO-OS-0.5B-Instruct`)
- **Package Container Format:** `YUNO_LLM_PROPRIETARY_V1`

---

## 🛡️ Key Features & Anti-Tamper Security

1. **🔒 Privacy-First & Offline-First**
   - Operates 100% locally on your workstation.
   - Conversation history, facts, and episodic embeddings never leave your device.
2. **🛡️ Anti-Tamper & Security Container**
   - Cryptographic **SHA-256 weight hash signature** embedded in binary metadata to prevent unauthorized modification.
   - **XOR 0x77 Anti-Decompile Cipher:** Model byte-stream protected against reverse engineering, editing, or unauthorized re-mastering.
3. **🤝 Human-In-The-Loop (HITL) Tool Safety**
   - State-modifying actions (`write_file`, `init_project`, `run_script`) **ALWAYS** require typed explicit user approval.
   - Read-only actions (`read_file`, `list_dir`, `search_files`) execute automatically.
4. **🗣️ Natural Hinglish Native**
   - Seamlessly handles mixed Hindi-English grammar, technical terminology, and natural daily phrasing.
5. **🧠 SQLite FTS5 Episodic Memory**
   - Zero-dependency, zero-cloud full-text searchable memory store for past discussions, project notes, and preferences.
6. **👁️ Vision & 🎙️ Voice Modules**
   - Native OCR, image analysis, desktop screenshot inspection, PDF text extraction, Text-To-Speech (TTS), and Speech-To-Text (STT).

---

## 📦 YUNO OS Release Package (`YUNO_OS_Release_Package`)

The repository includes a ready-to-run release package with both Web UI and CLI interfaces:

```text
YUNO_OS_Release_Package/
├── yuno-os-v0.5.0.yuno          # Official Anti-Tamper LLM Binary Container (2.38 GB)
├── README.md                    # Official Launch Release Documentation
├── yuno_os_launch_poster.png    # Cyberpunk Promotional Release Banner
├── app.py                       # Streamlit Glassmorphism Web Interface
├── cli.py                       # Interactive CLI Terminal Application
├── run_web_ui.bat               # One-click Launcher for Streamlit Web UI
├── run_cli.bat                  # One-click Launcher for Interactive CLI
├── extract_and_run.py           # Integrity unpacker & decryptor script
└── pack_yuno_bin.py             # Proprietary secure pack & signature utility
```

### 🚀 Launch Options

#### Option 1: Web Interface (Streamlit Glassmorphism UI)
Run the script or double-click `run_web_ui.bat`:
```bash
python -m streamlit run YUNO_OS_Release_Package/app.py
```
*Access Web UI at `http://localhost:8502`*

#### Option 2: Interactive CLI Terminal
Run the script or double-click `run_cli.bat`:
```bash
python YUNO_OS_Release_Package/cli.py
```

---

## 🏗️ System Architecture

```
                                  USER INTERFACE
                        (CLI / Web UI / Spoken / Screenshot)
                                       │
                                       ▼
                              YunoGenerator (Loop)
                                       │
           ┌──────────────────────────┼──────────────────────────┐
           ▼                          ▼                          ▼
     YunoPlanner                YunoIdentity                 YunoMemory
(Intent Classification)     (Dynamic Prompting)         (Short/Long/Episodic)
           │                          │                          │
           └──────────────────────────┼──────────────────────────┘
                                      │
                                      ▼
                              YunoToolRegistry
                              (15 HITL Tools)
       ┌──────────────┬───────────────┼───────────────┬──────────────┐
       ▼              ▼               ▼               ▼              ▼
   File I/O        Scripts         Vision           Voice        Automation
  (Read/Write)   (Sandboxed)   (OCR/Screenshot)   (TTS/STT)   (Sorter/Init)
```

---

## 🛠️ Complete Tool & Permission Matrix (15 Tools)

| Tool Name | Category | Permission Level | Description |
|---|---|---|---|
| `read_file` | File I/O | `AUTO` | Read contents of local text/code file. |
| `list_dir` | File I/O | `AUTO` | List files and directories. |
| `search_files` | File I/O | `AUTO` | Full-text search across workspace files. |
| `summarize_file` | File I/O | `AUTO` | Summarize large text files. |
| `analyze_image` | Vision | `AUTO` | Extract dimensions, format, & Pytesseract OCR text. |
| `analyze_screenshot` | Vision | `AUTO` | Capture screen & analyze visible text. |
| `extract_document_ocr` | Vision | `AUTO` | Extract text & OCR from PDF files. |
| `speak_text` | Voice | `AUTO` | Offline Text-To-Speech synthesis (`pyttsx3`). |
| `listen_speech` | Voice | `AUTO` | Microphone & audio file transcription. |
| `schedule_reminder` | Automation | `AUTO` | Log task reminders (`reminders.json`). |
| `organize_files` | Automation | `CONFIRM` | Sort loose directory files into subfolders (y/n prompt). |
| `write_file` | File I/O | `EXPLICIT` | Write or overwrite file (requires typed "yes"). |
| `create_note` | File I/O | `EXPLICIT` | Create Markdown notes. |
| `run_script` | Execution | `EXPLICIT` | Execute sandboxed Python script. |
| `init_project` | Automation | `EXPLICIT` | Scaffold project boilerplate (Python, Web, C++). |

---

## 💻 Quickstart & System Commands

### 1. Installation

```bash
git clone https://github.com/HcxUi/YUNO-OS-0.5B-Instruct-.git
cd YUNO-OS-0.5B-Instruct-
pip install -r requirements.txt
```

### 2. Verification & Live Execution

```bash
# Run End-To-End System Diagnostics Test Suite (10/10 Checks)
python tests/test_e2e_full_os.py

# Launch Main System CLI Session
python src/yuno_llm/generation.py

# Run Live Workflows Demo Script
python scripts/demo_os_cli.py
```

---

## 🎓 Training & Fine-Tuning Pipeline

YUNO-OS includes a complete LoRA SFT fine-tuning and evaluation workflow:

```bash
# 1. Generate clean Hinglish dataset (500 train, 50 eval)
python scripts/generate_dataset.py

# 2. Run LoRA SFT Fine-Tuning
python training/train_sft.py --config config/training_config.yaml --train datasets/train.jsonl --eval datasets/eval.jsonl

# 3. Run Benchmark Evaluations
python evaluation/run_evals.py --model Qwen/Qwen3-0.6B --n-samples 10 --benchmarks latency yuno_hinglish
```

---

## 📄 License

Distributed under the **MIT License**. Created by **[@hcxui](https://github.com/HcxUi)**.
