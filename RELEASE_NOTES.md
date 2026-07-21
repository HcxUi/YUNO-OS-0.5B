# YUNO OS 0.5B Instruct — v0.5.0 Release Notes

> **Official Release Notes for YUNO-OS-0.5B-Instruct v0.5.0**  
> **Developer:** [@hcxui](https://github.com/HcxUi) | **Release Date:** July 2026

---

## 🎉 What's New in v0.5.0

`YUNO-OS-0.5B-Instruct` v0.5.0 marks the full system release of the personal, local, offline-first AI operating system. Built for seamless desktop automation, multi-modal interaction, long-term personal memory, and trilingual conversation in **English**, **Hindi**, and **Hinglish**.

---

## 🌟 Key Highlights

### 1. 1-Piece Compiled Proprietary Container (`.yuno`)
- Package format: `YUNO_LLM_PROPRIETARY_V1`
- Cryptographic **SHA-256 weight hash signature** for anti-tamper security.
- **XOR 0x77 Anti-Decompile Cipher** protecting weight integrity against reverse engineering or modification.
- Single-file binary distribution package with one-click unpackers (`extract_and_run.py`, `pack_yuno_bin.py`).

### 2. Streamlit Glassmorphism Web UI & Interactive CLI
- Modern glassmorphism web application (`YUNO_OS_Release_Package/app.py`).
- One-click launcher scripts (`run_web_ui.bat` & `run_cli.bat`).
- Real-time response streaming, visual tool status badges, system prompt customization, and dark theme UI.

### 3. Complete 15-Tool Agentic Registry with 3-Tier HITL Safety
- **`AUTO` Tier (10 Tools):** Read files, search workspace, summarize files, list directories, analyze images, analyze screenshots, extract PDF OCR, TTS voice output, STT microphone listening, task scheduling.
- **`CONFIRM` Tier (1 Tool):** Directory file organizer (`y/n` prompt).
- **`EXPLICIT` Tier (4 Tools):** `write_file`, `create_note`, `run_script`, `init_project` (requires typed explicit `"yes"` approval).

### 4. Zero-Cloud SQLite FTS5 Episodic Memory
- Full-text search over conversation logs, facts, and workspace notes.
- Dynamic prompt builder (`YunoIdentity`) integrating system time, persona, short-term turns, and long-term memory.

### 5. Automated E2E Test Suite & LoRA SFT Pipeline
- Automated end-to-end test suite (`tests/test_e2e_full_os.py`) passing **10/10 module checks**.
- SFT fine-tuning pipeline (`training/train_sft.py`) and benchmark suite (`evaluation/run_evals.py`).

---

## 💻 Installation & Quickstart

```bash
# Clone the repository
git clone https://github.com/HcxUi/YUNO-OS-0.5B-Instruct-.git
cd YUNO-OS-0.5B-Instruct-

# Install requirements
pip install -r requirements.txt

# Run Web UI
python -m streamlit run YUNO_OS_Release_Package/app.py

# Run CLI
python YUNO_OS_Release_Package/cli.py
```

---

## 👑 Attribution

Created and developed by **[@hcxui](https://github.com/HcxUi)**.  
Licensed under the **MIT License**.
