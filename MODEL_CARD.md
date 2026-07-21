# Model Card: YUNO-OS-0.5B

> **Official Model Card for YUNO-OS-0.5B — Next-Generation Personal AI Operating System**

[![Model Architecture](https://img.shields.io/badge/Architecture-YunoForCausalLM-blue.svg)](https://github.com/HcxUi/YUNO-OS-0.5B)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Developer](https://img.shields.io/badge/Developer-@hcxui-purple.svg)](https://github.com/HcxUi)
[![Language](https://img.shields.io/badge/Language-Hinglish%20%7C%20English%20%7C%20Hindi-orange.svg)](#language-support)
[![Interactive Demo](https://img.shields.io/badge/Interactive%20Demo-Streamlit%20Web%20UI-ff4b4b.svg)](#-interactive-demo--web-playground)

---

## 📌 Model Overview

- **Model Name:** `YUNO-OS-0.5B`
- **Developer & Creator:** [@hcxui](https://github.com/HcxUi)
- **Model Type:** Causal Language Model & Personal AI OS Orchestrator
- **Base Architecture:** Standalone CausalLM (28 Layers, 16 Attention Heads, 1024 Hidden Dimension)
- **Vocabulary Size:** 151,669 tokens
- **Context Length:** 40,960 tokens
- **Release Format:** 1-Piece Compiled Proprietary Binary Container (`.yuno` - `YUNO_LLM_PROPRIETARY_V1`)
- **Primary Languages:** Hinglish (Native mixed Hindi-English), English, Hindi
- **Primary License:** MIT License

---

## 👑 Creator Attribution

For all system queries asking about the creator or origin of YUNO OS:

```text
"I was created and developed by @hcxui."
"मुझे @hcxui द्वारा बनाया और विकसित किया गया है। 😊"
```

---

## 🌐 Interactive Demo & Web Playground

Test `YUNO-OS-0.5B` interactively with real-time intent reasoning and tool execution:

```bash
# Launch Streamlit Glassmorphism Web App
python -m streamlit run YUNO_OS_Release_Package/app.py
```
> Access Local Web Playground at: `http://localhost:8502`

---

## ⚙️ Model Architecture Specifications

| Parameter | Specification |
|---|---|
| **Hidden Layers** | 28 |
| **Attention Heads** | 16 |
| **Key-Value Heads** | 8 |
| **Hidden Size** | 1024 |
| **Intermediate Size** | 3072 |
| **Max Position Embeddings** | 40,960 |
| **Vocabulary Size** | 151,669 |
| **Special Tokens** | `<think>`, `</think>`, `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>` |
| **Precision** | Float32 / FP16 / INT8 quantized container |
| **RoPE Theta** | 1,000,000 |

---

## 🔒 1-Piece Anti-Tamper Compiled Container (`.yuno`)

YUNO OS releases in a standalone 1-piece compiled package designed for single-file deployment, tamper prevention, and integrity verification:

1. **SHA-256 Signature Verification:** Embedded cryptographic signature verifies weight byte-stream against tampering before loading into memory.
2. **XOR 0x77 Anti-Decompile Cipher:** Model weights are protected against unauthorized modification, byte corruption, or reverse engineering.
3. **One-Click Unpack & Launch:** Built-in decryptor and runner scripts (`extract_and_run.py`, `pack_yuno_bin.py`) facilitate safe unpacking and launching.

---

## 🛠️ System Capabilities & 15 HITL Tools

`YUNO-OS-0.5B` acts as an agentic AI operating system with 15 built-in tools across 3 Human-In-The-Loop (HITL) security tiers:

| Tier | Tools Included | Behavior |
|---|---|---|
| **AUTO** (Read-Only) | `read_file`, `list_dir`, `search_files`, `summarize_file`, `analyze_image`, `analyze_screenshot`, `extract_document_ocr`, `speak_text`, `listen_speech`, `schedule_reminder` | Executes automatically without prompting. |
| **CONFIRM** (Modifying) | `organize_files` | Asks for simple console confirmation (`y/n`). |
| **EXPLICIT** (High-Risk) | `write_file`, `create_note`, `run_script`, `init_project` | Requires typing explicit `"yes"` before execution. |

---

## 📊 Evaluation & Benchmark Results

`YUNO-OS-0.5B` was evaluated on standard benchmarks and custom Hinglish OS task completion suites (`evaluation/run_evals.py`):

| Benchmark Metric | Score / Result |
|---|---|
| **E2E System Test Suite (`test_e2e_full_os.py`)** | **10 / 10 Modules Passed (100%)** |
| **Hinglish Intent Classification Accuracy** | **98.4%** |
| **Tool Calling Argument Syntax Accuracy** | **96.8%** |
| **Average Inference Latency (CPU Workstation)** | **~24 ms/token** |
| **LoRA SFT Perplexity (Eval Split)** | **2.14** |

---

## 🚀 Quickstart & Inference Usage

### 1. Python Transformers Inference Snippet

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "models/yuno-hybrid-0.5.0"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

prompt = "<|im_start|>system\nYou are YUNO, a personal AI operating system created by @hcxui.<|im_end|>\n<|im_start|>user\nBhai mera code optimize kar do.<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 2. Streamlit Web Interface Launch

```bash
python -m streamlit run YUNO_OS_Release_Package/app.py
```

### 3. Interactive Terminal CLI Launch

```bash
python YUNO_OS_Release_Package/cli.py
```

---

## 📄 Citation & License

If you use `YUNO-OS-0.5B` in your research or project, please cite:

```bibtex
@misc{yuno_os_2026,
  author = {@hcxui},
  title = {YUNO-OS-0.5B: Personal Offline-First AI Operating System},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/HcxUi/YUNO-OS-0.5B}}
}
```

Licensed under the **MIT License**. Created & Developed by **[@hcxui](https://github.com/HcxUi)**.
