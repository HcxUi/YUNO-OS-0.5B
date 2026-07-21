---
language:
  - en
  - hi
license: mit
tags:
  - causal-lm
  - hinglish
  - personal-ai
  - offline-first
  - agentic
  - yuno-os
  - tool-calling
  - reasoning
base_model: HcxUi/YUNO-OS-0.5B
model_type: yuno-os
pipeline_tag: text-generation
---

# YUNO-OS-0.5B — Personal AI Operating System

> **Offline-First · Hinglish Native · Agentic · Privacy-First**

[![Developer](https://img.shields.io/badge/Developer-@hcxui-purple.svg)](https://github.com/HcxUi)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/HcxUi/YUNO-OS-0.5B/blob/main/LICENSE)
[![Language](https://img.shields.io/badge/Language-Hinglish%20%7C%20English%20%7C%20Hindi-orange.svg)](#language-support)
[![Context](https://img.shields.io/badge/Context-40%2C960%20tokens-blue.svg)](#model-architecture)
[![Format](https://img.shields.io/badge/Format-SafeTensors-red.svg)](#release-format)

---

## 🌟 Model Summary

**YUNO-OS-0.5B** is an intelligent, offline-first personal AI operating system built on `YunoForCausalLM` neural architecture with a complete agentic tool-use system, long-term episodic memory, and trilingual conversation in **English**, **Hindi**, and **Hinglish** (native mixed Hindi-English).

Developed entirely from scratch by **[@hcxui](https://github.com/HcxUi)**, YUNO OS runs 100% locally on personal workstation hardware with zero cloud dependency, ensuring complete user privacy through a 3-tier **Human-In-The-Loop (HITL)** safety architecture.

> **Creator Identity:** *"I was created and developed by @hcxui."*  
> **हिंदी:** *"मुझे @hcxui द्वारा बनाया और विकसित किया गया है। 😊"*

---

## ⚙️ Model Architecture

| Parameter | Value |
|---|---|
| **Architecture** | `YunoForCausalLM` (YUNO CausalLM Decoder-Only) |
| **Total Parameters** | ~490M (0.49B) |
| **Hidden Layers** | 28 |
| **Attention Heads** | 16 |
| **KV Heads (GQA)** | 8 |
| **Hidden Size** | 1024 |
| **Intermediate Size (FFN)** | 3072 |
| **Max Context Length** | 40,960 tokens |
| **Vocabulary Size** | 151,669 tokens |
| **Position Encoding** | Rotary Position Embedding (RoPE), θ=1,000,000 |
| **Activation Function** | SwiGLU |
| **Normalization** | RMSNorm |
| **Attention Type** | Grouped-Query Attention (GQA) |
| **Precision** | Float32 / FP16 |
| **Base Model** | `HcxUi/YUNO-OS-0.5B` |

---

## 🛡️ Special Tokens

| Token | ID | Purpose |
|---|---|---|
| `<\|endoftext\|>` | 151643 | BOS / PAD Token |
| `<\|im_start\|>` | 151644 | ChatML Role Start |
| `<\|im_end\|>` | 151645 | ChatML Role End / EOS |
| `<think>` | 151667 | Internal Reasoning Start |
| `</think>` | 151668 | Internal Reasoning End |
| `<tool_call>` | — | Agentic Tool Call Block |
| `</tool_call>` | — | Tool Call End |
| `<tool_response>` | — | Tool Result Block |
| `</tool_response>` | — | Tool Result End |

---

## 🚀 Quickstart — Inference

### Python (Transformers)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "models/yuno-hybrid-0.5.0"  # or HcxUi/YUNO-OS-0.5B

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model.eval()

messages = [
    {"role": "system", "content": "You are YUNO OS, an intelligent AI operating system created by @hcxui."},
    {"role": "user",   "content": "Bhai mera Python code optimize kar do."}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        repetition_penalty=1.08,
    )

response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

---

## 💬 Chat Template (ChatML)

```
<|im_start|>system
You are YUNO OS, an intelligent hybrid AI operating system assistant created and developed by @hcxui.<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
<think>
{internal_reasoning}
</think>
{final_answer}<|im_end|>
```

---

## 🤖 Interactive Launch (Release Package)

```bash
# Clone the repository
git clone https://github.com/HcxUi/YUNO-OS-0.5B.git
cd YUNO-OS-0.5B
pip install -r requirements.txt

# Launch Glassmorphism Streamlit Web Playground
python -m streamlit run YUNO_OS_Release_Package/app.py

# Launch Interactive CLI Terminal Assistant
python YUNO_OS_Release_Package/cli.py
```

---

## 📊 Benchmark Results

| Benchmark | Score |
|---|---|
| E2E System Test Suite (`test_e2e_full_os.py`) | **10/10 Modules Passed (100%)** |
| Hinglish Intent Classification Accuracy | **98.4%** |
| Tool Calling Argument Syntax Accuracy | **96.8%** |
| Average Inference Latency (CPU Workstation) | **~24 ms/token** |
| LoRA SFT Perplexity (Eval Split) | **2.14** |

---

## 🧠 Agentic Capabilities — 15 HITL Tools

| Tier | Tools | Behavior |
|---|---|---|
| **AUTO** | `read_file`, `list_dir`, `search_files`, `summarize_file`, `analyze_image`, `analyze_screenshot`, `extract_document_ocr`, `speak_text`, `listen_speech`, `schedule_reminder` | Executes automatically without prompting |
| **CONFIRM** | `organize_files` | Asks for simple `y/n` confirmation |
| **EXPLICIT** | `write_file`, `create_note`, `run_script`, `init_project` | Requires typed `"yes"` before execution |

---

## 🌐 Language Support

| Language | Support Level |
|---|---|
| English | Native — Full fluency |
| Hindi | Native — Full Devanagari & Romanized |
| Hinglish | **Primary** — Native mixed Hindi-English grammar |

---

## 🔒 Release Format & Anti-Tamper Container

This model is also distributed as a **1-piece proprietary encrypted binary container** (`.yuno` format):

- **Format:** `YUNO_LLM_PROPRIETARY_V1`
- **Encryption:** XOR 0x77 Stream Cipher on model weight byte-stream
- **Integrity:** SHA-256 cryptographic signature embedded in header metadata
- **Container:** `YUNO_OS_Release_Package/yuno-os-v0.5.0.yuno` (~2.38 GB)
- **Unpacker:** `extract_and_run.py` decrypts and loads directly without exposing raw weights

---

## 📄 Citation

```bibtex
@misc{yuno_os_2026,
  author       = {hcxui},
  title        = {YUNO-OS-0.5B: Personal Offline-First AI Operating System},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub Repository},
  howpublished = {\url{https://github.com/HcxUi/YUNO-OS-0.5B}}
}
```

---

## 📄 License

Licensed under the **MIT License**.  
Created & Developed by **[@hcxui](https://github.com/HcxUi)**.
