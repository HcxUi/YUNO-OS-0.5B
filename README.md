# YUNO-LLM

> A research-grade, fully understood, open-source large language model — built and owned from the ground up.

---

## Mission

YUNO-LLM is not a wrapper. It is a research project to:

- Understand every component of a modern LLM (attention, normalization, embeddings, training)
- Modify the architecture deliberately, not blindly
- Build a training and evaluation pipeline we fully own
- Release versioned checkpoints with reproducible results

We start with the **Qwen3** architecture as our foundation and build YUNO-LLM on top of it.

---

## Project Structure

```
YUNO-LLM/
│
├── docs/                    # Theory + architecture documentation
│   ├── theory/              # Concept-by-concept LLM theory (11 documents)
│   └── architecture/        # Layer-by-layer architecture analysis
│
├── research/                # Papers, references, notes
├── datasets/                # Training and evaluation datasets
├── tokenizer/               # Tokenizer experiments and custom vocab
│
├── models/
│   ├── base/                # Downloaded base model weights
│   └── yuno/                # YUNO-LLM checkpoints
│
├── training/                # Training scripts (SFT, LoRA, full fine-tune)
├── inference/               # Generation scripts and API server
├── evaluation/              # Evaluation suite (reasoning, coding, math)
│
├── checkpoints/             # Training checkpoints
├── experiments/             # Experiment logs (JSON per run)
│
├── scripts/                 # Utility scripts (download, verify, convert)
├── tools/                   # Helper utilities
├── notebooks/               # Jupyter notebooks for exploration
├── tests/                   # Unit and integration tests
│
├── src/
│   └── yuno_llm/            # YUNO-LLM Python package
│       ├── config.py        # YunoConfig
│       ├── model.py         # YunoForCausalLM
│       ├── tokenizer.py     # YunoTokenizer
│       ├── generation.py    # Custom generation logic
│       └── identity.py      # System identity
│
├── config/
│   ├── yuno_config.yaml     # Master project configuration
│   └── training_config.yaml # Training hyperparameters
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourname/YUNO-LLM.git
cd YUNO-LLM
```

### 2. Create Python environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/macOS
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the base model
```bash
python scripts/download_model.py
```

### 5. Verify the installation
```bash
python scripts/verify_model.py
```

---

## Development Roadmap

| Phase | Status | Goal |
|-------|--------|------|
| Phase 0 | ✅ | Environment setup |
| Phase 1 | 🔄 | LLM theory documentation |
| Phase 2 | ⬜ | Base model download and verification |
| Phase 3 | ⬜ | Architecture analysis |
| Phase 4 | ⬜ | YUNO-LLM source package |
| Phase 5 | ⬜ | Fine-tuning pipeline (LoRA) |
| Phase 6 | ⬜ | Inference server |
| Phase 7 | ⬜ | Evaluation suite |
| Phase 8 | ⬜ | Release v1.0 |

---

## Versions

| Version | Description |
|---------|-------------|
| v0.1 | Architecture understood, model loads |
| v0.2 | First LoRA fine-tune on custom dataset |
| v0.5 | Evaluation suite, inference server |
| v1.0 | Full release with public weights |

---

## Base Model

YUNO-LLM is built on top of **Qwen3** (Alibaba Cloud).

- Model card: https://huggingface.co/Qwen/Qwen3-0.6B
- License: Apache 2.0

---

## Philosophy

> "We don't use components we don't understand. We understand first, then we modify."

Every change to YUNO-LLM must be:
1. Theoretically motivated
2. Documented in `docs/architecture/`
3. Measured against the evaluation suite

---

## License

Apache 2.0 — See `LICENSE` file.
