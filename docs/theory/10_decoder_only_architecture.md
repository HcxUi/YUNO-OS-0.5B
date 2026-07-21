# 10 — Decoder-Only Architecture

## Intuition

The **decoder-only** architecture (also called **causal language model**) is the design used by GPT, Qwen3, LLaMA, Mistral, Gemma, and YUNO-LLM.

It has one job: **predict the next token** given all previous tokens.

During training, every position predicts its successor. At inference, we generate one token at a time, appending each generated token to the input and repeating.

---

## Encoder vs Decoder vs Both

| Architecture | Sees context | Use case |
|---|---|---|
| Encoder-only (BERT) | Both left and right (bidirectional) | Classification, embeddings |
| Decoder-only (GPT, Qwen3) | Left only (causal) | Text generation |
| Encoder-Decoder (T5) | Encoder: both; Decoder: left + encoder | Translation, summarization |

**Why decoder-only for LLMs?**
- Simpler: single stack of layers
- Scales better: more capacity per parameter for generation
- Unified: both prompts and responses processed the same way

---

## Causal Self-Attention

The key constraint of decoder-only: each token can **only attend to itself and tokens before it** (causal masking).

```
Sequence: [A, B, C, D, E]

A attends to: [A]
B attends to: [A, B]
C attends to: [A, B, C]
D attends to: [A, B, C, D]
E attends to: [A, B, C, D, E]
```

This ensures that when training on a sequence, computing the prediction for token $t$ doesn't "cheat" by looking at future tokens $t+1, t+2, \ldots$

---

## Full Architecture (Qwen3 / YUNO-LLM)

```
┌──────────────────────────────────────────────────────┐
│                  YUNO-LLM / Qwen3-0.6B               │
├──────────────────────────────────────────────────────┤
│                                                      │
│  Input: token IDs [t₁, t₂, ..., tₙ]                 │
│            ↓                                         │
│  Embedding Layer (embed_tokens)                      │
│    - Shape: [n] → [n, 1024]                          │
│    - Converts token IDs to dense vectors              │
│            ↓                                         │
│  ┌────────────────────────────────────────────────┐  │
│  │  Decoder Block × 28                           │  │
│  │  ┌──────────────────────────────────────────┐ │  │
│  │  │  input_layernorm (RMSNorm)               │ │  │
│  │  │  self_attn (GQA + RoPE + Causal Mask)   │ │  │
│  │  │  Residual Add                            │ │  │
│  │  │  post_attention_layernorm (RMSNorm)      │ │  │
│  │  │  mlp (SwiGLU FFN)                        │ │  │
│  │  │  Residual Add                            │ │  │
│  │  └──────────────────────────────────────────┘ │  │
│  └────────────────────────────────────────────────┘  │
│            ↓                                         │
│  Final RMSNorm (norm)                                │
│            ↓                                         │
│  LM Head (lm_head) — Linear, no bias                 │
│    - Shape: [n, 1024] → [n, 151936]                  │
│            ↓                                         │
│  Output: logits [n, 151936]                          │
│          → softmax → probabilities                   │
│          → sample/argmax → next token                │
└──────────────────────────────────────────────────────┘
```

---

## Mathematics

**Training objective — Cross-Entropy Loss:**

Given sequence $\mathbf{t} = [t_1, t_2, \ldots, t_n]$, the model predicts:

$$P(t_i | t_1, \ldots, t_{i-1}; \theta)$$

Training minimizes the **negative log-likelihood**:

$$\mathcal{L} = -\frac{1}{n} \sum_{i=1}^{n} \log P(t_i | t_1, \ldots, t_{i-1}; \theta)$$

This is equivalent to minimizing cross-entropy between the predicted distribution and the one-hot true token.

---

## Training vs. Inference Mode

| Mode | How tokens are fed | Speed |
|------|-------------------|-------|
| **Training** | Full sequence at once (teacher forcing) | Fast (parallel) |
| **Inference** | One token at a time (autoregressive) | Slower (sequential) |

**Teacher forcing**: during training, the true tokens from the dataset are fed as inputs at every step, not the model's own predictions. This avoids compounding errors during training and enables full parallelism.

---

## Code Example

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# --- Training mode: compute loss ---
text = "The sky is blue because of Rayleigh scattering."
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs.input_ids

with torch.no_grad():
    outputs = model(input_ids=input_ids, labels=input_ids)
    print("Loss:", outputs.loss.item())      # Cross-entropy loss
    print("Logits shape:", outputs.logits.shape)  # [1, n_tokens, 151936]

# --- Inference mode: generate ---
prompt = "The capital of Japan is"
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    generated = model.generate(**inputs, max_new_tokens=10)
print(tokenizer.decode(generated[0]))

# --- Inspect architecture ---
print("\nModel components:")
print("  Embedding:   ", model.model.embed_tokens)
print("  Layers:       28 decoder blocks")
print("  Final norm:  ", type(model.model.norm).__name__)
print("  LM head:     ", model.lm_head)
```

---

## How This Applies to YUNO-LLM

- YUNO-LLM is a decoder-only model — it generates text autoregressively
- During fine-tuning, we use teacher forcing (standard SFT)
- The output head (`lm_head`) is shared with the embedding layer (weight tying)
- The causal mask is applied automatically by the HuggingFace attention implementation

**Key files:**
- `src/yuno_llm/model.py` — `YunoForCausalLM` wraps the decoder stack
- `training/train_sft.py` — Teacher forcing with cross-entropy loss
- `inference/generate.py` — Autoregressive token generation

---

## References

- Radford et al. "Language Models are Unsupervised Multitask Learners." 2019 — GPT-2, decoder-only for generation
- Brown et al. "Language Models are Few-Shot Learners." NeurIPS 2020 — GPT-3 scaling
- https://arxiv.org/abs/2309.00071 — Qwen technical report
