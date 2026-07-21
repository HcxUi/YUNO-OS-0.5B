# 03 — The Transformer Architecture

## Intuition

The **Transformer** (Vaswani et al., 2017) replaced recurrent neural networks (RNNs/LSTMs) as the dominant architecture for language modeling.

The key insight: instead of processing tokens sequentially (slow, forgets distant context), the Transformer processes **all tokens in parallel** and uses **attention** to let every token look at every other token simultaneously.

---

## Two Families

| Architecture | Variant | Use Case |
|---|---|---|
| Encoder-only | BERT, RoBERTa | Classification, embedding |
| Encoder-Decoder | T5, BART | Translation, summarization |
| **Decoder-only** | **GPT, Qwen3, LLaMA** | **Language generation** |

**YUNO-LLM is decoder-only.** We only care about the decoder stack.

---

## High-Level Flow

```
Input Text
    ↓
Tokenizer → [token IDs]
    ↓
Embedding Layer → [token vectors]  shape: [batch, seq_len, d_model]
    ↓
┌─────────────────────────────┐
│   Transformer Block × N     │  N = number of layers
│  ┌─────────────────────┐    │
│  │  RMSNorm            │    │
│  │  Self-Attention     │    │
│  │  Residual Add       │    │
│  │  RMSNorm            │    │
│  │  Feed-Forward       │    │
│  │  Residual Add       │    │
│  └─────────────────────┘    │
└─────────────────────────────┘
    ↓
Final RMSNorm
    ↓
LM Head (Linear) → [logits]  shape: [batch, seq_len, vocab_size]
    ↓
Softmax → probabilities
    ↓
Sample next token
```

---

## Qwen3-0.6B Architecture Specs

| Parameter | Value |
|---|---|
| `d_model` (hidden size) | 1024 |
| `n_layers` (num_hidden_layers) | 28 |
| `n_heads` (attention heads) | 16 |
| `n_kv_heads` (GQA groups) | 8 |
| `d_ffn` (intermediate size) | 3072 |
| `vocab_size` | 151,936 |
| `max_position_embeddings` | 32,768 |
| `head_dim` | 64 |
| Normalization | RMSNorm |
| Positional Encoding | RoPE |
| Activation | SiLU (SwiGLU FFN) |

---

## What Each Component Does

```
Token Embeddings    — Convert integer token IDs → dense vectors
        ↓
Positional Encoding — Tell the model WHERE each token is (via RoPE)
        ↓
Self-Attention      — Let each token attend to relevant other tokens
        ↓
Residual Add        — Add the input back in (prevents vanishing gradients)
        ↓
RMSNorm             — Normalize activations (training stability)
        ↓
Feed-Forward (FFN)  — Per-token nonlinear transformation (adds capacity)
        ↓
Residual Add        — Add again
        ↓
[Repeat × N layers]
        ↓
LM Head             — Project final hidden state → vocabulary logits
```

---

## Mathematics (Single Block)

Given input $\mathbf{X} \in \mathbb{R}^{n \times d}$:

**Attention sub-layer:**
$$\mathbf{X}' = \mathbf{X} + \text{Attention}(\text{RMSNorm}(\mathbf{X}))$$

**FFN sub-layer:**
$$\mathbf{X}'' = \mathbf{X}' + \text{FFN}(\text{RMSNorm}(\mathbf{X}'))$$

This **pre-norm** pattern (normalize before the sub-layer, add residual after) is what Qwen3 and most modern LLMs use.

---

## Code Example

```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

# Inspect the full architecture
print(model)

# Count layers
print("Number of decoder layers:", len(model.model.layers))  # 28

# Inspect one block
block = model.model.layers[0]
print("Block type:", type(block))
print("Block components:", list(block._modules.keys()))
# ['self_attn', 'mlp', 'input_layernorm', 'post_attention_layernorm']

# Count parameters
total = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total:,}")  # ~596M
```

---

## How This Applies to YUNO-LLM

- YUNO-LLM uses 28 decoder blocks identical in structure to Qwen3-0.6B
- We study each block before modifying it
- Future work: experimenting with different numbers of layers, head configurations, or FFN sizes

**Key files:**
- `src/yuno_llm/model.py` — YunoForCausalLM wraps this architecture
- `docs/architecture/Architecture.md` — Layer-by-layer breakdown
- `docs/architecture/decoder_analysis.md` — Deep dive into decoder block

---

## References

- Vaswani, Ashish, et al. "Attention is all you need." NeurIPS 2017 — Original Transformer paper
- https://arxiv.org/abs/2309.00071 — Qwen technical report
- https://jalammar.github.io/illustrated-transformer/ — Visual guide to Transformers
