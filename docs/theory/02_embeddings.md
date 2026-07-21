# 02 — Embeddings

## Intuition

After tokenization, each token is an integer (e.g., `9707`). But a neural network can't do meaningful math on raw integers — the difference between token `9707` and token `9708` means nothing semantically.

**Embeddings** solve this by mapping each token to a **dense vector** in a high-dimensional space. Similar tokens end up near each other. Different tokens are far apart.

Think of it like a map: every word gets coordinates. "Paris" and "France" are close. "Paris" and "banana" are far apart.

---

## The Embedding Matrix

The embedding layer is simply a **lookup table** (a large matrix):

$$E \in \mathbb{R}^{|V| \times d_{\text{model}}}$$

Where:
- $|V|$ = vocabulary size (151,936 for Qwen3)
- $d_{\text{model}}$ = embedding dimension (1024 for Qwen3-0.6B)

For token index $t_i$, the embedding is the $t_i$-th row of $E$:

$$\mathbf{e}_i = E[t_i] \in \mathbb{R}^{d_{\text{model}}}$$

This is called a **row lookup** — it is mathematically equivalent to one-hot encoding followed by matrix multiplication, but much more efficient.

---

## Properties of Good Embeddings

Trained embeddings develop semantic geometry:

| Relationship | Vector Arithmetic |
|-------------|------------------|
| Synonyms | $\text{vec}(\text{"big"}) \approx \text{vec}(\text{"large"})$ |
| Analogies | $\text{vec}(\text{"king"}) - \text{vec}(\text{"man"}) + \text{vec}(\text{"woman"}) \approx \text{vec}(\text{"queen"})$ |
| Opposites | Far apart in embedding space |

---

## Mathematics

**Forward pass through embedding layer:**

Input token sequence: $\mathbf{t} = [t_1, t_2, \ldots, t_n]$

Embedding output:
$$\mathbf{X} = \begin{bmatrix} \mathbf{e}_1 \\ \mathbf{e}_2 \\ \vdots \\ \mathbf{e}_n \end{bmatrix} \in \mathbb{R}^{n \times d_{\text{model}}}$$

where $\mathbf{e}_i = E[t_i]$.

**Scaling (used in some models):**
$$\mathbf{e}_i = E[t_i] \cdot \sqrt{d_{\text{model}}}$$

This scaling stabilizes the variance of embeddings before they enter attention.

---

## Input Embeddings vs. Output Embeddings

In most decoder-only LLMs (including Qwen3), the **same embedding matrix** is used for:
1. **Input**: Look up token → vector
2. **Output (weight tying)**: Multiply final hidden state by $E^T$ to get logits

This is called **weight tying** and reduces parameter count significantly:

$$\text{Logits} = \mathbf{h}_n \cdot E^T \in \mathbb{R}^{|V|}$$

---

## Qwen3-0.6B Embedding Dimensions

| Parameter | Value |
|-----------|-------|
| Vocabulary size ($\|V\|$) | 151,936 |
| Embedding dimension ($d_{\text{model}}$) | 1,024 |
| Embedding matrix size | 151,936 × 1,024 = ~155M parameters |
| % of total model params | ~26% |

The embedding matrix is one of the largest components in small models.

---

## Code Example

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# Inspect the embedding layer
embed = model.model.embed_tokens
print("Embedding shape:", embed.weight.shape)  # [151936, 1024]

# Get embedding for a token
token_id = tokenizer.encode("hello", add_special_tokens=False)[0]
vec = embed.weight[token_id]
print("Token ID:", token_id)
print("Vector shape:", vec.shape)   # [1024]
print("First 5 values:", vec[:5])

# Check weight tying (input embed == output lm_head)
lm_head = model.lm_head
print("LM head shares weights:", embed.weight.data_ptr() == lm_head.weight.data_ptr())
```

---

## How This Applies to YUNO-LLM

- YUNO-LLM uses the Qwen3 embedding layer unchanged initially
- Possible future modification: **domain-specific token initialization** (initialize new token embeddings from the average of related tokens rather than random)
- The embedding layer is shared with the output head (weight tying)

**Key files:**
- `src/yuno_llm/model.py` — Access to `model.embed_tokens`
- `docs/architecture/embedding_analysis.md` — Full embedding analysis

---

## References

- Mikolov, Tomas, et al. "Efficient estimation of word representations in vector space." (2013) — Word2Vec
- Press & Wolf. "Using the Output Embedding to Improve Language Models." (2017) — Weight tying
- https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html — PyTorch Embedding docs
