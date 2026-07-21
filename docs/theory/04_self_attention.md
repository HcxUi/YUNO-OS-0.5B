# 04 — Self-Attention

## Intuition

**Self-attention** allows each token in a sequence to "look at" all other tokens and decide how much each one matters to its own representation.

Consider the sentence: *"The animal didn't cross the street because **it** was too tired."*

What does "it" refer to? "Animal" or "street"? A human knows it's "animal." Self-attention lets the model figure this out by computing a weighted relationship between "it" and every other word.

---

## Queries, Keys, Values

Self-attention has three roles for each token:

| Role | Question |
|------|---------|
| **Query (Q)** | "What am I looking for?" |
| **Key (K)** | "What information do I have to offer?" |
| **Value (V)** | "What information do I actually pass along?" |

Every token projects itself into Q, K, and V spaces using learned linear layers:

$$\mathbf{Q} = \mathbf{X} W_Q, \quad \mathbf{K} = \mathbf{X} W_K, \quad \mathbf{V} = \mathbf{X} W_V$$

Where:
- $\mathbf{X} \in \mathbb{R}^{n \times d_{\text{model}}}$ — input sequence
- $W_Q, W_K \in \mathbb{R}^{d_{\text{model}} \times d_k}$ — projection weights
- $W_V \in \mathbb{R}^{d_{\text{model}} \times d_v}$ — value projection

---

## Mathematics

**Scaled Dot-Product Attention:**

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

Step-by-step:

1. **Dot product:** $\mathbf{Q}\mathbf{K}^T \in \mathbb{R}^{n \times n}$ — similarity scores between all token pairs
2. **Scale:** Divide by $\sqrt{d_k}$ to prevent extremely large values (which would push softmax into near-zero gradient regions)
3. **Mask** (causal): Set future positions to $-\infty$ so each token only attends to past/current tokens
4. **Softmax:** Convert scores to probabilities (each row sums to 1)
5. **Weighted sum:** Multiply by $\mathbf{V}$ to get attended output

---

## Causal Masking

For language generation, we must prevent a token at position $i$ from attending to tokens at position $j > i$ (future tokens).

This is enforced by an **attention mask** — a lower-triangular matrix added before softmax:

$$M_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

After adding $M$ and applying softmax, positions with $-\infty$ become $0$ — effectively ignored.

```
Token sequence: [A, B, C, D]

Attention mask (lower triangular):
A: [1, 0, 0, 0]  ← A can only see A
B: [1, 1, 0, 0]  ← B can see A, B
C: [1, 1, 1, 0]  ← C can see A, B, C
D: [1, 1, 1, 1]  ← D can see all
```

---

## Attention as a Retrieval Mechanism

Think of self-attention as a **soft database lookup**:
- **Query**: your search query
- **Keys**: the labels on each record
- **Values**: the content of each record
- **Dot product**: how well your query matches each label
- **Softmax**: normalize to a probability distribution
- **Output**: a weighted blend of all values

Unlike a hard database lookup (return exactly one record), attention returns a **soft blend** of all records, weighted by relevance.

---

## Computational Complexity

| Aspect | Complexity |
|--------|-----------|
| Time | $O(n^2 \cdot d)$ |
| Memory | $O(n^2)$ |

The $n^2$ factor is the bottleneck for long sequences. For `n = 2048`, the attention matrix has $2048^2 = 4M$ entries.

This is why modern LLMs use tricks like:
- **GQA (Grouped Query Attention)** — share K/V across multiple Q heads (Qwen3 uses this)
- **Flash Attention** — compute attention without materializing the full $n^2$ matrix
- **Sliding window attention** — attend only to a local window

---

## Code Example

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Args:
        Q: [batch, n_heads, seq_len, head_dim]
        K: [batch, n_heads, seq_len, head_dim]
        V: [batch, n_heads, seq_len, head_dim]
        mask: causal mask [seq_len, seq_len]
    Returns:
        output: [batch, n_heads, seq_len, head_dim]
        attn_weights: [batch, n_heads, seq_len, seq_len]
    """
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

    if mask is not None:
        scores = scores + mask  # mask has -inf for future positions

    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights

# Test with dummy data
batch, heads, seq_len, d_k = 2, 8, 16, 64
Q = torch.randn(batch, heads, seq_len, d_k)
K = torch.randn(batch, heads, seq_len, d_k)
V = torch.randn(batch, heads, seq_len, d_k)

# Causal mask
mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
out, attn = scaled_dot_product_attention(Q, K, V, mask)
print("Output shape:", out.shape)    # [2, 8, 16, 64]
print("Attention shape:", attn.shape) # [2, 8, 16, 16]
```

---

## How This Applies to YUNO-LLM

- Qwen3 uses **scaled dot-product attention** with **RoPE** positional encoding applied to Q and K
- Qwen3 uses **Grouped Query Attention (GQA)**: 16 query heads, 8 key-value heads
- GQA reduces memory bandwidth during inference by sharing K/V across Q groups
- PyTorch's `F.scaled_dot_product_attention` uses Flash Attention under the hood

**Key files:**
- `docs/architecture/attention_analysis.md` — Full analysis of Qwen3's attention
- `src/yuno_llm/model.py` — Where we may later experiment with attention variants

---

## References

- Vaswani et al. "Attention is all you need." NeurIPS 2017
- Ainslie et al. "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." 2023
- Dao et al. "FlashAttention: Fast and Memory-Efficient Exact Attention." NeurIPS 2022
