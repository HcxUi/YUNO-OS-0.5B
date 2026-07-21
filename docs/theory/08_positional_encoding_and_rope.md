# 08 — Positional Encoding and RoPE

## Part A: Why Position Matters

Self-attention has no inherent notion of order — the sentence "dog bites man" and "man bites dog" would produce identical attention scores without positional information.

We must inject **position information** so the model knows that token at position 3 comes after position 2.

---

## History of Positional Encodings

| Method | Model | Pros | Cons |
|--------|-------|------|------|
| Sinusoidal (fixed) | Original Transformer | No parameters | Cannot generalize beyond training length easily |
| Learned absolute | GPT-2 | Simple | Cannot generalize beyond training length |
| Relative position | Transformer-XL | Better extrapolation | Complex implementation |
| **RoPE** | **Qwen3, LLaMA, Gemma** | **Extrapolates well, elegant** | Slightly more compute |
| ALiBi | PaLM-like | Good extrapolation | Weaker short-range |

---

## Part B: Rotary Position Embedding (RoPE)

### Intuition

Instead of adding a fixed position vector to embeddings (like sinusoidal encoding), RoPE **rotates** the query and key vectors based on their position.

The key insight: dot products between Q and K naturally encode relative positions if Q and K are rotated by amounts proportional to their absolute positions. Two tokens close together will have similar rotation; two tokens far apart will have large rotational difference.

$$\langle \mathbf{q}_m, \mathbf{k}_n \rangle = f(\mathbf{q}, \mathbf{k}, m-n)$$

The attention score between position $m$ and $n$ depends only on their **relative distance** $m-n$, not their absolute positions.

---

### Mathematics

For a query vector $\mathbf{q} \in \mathbb{R}^{d}$ at position $m$, RoPE applies a rotation matrix:

$$\mathbf{q}_m = R_m \mathbf{q}$$

The rotation matrix $R_m$ for a 2D subspace (pair of dimensions) at position $m$ with frequency $\theta_i$:

$$R_m^{(i)} = \begin{bmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{bmatrix}$$

For all $d$ dimensions, the head is split into $d/2$ pairs, each rotated at a different frequency:

$$\theta_i = 10000^{-2i/d}, \quad i = 0, 1, \ldots, \frac{d}{2}-1$$

The full rotation for position $m$:
$$R_m = \text{diag}(R_m^{(0)}, R_m^{(1)}, \ldots, R_m^{(d/2-1)})$$

---

### Efficient Implementation

Instead of explicit matrix multiplication, RoPE is implemented as:

```
q_rotated[2i]   = q[2i]   * cos(m * θ_i) - q[2i+1] * sin(m * θ_i)
q_rotated[2i+1] = q[2i+1] * cos(m * θ_i) + q[2i]   * sin(m * θ_i)
```

This avoids materializing the full rotation matrix.

---

### Qwen3 RoPE Parameters

| Parameter | Value |
|-----------|-------|
| `rope_theta` | 1,000,000 (10^6 — much higher than original 10^4) |
| `max_position_embeddings` | 32,768 |
| Rope scaling type | None (default for 0.6B) |

A higher `rope_theta` extends the wavelength of position encodings, allowing better handling of longer sequences. Qwen3 uses $\theta_{\text{base}} = 10^6$ (vs. LLaMA's $10^4$).

---

## Code Example

```python
import torch
import math

def precompute_freqs(dim: int, max_seq_len: int, theta: float = 1_000_000.0):
    """Precompute sin/cos for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)  # [seq_len, dim/2]
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin

def apply_rope(q, cos, sin):
    """Apply rotary embedding to query tensor."""
    # q: [batch, heads, seq_len, head_dim]
    q1, q2 = q[..., ::2], q[..., 1::2]  # Split into pairs
    q_rot = torch.stack([-q2, q1], dim=-1).flatten(-2)
    return q * cos + q_rot * sin

# Test
cos, sin = precompute_freqs(dim=64, max_seq_len=32)
q = torch.randn(1, 16, 32, 64)
q_rope = apply_rope(q, cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0))
print("RoPE output shape:", q_rope.shape)  # [1, 16, 32, 64]

# Inspect Qwen3's RoPE
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
attn = model.model.layers[0].self_attn
print("Rope theta:", attn.rope_theta)  # 1000000.0
```

---

## Why RoPE Is Powerful

1. **Relative position naturally encoded**: The dot product $\langle \mathbf{q}_m, \mathbf{k}_n \rangle$ depends on $m-n$ only
2. **Sequence length generalization**: Models can handle longer sequences than trained on (with some degradation)
3. **No extra parameters**: Unlike learned position embeddings, RoPE adds zero parameters
4. **Compatible with GQA**: Applied to Q and K before attention computation

---

## How This Applies to YUNO-LLM

- YUNO-LLM inherits Qwen3's RoPE with $\theta_{\text{base}} = 10^6$
- Max sequence length: 32,768 tokens out of the box
- Future experiment: **YaRN** or **LongRoPE** extension to handle 128K+ context

**Key files:**
- `docs/architecture/attention_analysis.md` — How RoPE integrates with GQA
- `config/yuno_config.yaml` — `max_position_embeddings` setting

---

## References

- Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding." 2021
- Press et al. "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation." 2022 — ALiBi
- Peng et al. "YaRN: Efficient Context Window Extension of Large Language Models." 2023
