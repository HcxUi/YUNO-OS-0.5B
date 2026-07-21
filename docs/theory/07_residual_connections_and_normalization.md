# 07 — Residual Connections and Normalization

## Part A: Residual Connections

### Intuition

In deep networks, as signals pass through many layers, gradients can **vanish** (become too small to train) or **explode** (become too large and cause instability). With 28 layers in Qwen3-0.6B, this is a real problem.

**Residual connections** (He et al., 2015) solve this by adding a **shortcut** that bypasses each sub-layer:

$$\mathbf{X}' = \mathbf{X} + \text{SubLayer}(\mathbf{X})$$

The layer only needs to learn the **difference** from the input (the residual), not the full transformation. This makes gradient flow much easier.

---

### Mathematics

Without residuals (plain network):
$$\mathbf{h}_L = f_L(f_{L-1}(\ldots f_1(\mathbf{x})))$$
Gradient at layer 1: $\frac{\partial \mathcal{L}}{\partial \mathbf{h}_1} = \prod_{l=2}^{L} \frac{\partial f_l}{\partial \mathbf{h}_{l-1}} \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{h}_L}$

This product of many Jacobians can vanish or explode.

With residuals:
$$\mathbf{h}_l = \mathbf{h}_{l-1} + f_l(\mathbf{h}_{l-1})$$
Gradient: $\frac{\partial \mathbf{h}_l}{\partial \mathbf{h}_{l-1}} = I + \frac{\partial f_l}{\partial \mathbf{h}_{l-1}}$

The identity $I$ term ensures gradients always have a path home, even if $f_l$ contributes nothing.

---

### Pattern in Qwen3 (Pre-Norm)

Qwen3 uses the **pre-norm** pattern (norm before the sub-layer):

```
Attention block:
    X → LayerNorm → Self-Attention → + → X'
    ↑_________________________________|

FFN block:
    X' → LayerNorm → FFN → + → X''
    ↑__________________________|
```

This is more stable during training than the original **post-norm** pattern.

---

## Part B: RMSNorm

### Intuition

**Layer Normalization (LayerNorm)** normalizes the activations within each token's representation to have zero mean and unit variance. This dramatically stabilizes training.

**RMSNorm** (Root Mean Square Normalization) is a simplified version that removes the mean centering — it is cheaper and works just as well in practice.

Qwen3 uses RMSNorm, as do most modern LLMs (LLaMA, Mistral, Gemma).

---

### LayerNorm Mathematics

For a vector $\mathbf{x} \in \mathbb{R}^d$:

$$\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

Where:
- $\mu = \frac{1}{d}\sum_{i=1}^d x_i$ — mean
- $\sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2$ — variance
- $\gamma, \beta \in \mathbb{R}^d$ — learned scale and bias parameters
- $\epsilon$ — small constant for numerical stability (e.g., $10^{-5}$)

---

### RMSNorm Mathematics

$$\text{RMSNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})}$$

Where:
$$\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}$$

**Key differences from LayerNorm:**
- No mean subtraction (removes centering)
- No learned bias $\beta$ (simpler)
- ~30% faster than LayerNorm in practice
- Empirically matches LayerNorm quality in LLMs

---

### Why Normalization Matters

Without normalization, activations can become extremely large or small as they pass through layers. This causes:
- Slow convergence
- Gradient instability
- Poor generalization

Normalization keeps activations in a reasonable range at every layer.

---

## Code Example

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

# RMSNorm implementation
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))  # gamma

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / rms

# Test
norm = RMSNorm(1024)
x = torch.randn(2, 10, 1024)
out = norm(x)
print("Output shape:", out.shape)  # [2, 10, 1024]

# Inspect Qwen3's actual RMSNorm
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
rms = model.model.layers[0].input_layernorm
print("RMSNorm type:", type(rms).__name__)        # Qwen3RMSNorm
print("RMSNorm weight:", rms.weight.shape)         # [1024]
print("RMSNorm epsilon:", rms.variance_epsilon)    # 1e-06

# Show residual connection explicitly
block = model.model.layers[0]
x = torch.randn(1, 5, 1024)
# Pre-norm attention residual
normed = block.input_layernorm(x)
# attn_out = block.self_attn(normed, ...)  # needs full inputs
# x = x + attn_out  ← residual
```

---

## Where They Appear in Qwen3

Each decoder block has **two** RMSNorm layers and **two** residual connections:

```python
# Qwen3DecoderLayer forward pass (simplified):
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)          # RMSNorm 1
hidden_states = self.self_attn(hidden_states, ...)
hidden_states = residual + hidden_states                     # Residual 1

residual = hidden_states
hidden_states = self.post_attention_layernorm(hidden_states) # RMSNorm 2
hidden_states = self.mlp(hidden_states)
hidden_states = residual + hidden_states                     # Residual 2
```

Plus one final RMSNorm after all 28 layers:
```python
hidden_states = model.model.norm(hidden_states)  # Final RMSNorm
```

---

## How This Applies to YUNO-LLM

- YUNO-LLM uses RMSNorm and residual connections exactly as in Qwen3
- These are **critical for training stability** — do not remove or change without strong motivation
- The learned `gamma` weights of RMSNorm are updated during fine-tuning

**Key files:**
- `docs/architecture/decoder_analysis.md` — Full decoder block with residuals shown

---

## References

- He et al. "Deep Residual Learning for Image Recognition." CVPR 2016 — Residual connections
- Ba et al. "Layer Normalization." 2016 — LayerNorm
- Zhang & Sennrich. "Root Mean Square Layer Normalization." NeurIPS 2019 — RMSNorm
