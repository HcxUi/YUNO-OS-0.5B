# 06 — Feed-Forward Networks (FFN)

## Intuition

After self-attention allows tokens to communicate with each other, the **Feed-Forward Network (FFN)** applies a **per-token** transformation — the same operation is applied independently to each token's representation.

If attention is about **communication between tokens**, FFN is about **thinking within each token**.

The FFN dramatically increases the model's capacity to store knowledge. Research suggests FFN layers act as "key-value memory" — they memorize facts during training and retrieve them during inference.

---

## Standard FFN (Transformer Paper)

The original Transformer used a simple two-layer MLP:

$$\text{FFN}(\mathbf{x}) = \text{ReLU}(\mathbf{x} W_1 + b_1) W_2 + b_2$$

Where:
- $\mathbf{x} \in \mathbb{R}^{d_{\text{model}}}$ — single token representation
- $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ffn}}}$ — expand
- $W_2 \in \mathbb{R}^{d_{\text{ffn}} \times d_{\text{model}}}$ — contract
- $d_{\text{ffn}} = 4 \times d_{\text{model}}$ — standard expansion ratio

---

## SwiGLU FFN (Used by Qwen3)

Modern LLMs (Qwen3, LLaMA, Mistral) replace the standard FFN with **SwiGLU**, which uses a gating mechanism:

$$\text{SwiGLU}(\mathbf{x}) = \left(\text{SiLU}(\mathbf{x} W_{\text{gate}}) \odot (\mathbf{x} W_{\text{up}})\right) W_{\text{down}}$$

Where:
- $W_{\text{gate}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ffn}}}$ — gate projection
- $W_{\text{up}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ffn}}}$ — up projection
- $W_{\text{down}} \in \mathbb{R}^{d_{\text{ffn}} \times d_{\text{model}}}$ — down projection
- $\odot$ — element-wise multiplication
- $\text{SiLU}(x) = x \cdot \sigma(x)$ — Sigmoid Linear Unit

**Differences from standard FFN:**
1. **Three matrices** instead of two
2. **Gating**: the gate controls how much of the `up` projection passes through
3. **SiLU** instead of ReLU — smoother gradient, better performance

---

## SiLU Activation

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

Properties:
- Smooth (differentiable everywhere)
- Not zero for negative inputs (unlike ReLU)
- Also called Swish

```python
import torch
import matplotlib.pyplot as plt

x = torch.linspace(-4, 4, 100)
silu = x * torch.sigmoid(x)
relu = torch.relu(x)

# SiLU is smoother than ReLU and allows small negative values
```

---

## Qwen3-0.6B FFN Parameters

| Matrix | Shape | Parameters |
|--------|-------|-----------|
| gate_proj | 1024 → 3072 | 3,145,728 |
| up_proj | 1024 → 3072 | 3,145,728 |
| down_proj | 3072 → 1024 | 3,145,728 |
| **Total per layer** | | **9,437,184** |
| **Total (28 layers)** | | **264,241,152** |

The FFN accounts for **~44%** of total parameters — it is the largest component by parameter count.

---

## Why the FFN Is So Large

Research by Geva et al. (2021) shows that FFN layers in Transformers act as "key-value memories":

- Each row of $W_1$ (gate/up) is a **key** that activates on certain input patterns
- The corresponding row of $W_2$ (down) is the **value** retrieved
- The model stores facts like "Paris is the capital of France" distributed across many neurons

Increasing FFN size = more storage capacity for knowledge.

---

## Code Example

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
ffn = model.model.layers[0].mlp

print("FFN type:", type(ffn).__name__)       # Qwen3MLP
print("gate_proj:", ffn.gate_proj.weight.shape)  # [3072, 1024]
print("up_proj:  ", ffn.up_proj.weight.shape)    # [3072, 1024]
print("down_proj:", ffn.down_proj.weight.shape)  # [1024, 3072]
print("activation:", type(ffn.act_fn).__name__)   # SiLU

# Manual forward pass through FFN
import torch
x = torch.randn(1, 10, 1024)  # [batch, seq_len, d_model]

gate_out = ffn.act_fn(ffn.gate_proj(x))  # SiLU(x @ gate_proj.T)
up_out = ffn.up_proj(x)                  # x @ up_proj.T
gated = gate_out * up_out                # element-wise gate
out = ffn.down_proj(gated)               # contract back to d_model
print("FFN output shape:", out.shape)    # [1, 10, 1024]
```

---

## How This Applies to YUNO-LLM

- YUNO-LLM uses Qwen3's SwiGLU FFN unchanged initially
- The FFN is where most **factual knowledge** is stored
- Fine-tuning updates FFN weights → teaches YUNO-LLM new behaviors and facts
- LoRA targets `gate_proj`, `up_proj`, `down_proj` for efficient fine-tuning

**Future experiment ideas:**
- Increase `d_ffn` (intermediate_size) for more capacity
- Try different activation functions (GELU, ReLU²)
- Mixture-of-Experts (MoE): replace dense FFN with a router + multiple expert FFNs

**Key files:**
- `config/training_config.yaml` — LoRA targets include FFN layers
- `docs/architecture/decoder_analysis.md` — Full decoder block including FFN

---

## References

- Vaswani et al. "Attention is all you need." 2017
- Shazeer. "GLU Variants Improve Transformer." 2020 — SwiGLU
- Geva et al. "Transformer Feed-Forward Layers Are Key-Value Memories." EMNLP 2021
- Ramachandran et al. "Searching for Activation Functions." 2017 — SiLU/Swish
