# 05 — Multi-Head Attention

## Intuition

Single-head attention can only capture one type of relationship at a time. **Multi-head attention (MHA)** runs several attention operations in parallel — each "head" can specialize in a different type of relationship.

Examples of what different heads learn:
- Head 1: syntactic relationships (subject-verb agreement)
- Head 2: coreference ("it" → "the animal")
- Head 3: positional proximity (next word relationships)
- Head 4: semantic similarity (synonyms)

Different heads independently attend to different parts of the representation.

---

## How Multi-Head Attention Works

Instead of one set of (Q, K, V) projections, we have **h sets**:

1. Project input into h separate Q, K, V subspaces
2. Run scaled dot-product attention **independently** in each head
3. Concatenate all h outputs
4. Apply a final output projection $W_O$

```
Input X
    ↓
[W_Q1, W_K1, W_V1] → head_1 = Attention(Q1, K1, V1)
[W_Q2, W_K2, W_V2] → head_2 = Attention(Q2, K2, V2)
     ...
[W_Qh, W_Kh, W_Vh] → head_h = Attention(Qh, Kh, Vh)
    ↓
Concat([head_1, ..., head_h])
    ↓
× W_O
    ↓
Output
```

---

## Mathematics

$$\text{MultiHead}(\mathbf{X}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \cdot W_O$$

Where each head is:
$$\text{head}_i = \text{Attention}(\mathbf{X}W_{Q_i}, \mathbf{X}W_{K_i}, \mathbf{X}W_{V_i})$$

Dimensions in Qwen3-0.6B:
- $d_{\text{model}} = 1024$
- $h = 16$ (number of query heads)
- $d_k = d_v = d_{\text{model}} / h = 64$ (per-head dimension)
- $W_{Q_i} \in \mathbb{R}^{1024 \times 64}$ per head
- $W_O \in \mathbb{R}^{(h \cdot d_v) \times d_{\text{model}}} = \mathbb{R}^{1024 \times 1024}$

---

## Grouped Query Attention (GQA)

Qwen3 uses **GQA** instead of standard MHA. In GQA:
- There are still 16 **query** heads
- But only 8 **key-value** heads (one KV pair shared by every 2 Q heads)

```
Standard MHA:   16 Q heads, 16 K heads, 16 V heads
GQA (Qwen3):    16 Q heads,  8 K heads,  8 V heads
```

**Why GQA?**
- During inference, K and V are stored in the **KV cache**
- Fewer KV heads = smaller cache = less memory = faster inference
- Quality is nearly identical to full MHA

**GQA grouping:**
```
Q heads:  [Q1, Q2] share → [K1, V1]
          [Q3, Q4] share → [K2, V2]
          ...
          [Q15, Q16] share → [K8, V8]
```

Each Q head attends using its own Q but shares the same K and V with its group partner.

---

## Parameter Counts (Qwen3-0.6B per attention layer)

| Matrix | Shape | Parameters |
|--------|-------|-----------|
| $W_Q$ (q_proj) | 1024 × 1024 | 1,048,576 |
| $W_K$ (k_proj) | 1024 × 512 | 524,288 |
| $W_V$ (v_proj) | 1024 × 512 | 524,288 |
| $W_O$ (o_proj) | 1024 × 1024 | 1,048,576 |
| **Total per layer** | | **3,145,728** |
| **Total (28 layers)** | | **88,080,384** |

Note: k_proj and v_proj are smaller (512 not 1024) because we only have 8 KV heads × 64 head_dim = 512.

---

## Code Example

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
attn = model.model.layers[0].self_attn

print("Query heads:", attn.num_heads)         # 16
print("KV heads:", attn.num_key_value_heads)  # 8
print("Head dim:", attn.head_dim)             # 64

print("q_proj:", attn.q_proj.weight.shape)    # [1024, 1024]
print("k_proj:", attn.k_proj.weight.shape)    # [512, 1024]
print("v_proj:", attn.v_proj.weight.shape)    # [512, 1024]
print("o_proj:", attn.o_proj.weight.shape)    # [1024, 1024]
```

---

## Attention Score Visualization

During inference, you can visualize which tokens each head attends to:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

text = "The cat sat on the mat"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

# Attention weights: list of [batch, heads, seq, seq] per layer
attn_layer0 = outputs.attentions[0]  # Layer 0
print("Attention shape:", attn_layer0.shape)   # [1, 16, 6, 6]
# Each head has a 6×6 attention matrix (6 tokens)
```

---

## How This Applies to YUNO-LLM

- YUNO-LLM inherits Qwen3's GQA (16 Q heads, 8 KV heads)
- Possible future experiment: try different head counts and measure quality vs. efficiency tradeoff
- **Do not change** head configuration until Phase 5+ and evaluation baseline is established

**Key files:**
- `docs/architecture/attention_analysis.md` — GQA implementation details
- `src/yuno_llm/model.py` — Where attention config is referenced

---

## References

- Vaswani et al. "Attention is all you need." 2017 — Original MHA
- Ainslie et al. "GQA: Training Generalized Multi-Query Transformer Models." 2023
- Shazeer. "Fast Transformer Decoding: One Write-Head is All You Need." 2019 — Multi-Query Attention precursor
