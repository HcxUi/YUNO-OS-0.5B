# 09 — KV Cache

## Intuition

During **autoregressive generation**, the model generates one token at a time. For each new token, it must compute attention over all previous tokens.

Naively, this means recomputing the **Key (K)** and **Value (V)** tensors for every previous token at every generation step — extremely wasteful.

The **KV Cache** stores the K and V tensors for all previous tokens and reuses them. Only the new token's K and V need to be computed.

---

## The Problem (Without Cache)

Generating a 100-token response from a 50-token prompt:
- Step 1: Process all 50 prompt tokens → generate token 51
- Step 2: Process tokens 1–51 → generate token 52
- Step 3: Process tokens 1–52 → generate token 53
- ...

Without a cache, generating token `t` requires recomputing K and V for all `t-1` previous tokens. Total compute scales as $O(n^2)$ with sequence length — very slow.

---

## The Solution (With KV Cache)

At each generation step, instead of recomputing everything:

1. **Prefill phase**: Process the full prompt once, compute and **store** K, V for all prompt tokens
2. **Decode phase**: For each new token, compute only the new K, V and **append** to the cache

At step $t$, the cache contains:
$$\text{cache}_K = [K_1, K_2, \ldots, K_{t-1}], \quad \text{cache}_V = [V_1, V_2, \ldots, V_{t-1}]$$

New token attention:
$$\text{Attention}(Q_t, [K_1, \ldots, K_t], [V_1, \ldots, V_t])$$

---

## Memory Cost of KV Cache

For Qwen3-0.6B with GQA (8 KV heads, head_dim=64):

$$\text{Cache size per layer} = 2 \times n_{\text{kv\_heads}} \times \text{head\_dim} \times \text{seq\_len} \times \text{bytes}$$
$$= 2 \times 8 \times 64 \times n \times 2 \quad \text{(float16)}$$

For 2048 token context, 28 layers:
$$= 2 \times 8 \times 64 \times 2048 \times 2 \times 28 = \text{~118MB}$$

For 32,768 token context: **~1.9GB** just for the KV cache.

This is why GQA (fewer KV heads) is critical — it directly reduces cache memory.

---

## Code Example

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate WITHOUT cache (slower, for demonstration)
with torch.no_grad():
    out_no_cache = model.generate(
        **inputs,
        max_new_tokens=20,
        use_cache=False
    )

# Generate WITH cache (default, faster)
with torch.no_grad():
    out_with_cache = model.generate(
        **inputs,
        max_new_tokens=20,
        use_cache=True   # Default
    )

print(tokenizer.decode(out_with_cache[0]))
```

---

## Inspecting the Cache During Generation

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import torch

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

inputs = tokenizer("Hello, my name is", return_tensors="pt")

# Run one forward pass and capture the cache
with torch.no_grad():
    outputs = model(**inputs, use_cache=True)

past_kv = outputs.past_key_values
print("Number of layers in cache:", len(past_kv))  # 28

# Each layer has (key, value) tensors
k, v = past_kv[0]
print("Key shape:", k.shape)    # [batch, n_kv_heads, seq_len, head_dim]
print("Value shape:", v.shape)  # [batch, n_kv_heads, seq_len, head_dim]
# → [1, 8, 4, 64]  (1 batch, 8 KV heads, 4 tokens, 64 head_dim)
```

---

## Modern Cache Implementations

| Cache Type | Description |
|-----------|-------------|
| `DynamicCache` | Grows dynamically (HuggingFace default) |
| `StaticCache` | Pre-allocated fixed size (faster, CUDA graphs compatible) |
| `SinkCache` | Keeps "attention sinks" (first few tokens always) |
| `QuantizedCache` | Stores K/V in int4/int8 to save memory |

For YUNO-LLM inference, `DynamicCache` is fine initially. `StaticCache` with `torch.compile` can give 2–3× speedup.

---

## Impact on YUNO-LLM

- The KV cache is automatically managed by HuggingFace's `generate()` — no manual work needed initially
- Understanding the cache matters for:
  - Building a streaming inference server (Phase 6)
  - Optimizing for long-context performance
  - Implementing prefix caching (reuse cache for shared system prompts)

**Key files:**
- `inference/generate.py` — Uses `use_cache=True` by default
- `inference/server.py` — Will manage cache across requests

---

## References

- Pope et al. "Efficiently Scaling Transformer Inference." 2023 — KV cache analysis
- Shazeer. "Fast Transformer Decoding: One Write-Head is All You Need." 2019 — Multi-Query Attention reduces cache
- https://huggingface.co/docs/transformers/kv_cache — HuggingFace cache docs
