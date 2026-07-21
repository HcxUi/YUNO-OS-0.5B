# YUNO-LLM Architecture Reference

> Base model: **Qwen3-0.6B** — decoder-only transformer with GQA and RoPE
> This document is the single source of truth for YUNO-LLM's architecture.

---

## Quick Reference

| Property | Value |
|----------|-------|
| Architecture | Decoder-only Transformer |
| Base model | Qwen/Qwen3-0.6B |
| Hidden size (`d_model`) | 1024 |
| Number of layers | 28 |
| Attention heads (Q) | 16 |
| KV heads (GQA) | 8 |
| Head dimension | 64 |
| FFN intermediate size | 3072 |
| FFN activation | SiLU (SwiGLU) |
| Normalization | RMSNorm (ε=1e-6) |
| Positional encoding | RoPE (θ=1,000,000) |
| Vocabulary size | 151,936 |
| Max sequence length | 32,768 |
| Total parameters | ~596M |
| Model size (fp32) | ~2.4 GB |
| Model size (fp16) | ~1.2 GB |

---

## Full Layer-by-Layer Breakdown

### Layer 0 — Token Embedding

| Property | Value |
|----------|-------|
| **Layer** | `model.embed_tokens` |
| **Type** | `nn.Embedding` |
| **Input shape** | `[batch, seq_len]` (token IDs) |
| **Output shape** | `[batch, seq_len, 1024]` |
| **Parameters** | 151,936 × 1024 = 155,582,464 |
| **Purpose** | Convert integer token IDs → dense float vectors |
| **Mathematics** | $\mathbf{X} = E[\mathbf{t}]$ where $E \in \mathbb{R}^{151936 \times 1024}$ |
| **Code location** | `transformers/models/qwen3/modeling_qwen3.py` → `Qwen3Model.embed_tokens` |
| **Shared with** | `lm_head` (weight tying) |
| **Possible improvements** | Domain-specific token initialization; vocabulary extension |

---

### Layers 1–28 — Decoder Blocks (×28)

Each block has this structure:

```
input
  │
  ├─── [Residual path] ──────────────────────────┐
  ↓                                               │
input_layernorm (RMSNorm)                         │
  ↓                                               │
self_attn (GQA + RoPE + Causal Mask)              │
  ↓                                               │
  └──────────────────────────── (+) ←─────────── ┘
                                  │
                              attention output
                                  │
  ┌─── [Residual path] ──────────┘
  │
  ↓
post_attention_layernorm (RMSNorm)
  │
  ↓
mlp (SwiGLU FFN)
  │
  └──────────────────────────── (+) ←─────────── ┘
                                  │
                             block output
```

#### Decoder Block — Attention Sub-Layer

| Property | Value |
|----------|-------|
| **Component** | `model.layers[i].input_layernorm` |
| **Type** | `Qwen3RMSNorm` |
| **Input shape** | `[batch, seq_len, 1024]` |
| **Output shape** | `[batch, seq_len, 1024]` |
| **Parameters** | 1024 (gamma only) |
| **Purpose** | Normalize activations before attention |
| **Mathematics** | $\text{RMSNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}}$ |

| Property | Value |
|----------|-------|
| **Component** | `model.layers[i].self_attn` |
| **Type** | `Qwen3Attention` |
| **Input shape** | `[batch, seq_len, 1024]` |
| **Output shape** | `[batch, seq_len, 1024]` |
| **Q projection** | `[1024, 1024]` → 16 heads × 64 dim |
| **K projection** | `[512, 1024]` → 8 KV heads × 64 dim |
| **V projection** | `[512, 1024]` → 8 KV heads × 64 dim |
| **O projection** | `[1024, 1024]` |
| **Parameters** | 3,145,728 per layer |
| **Purpose** | Attend to all past positions with positional awareness |
| **Mathematics** | $\text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{64}}\right)V$ with RoPE on Q and K |
| **Possible improvements** | Increase head_dim; try FlashAttention2; adjust GQA ratio |

#### Decoder Block — FFN Sub-Layer

| Property | Value |
|----------|-------|
| **Component** | `model.layers[i].mlp` |
| **Type** | `Qwen3MLP` (SwiGLU) |
| **Input shape** | `[batch, seq_len, 1024]` |
| **Output shape** | `[batch, seq_len, 1024]` |
| **gate_proj** | `[3072, 1024]` |
| **up_proj** | `[3072, 1024]` |
| **down_proj** | `[1024, 3072]` |
| **Parameters** | 9,437,184 per layer |
| **Purpose** | Per-token nonlinear transformation; factual memory |
| **Mathematics** | $\text{FFN}(\mathbf{x}) = \text{SiLU}(\mathbf{x}W_g) \odot (\mathbf{x}W_u) \cdot W_d$ |
| **Possible improvements** | Increase intermediate_size; try MoE; change activation |

---

### Layer 29 — Final RMSNorm

| Property | Value |
|----------|-------|
| **Layer** | `model.model.norm` |
| **Type** | `Qwen3RMSNorm` |
| **Input shape** | `[batch, seq_len, 1024]` |
| **Output shape** | `[batch, seq_len, 1024]` |
| **Parameters** | 1,024 |
| **Purpose** | Final normalization before projection to vocabulary |

---

### Layer 30 — LM Head

| Property | Value |
|----------|-------|
| **Layer** | `model.lm_head` |
| **Type** | `nn.Linear` (no bias) |
| **Input shape** | `[batch, seq_len, 1024]` |
| **Output shape** | `[batch, seq_len, 151936]` |
| **Parameters** | 0 (weight-tied to embed_tokens) |
| **Purpose** | Project hidden state → vocabulary logits |
| **Mathematics** | $\text{logits} = \mathbf{h} \cdot E^T$ (weight tying) |

---

## Parameter Budget

| Component | Parameters | % of Total |
|-----------|-----------|-----------|
| Embedding (embed_tokens) | 155,582,464 | 26.1% |
| Attention (all 28 layers) | 88,080,384 | 14.8% |
| FFN (all 28 layers) | 264,241,152 | 44.3% |
| LayerNorms (all) | ~58,000 | <0.01% |
| LM Head | 0 (tied) | 0% |
| **Total** | **~596M** | **100%** |

---

## Data Flow (Single Forward Pass)

```
Input: "Hello, YUNO!"
    ↓ tokenizer()
[9707, 11, 816, 1502, 0]   shape: [1, 5]
    ↓ embed_tokens
[[v1], [v2], [v3], [v4], [v5]]  shape: [1, 5, 1024]
    ↓ × 28 decoder blocks
[h1, h2, h3, h4, h5]  shape: [1, 5, 1024]
    ↓ final RMSNorm
[h1', h2', h3', h4', h5']  shape: [1, 5, 1024]
    ↓ lm_head (× E^T)
logits  shape: [1, 5, 151936]
    ↓ softmax + sample on last position
next token ID
    ↓ tokenizer.decode()
"How"
```

---

## LoRA Target Modules (for Fine-Tuning)

For efficient fine-tuning with PEFT/LoRA:

| Layer | Module | Shape | Priority |
|-------|--------|-------|---------|
| Attention | q_proj | 1024×1024 | ✅ High |
| Attention | k_proj | 512×1024 | ✅ High |
| Attention | v_proj | 512×1024 | ✅ High |
| Attention | o_proj | 1024×1024 | ✅ High |
| FFN | gate_proj | 3072×1024 | Medium |
| FFN | up_proj | 3072×1024 | Medium |
| FFN | down_proj | 1024×3072 | Medium |

Start with attention-only LoRA, add FFN if more capacity is needed.

---

## Files in This Section

| File | Contents |
|------|---------|
| `Architecture.md` (this file) | Full layer table |
| `config_analysis.md` | Qwen3Config detailed analysis |
| `tokenizer_analysis.md` | Qwen3 tokenizer internals |
| `embedding_analysis.md` | Embedding layer deep dive |
| `attention_analysis.md` | GQA + RoPE implementation |
| `decoder_analysis.md` | Full decoder block walkthrough |
| `output_head_analysis.md` | LM head + weight tying |

---

## Change Log

| Version | Date | Change |
|---------|------|--------|
| v0.1 | 2026-07-21 | Initial architecture analysis based on Qwen3-0.6B |
