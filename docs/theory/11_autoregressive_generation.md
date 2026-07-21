# 11 — Autoregressive Generation

## Intuition

**Autoregressive generation** is how a language model produces text: one token at a time, where each new token is conditioned on all previously generated tokens.

```
Prompt: "The sky is"
Step 1: model sees "The sky is"         → predicts "blue"
Step 2: model sees "The sky is blue"    → predicts "because"
Step 3: model sees "The sky is blue because" → predicts "of"
...
```

Each generated token is appended to the context and becomes part of the next input. This continues until an **end-of-sequence (EOS)** token is generated or `max_new_tokens` is reached.

---

## From Logits to Tokens — Decoding Strategies

The model outputs **logits** (raw scores over the vocabulary). We must convert these to a single token choice. Several strategies exist:

### 1. Greedy Decoding
Always pick the highest-probability token:
$$t_i = \arg\max_t P(t | t_1, \ldots, t_{i-1})$$

- ✅ Fast, deterministic
- ❌ Repetitive, boring output
- Use for: evaluation benchmarks where reproducibility matters

### 2. Beam Search
Keep the top-$k$ most probable sequences at each step and pick the globally best:
- ✅ Higher quality than greedy
- ❌ Still repetitive, slow for large beams
- Use for: translation, summarization (not conversational LLMs)

### 3. Temperature Sampling
Sample from the distribution, scaling by temperature $T$:
$$P'(t) = \frac{\exp(\text{logit}_t / T)}{\sum_j \exp(\text{logit}_j / T)}$$

- $T < 1$: sharper distribution (more confident, less creative)
- $T = 1$: original distribution
- $T > 1$: flatter distribution (more random)
- ✅ Diverse, creative output
- ❌ Can produce incoherent text at high temperatures

### 4. Top-k Sampling
Sample only from the top-$k$ highest probability tokens:
$$P'(t) = \begin{cases} P(t) / Z & \text{if } t \in \text{top-}k \\ 0 & \text{otherwise} \end{cases}$$

- Truncates long tail of unlikely tokens
- Typical: $k = 50$

### 5. Top-p (Nucleus) Sampling
Sample from the smallest set of tokens whose cumulative probability exceeds $p$:
$$P'(t) \text{ from } \{t : \sum_{i=1}^{k} P(t_i) \geq p\}$$

- Adapts the number of candidates based on the distribution shape
- Typical: $p = 0.9$ or $p = 0.95$
- ✅ Usually better than top-k alone

### 6. Combined (YUNO-LLM Default)
Most production systems use: **temperature + top-p + repetition penalty**

---

## Repetition Penalty

Reduces the probability of tokens that have already appeared:

$$\text{logit}'_t = \begin{cases} \text{logit}_t / r & \text{if } t \in \text{past tokens} \\ \text{logit}_t & \text{otherwise} \end{cases}$$

Where $r > 1$ penalizes repetition. Qwen3 default: $r = 1.1$

---

## Mathematics of Autoregressive Generation

The model computes:
$$P(\mathbf{t}) = \prod_{i=1}^{n} P(t_i | t_1, \ldots, t_{i-1}; \theta)$$

Generation finds a sequence that approximately maximizes this joint probability (exact maximization is NP-hard — sampling approximates it).

**Perplexity** measures how surprised the model is by the true sequence:
$$\text{PPL} = \exp\!\left(-\frac{1}{n} \sum_{i=1}^{n} \log P(t_i | t_1, \ldots, t_{i-1})\right)$$

Lower perplexity = better model. Random guess on vocab of 151,936 = PPL of 151,936. A good LLM: PPL < 10 on its training distribution.

---

## Code Example

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

prompt = "Explain quantum entanglement in simple terms:"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")

# --- Greedy ---
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
print("Greedy:", tokenizer.decode(out[0][inputs.input_ids.shape[1]:]))

# --- Temperature + Top-p sampling ---
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )
print("Sampled:", tokenizer.decode(out[0][inputs.input_ids.shape[1]:]))

# --- Streaming output ---
streamer = TextStreamer(tokenizer, skip_special_tokens=True)
with torch.no_grad():
    model.generate(**inputs, max_new_tokens=100, do_sample=True,
                   temperature=0.7, streamer=streamer)
```

---

## Generation Pipeline Summary

```
Prompt (text)
    ↓ tokenizer.apply_chat_template()
Formatted prompt with special tokens
    ↓ tokenizer()
Token IDs  [1, 9707, 374, ...]
    ↓ model.forward()
Logits  [vocab_size=151936 values]
    ↓ Apply temperature, top-k, top-p
Filtered distribution
    ↓ torch.multinomial() or argmax
Next token ID
    ↓ Append to context
Repeat until EOS or max_new_tokens
    ↓ tokenizer.decode()
Generated text
```

---

## YUNO-LLM Generation Defaults

From `config/yuno_config.yaml`:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_new_tokens` | 512 | Balanced response length |
| `temperature` | 0.7 | Slightly creative, not chaotic |
| `top_p` | 0.9 | Nucleus sampling |
| `repetition_penalty` | 1.1 | Mild anti-repetition |
| `do_sample` | true | Sampling, not greedy |

These defaults can be overridden per-request in `inference/generate.py`.

---

## How This Applies to YUNO-LLM

- All YUNO-LLM inference uses autoregressive generation with nucleus sampling
- The system prompt (identity) is injected before user messages via the chat template
- Streaming output is implemented in `inference/generate.py` using `TextStreamer`

**Key files:**
- `src/yuno_llm/generation.py` — Custom generation logic and defaults
- `inference/generate.py` — CLI inference script
- `config/yuno_config.yaml` — `generation:` section

---

## References

- Holtzman et al. "The Curious Case of Neural Text Degeneration." ICLR 2020 — Top-p sampling
- Fan et al. "Hierarchical Neural Story Generation." ACL 2018 — Top-k sampling
- Vijayakumar et al. "Diverse Beam Search." 2016
- https://huggingface.co/docs/transformers/generation_strategies — HuggingFace generation docs
