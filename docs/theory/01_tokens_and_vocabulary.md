# 01 — Tokens and Vocabulary

## Intuition

Before a language model can process text, it needs to convert text into numbers. This is what **tokenization** does.

A **token** is the basic unit of text that the model works with. It is not always a word — it can be:
- A full word: `"hello"` → `[15339]`
- A subword: `"unbelievable"` → `["un", "believ", "able"]` → `[946, 48194, 481]`
- A character: `"a"` → `[64]`
- A special symbol: `"<|endoftext|>"` → `[100257]`

A **vocabulary** is the complete set of all tokens the model knows. In Qwen3, the vocabulary size is **151,936 tokens**.

---

## Why Subwords?

Three approaches exist:

| Approach | Vocab Size | Problem |
|----------|-----------|---------|
| Character-level | ~100 | Sequences become very long |
| Word-level | ~500,000+ | Unknown words (OOV), huge vocab |
| **Subword (BPE)** | ~32K–150K | ✅ Balances length and coverage |

Subword tokenization is the industry standard. It handles:
- Rare words by splitting them into known parts
- New words (typos, jargon) gracefully
- Multiple languages with one vocabulary

---

## Byte-Pair Encoding (BPE)

Qwen3 uses **BPE** (Byte-Pair Encoding), invented for NLP by Sennrich et al. (2016).

**Algorithm:**
1. Start with character-level vocabulary
2. Count all adjacent pairs in the corpus
3. Merge the most frequent pair into a new token
4. Repeat until vocabulary size target is reached

**Example:**
```
Corpus: "low low lower lowest"

Start: l o w | l o w | l o w e r | l o w e s t

Merge "lo" → lo w | lo w | lo w e r | lo w e s t
Merge "low" → low | low | low e r | low e s t
Merge "lowe" → low | low | lowe r | lowe s t
...
```

---

## Mathematics

Let vocabulary $V$ have size $|V|$. Each token $t_i \in V$ is an integer index.

For a string $s$, the tokenizer produces a sequence:
$$\text{tokens}(s) = [t_1, t_2, \ldots, t_n], \quad t_i \in \{0, 1, \ldots, |V|-1\}$$

The embedding layer maps each token index to a vector:
$$\mathbf{e}_i = E[t_i] \in \mathbb{R}^{d_{\text{model}}}$$

where $E \in \mathbb{R}^{|V| \times d_{\text{model}}}$ is the embedding matrix.

---

## Special Tokens

Qwen3 uses several special tokens:

| Token | Purpose |
|-------|---------|
| `<\|im_start\|>` | Marks the start of a chat message turn |
| `<\|im_end\|>` | Marks the end of a chat message turn |
| `<\|endoftext\|>` | Marks end of document (EOS) |
| `<\|pad\|>` | Padding token for batched inputs |

**Chat template example:**
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
4.<|im_end|>
```

---

## Code Example

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# Tokenize text
text = "Hello, how are you?"
tokens = tokenizer(text, return_tensors="pt")
print(tokens.input_ids)         # tensor([[9707,  11,  1246,  553,  498,  30]])

# Decode back
decoded = tokenizer.decode(tokens.input_ids[0])
print(decoded)                  # "Hello, how are you?"

# Inspect vocabulary
print("Vocab size:", tokenizer.vocab_size)   # 151936
print("EOS token:", tokenizer.eos_token)     # <|endoftext|>
```

---

## How This Applies to YUNO-LLM

- YUNO-LLM inherits the Qwen3 tokenizer (vocab size: 151,936)
- All inputs are first tokenized before reaching the model
- The tokenizer is **frozen** — we do not retrain it unless adding domain-specific tokens
- Custom special tokens (e.g., `<|yuno_start|>`) can be added to extend the vocabulary

**Key files:**
- `tokenizer/` — Tokenizer experiments
- `src/yuno_llm/tokenizer.py` — YunoTokenizer wrapper
- `docs/architecture/tokenizer_analysis.md` — Deep dive into Qwen3's tokenizer

---

## References

- Sennrich, Rico, et al. "Neural machine translation of rare words with subword units." (2016) — Original BPE paper
- https://huggingface.co/learn/nlp-course/chapter6/5 — HuggingFace tokenizer tutorial
- https://huggingface.co/Qwen/Qwen3-0.6B — Qwen3 tokenizer details
