"""
YUNO-LLM: Verify Model Installation
======================================
Loads the base model, runs a forward pass, prints layer shapes and
parameter counts. Use this to confirm the model is correctly installed.

Usage:
    python scripts/verify_model.py
    python scripts/verify_model.py --model Qwen/Qwen3-0.6B
    python scripts/verify_model.py --local models/base/Qwen--Qwen3-0.6B
"""

import argparse
import sys
import numpy as np
np.complex = complex  # Monkeypatch for legacy librosa compatibility with modern numpy
from pathlib import Path
import time

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def verify_model(model_id: str, local_path: str = None) -> None:
    """Load model, run forward pass, print diagnostic information."""
    print()
    print("  YUNO-LLM — Model Verification")
    print("  " + "="*50)

    # ── 1. Import ──────────────────────────────────────────────────────────────
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"  PyTorch version:  {torch.__version__}")
        print(f"  Device:           {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        if torch.cuda.is_available():
            print(f"  GPU:              {torch.cuda.get_device_name(0)}")
            print(f"  VRAM:             {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError as e:
        print(f"  [ERROR] Import failed: {e}")
        sys.exit(1)

    source = local_path or model_id
    print(f"  Model source:     {source}")
    print()

    # ── 2. Load tokenizer ──────────────────────────────────────────────────────
    print("  Loading tokenizer...")
    t0 = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
        print(f"  [SUCCESS] Tokenizer loaded in {time.time()-t0:.1f}s")
        print(f"     Vocab size:     {tokenizer.vocab_size:,}")
        print(f"     EOS token:      {tokenizer.eos_token!r}")
        print(f"     PAD token:      {tokenizer.pad_token!r}")
    except Exception as e:
        print(f"  [ERROR] Tokenizer failed: {e}")
        sys.exit(1)

    # ── 3. Load model ──────────────────────────────────────────────────────────
    print()
    print("  Loading model (CPU, float32 — this may take ~30s)...")
    t0 = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            source,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        model.eval()
        print(f"  [SUCCESS] Model loaded in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"  [ERROR] Model load failed: {e}")
        sys.exit(1)

    # ── 4. Architecture summary ────────────────────────────────────────────────
    print()
    print("  Architecture:")
    cfg = model.config
    print(f"     Model type:           {cfg.model_type}")
    print(f"     Hidden size (d_model): {cfg.hidden_size}")
    print(f"     Num layers:            {cfg.num_hidden_layers}")
    print(f"     Num attention heads:   {cfg.num_attention_heads}")
    print(f"     Num KV heads (GQA):    {cfg.num_key_value_heads}")
    print(f"     Intermediate size:     {cfg.intermediate_size}")
    print(f"     Max position emb:      {cfg.max_position_embeddings}")
    print(f"     Vocab size:            {cfg.vocab_size:,}")
    print(f"     RoPE theta:            {getattr(cfg, 'rope_theta', 'N/A')}")
    print(f"     Norm epsilon:          {getattr(cfg, 'rms_norm_eps', 'N/A')}")

    # ── 5. Parameter count ─────────────────────────────────────────────────────
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print()
    print("  Parameters:")
    print(f"     Total:     {total:>15,}")
    print(f"     Trainable: {trainable:>15,}")
    print(f"     Size (fp32): ~{total * 4 / 1e9:.2f} GB")
    print(f"     Size (fp16): ~{total * 2 / 1e9:.2f} GB")

    # ── 6. Layer shapes ────────────────────────────────────────────────────────
    print()
    print("  Layer 0 shapes:")
    block = model.model.layers[0]
    attn = block.self_attn
    mlp = block.mlp
    print(f"     embed_tokens:   {model.model.embed_tokens.weight.shape}")
    print(f"     q_proj:         {attn.q_proj.weight.shape}")
    print(f"     k_proj:         {attn.k_proj.weight.shape}")
    print(f"     v_proj:         {attn.v_proj.weight.shape}")
    print(f"     o_proj:         {attn.o_proj.weight.shape}")
    print(f"     gate_proj:      {mlp.gate_proj.weight.shape}")
    print(f"     up_proj:        {mlp.up_proj.weight.shape}")
    print(f"     down_proj:      {mlp.down_proj.weight.shape}")
    print(f"     lm_head:        {model.lm_head.weight.shape}")

    # ── 7. Forward pass ────────────────────────────────────────────────────────
    print()
    print("  Running forward pass...")
    import torch
    test_text = "YUNO-LLM is a research-grade language model."
    inputs = tokenizer(test_text, return_tensors="pt")
    t0 = time.time()
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
    elapsed = time.time() - t0
    print(f"  [SUCCESS] Forward pass complete in {elapsed:.2f}s")
    print(f"     Input tokens:   {inputs.input_ids.shape[1]}")
    print(f"     Logits shape:   {outputs.logits.shape}")
    print(f"     Loss:           {outputs.loss.item():.4f}")

    # ── 8. Generation test ────────────────────────────────────────────────────
    print()
    print("  Running generation test (greedy, 20 tokens)...")
    prompt = "The most important thing about AI safety is"
    gen_inputs = tokenizer(prompt, return_tensors="pt")
    t0 = time.time()
    with torch.no_grad():
        gen_out = model.generate(**gen_inputs, max_new_tokens=20, do_sample=False)
    elapsed = time.time() - t0
    new_tokens = gen_out[0][gen_inputs.input_ids.shape[1]:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    tps = len(new_tokens) / elapsed
    print(f"  [SUCCESS] Generation complete: {tps:.1f} tokens/sec")
    print(f"     Prompt:    {prompt!r}")
    print(f"     Generated: {generated_text!r}")

    print()
    print("  " + "="*50)
    print("  [SUCCESS] All checks passed. Model is ready for YUNO-LLM.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify YUNO-LLM base model")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace model ID (default: Qwen/Qwen3-0.6B)"
    )
    parser.add_argument(
        "--local",
        default=None,
        help="Path to local model directory (overrides --model)"
    )
    args = parser.parse_args()
    verify_model(args.model, args.local)
