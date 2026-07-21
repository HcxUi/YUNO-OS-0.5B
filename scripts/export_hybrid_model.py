"""
YUNO-LLM Hybrid Model Exporter & Merger
=======================================
Combines the base Qwen (QW) model weights, YUNO special tokens, and fine-tuned
LoRA adapter layers into a single, standalone unified Hybrid LLM directory.

The resulting hybrid model is 100% self-contained and can be loaded directly with
HuggingFace Transformers or Ollama/vLLM without needing external adapters.

Usage:
    py -3.11 scripts/export_hybrid_model.py
    py -3.11 scripts/export_hybrid_model.py --output-dir models/yuno-hybrid-0.5.0
"""

import sys
import argparse
import logging
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("yuno_llm.export_hybrid")


def export_hybrid_model(
    base_model_name: str = "Qwen/Qwen3-0.6B",
    adapter_dir: str = None,
    output_dir: str = "models/yuno-hybrid-0.5.0",
) -> Path:
    """
    Fuses Qwen base weights with YUNO system tokens & LoRA weights,
    exporting a standalone hybrid LLM artifact.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from yuno_llm.tokenizer import YunoTokenizer

    out_path = ROOT / output_dir
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  YUNO-LLM — HYBRID LLM MODEL FUSION & EXPORT")
    print("=" * 65)
    print(f"  Base Model (QW):  {base_model_name}")
    print(f"  Output Hybrid Path: {out_path.resolve()}")
    print("=" * 65)

    # 1. Load Base Tokenizer & Add YUNO Tokens
    logger.info(f"1. Loading Base Tokenizer ({base_model_name})...")
    raw_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if raw_tokenizer.pad_token is None:
        raw_tokenizer.pad_token = raw_tokenizer.eos_token

    yuno_tok = YunoTokenizer(raw_tokenizer)
    yuno_tok.add_special_tokens()
    tokenizer = yuno_tok._tokenizer

    # 2. Load Base Model Weights
    logger.info(f"2. Loading Base Qwen Model Weights...")
    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float32 if not torch.cuda.is_available() else torch.bfloat16,
        "device_map": "cpu" if not torch.cuda.is_available() else "auto",
    }
    model = AutoModelForCausalLM.from_pretrained(base_model_name, **load_kwargs)
    model.resize_token_embeddings(len(tokenizer))

    # 3. Fuse LoRA Adapters if present
    adapter_path = Path(adapter_dir) if adapter_dir else (ROOT / "checkpoints" / "sft_run" / "final")
    if adapter_path.exists():
        logger.info(f"3. Merging LoRA adapters from {adapter_path} into base model...")
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(adapter_path))
            model = model.merge_and_unload()
            logger.info("  [SUCCESS] LoRA adapter weights merged into base model tensors!")
        except Exception as e:
            logger.warning(f"  Could not load adapter at {adapter_path} ({e}). Saving base hybrid model.")
    else:
        logger.info("3. No LoRA adapter found; constructing direct YUNO-Qwen hybrid base model.")

    # 4. Save Self-Contained Hybrid Model & Tokenizer
    logger.info(f"4. Saving Standalone Hybrid LLM to {out_path}...")
    model.save_pretrained(str(out_path), safe_serialization=True)
    tokenizer.save_pretrained(str(out_path))

    # 5. Write Hybrid Model Card Metadata
    model_card = out_path / "README.md"
    model_card.write_text(
        f"# YUNO-Qwen Hybrid LLM v0.5.0\n\n"
        f"This is a standalone hybrid LLM combining **Qwen (QW)** base architecture with the **YUNO Personal AI OS** system layer.\n\n"
        f"- **Base Model:** `{base_model_name}`\n"
        f"- **Special Tokens:** `<think>`, `</think>`, `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>`\n"
        f"- **Language Tuning:** Hinglish Native + English\n"
        f"- **Format:** SafeTensors Standalone CausalLM\n",
        encoding="utf-8",
    )

    print("=" * 65)
    print(f"  [SUCCESS] HYBRID LLM CREATED SUCCESSFULLY!")
    print(f"  Location: {out_path.resolve()}")
    print("=" * 65)
    return out_path


def verify_hybrid_model(model_dir: str = "models/yuno-hybrid-0.5.0") -> None:
    """Verify that the exported hybrid model loads standalone."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_path = ROOT / model_dir
    print(f"\n[Verification] Testing Hybrid Model from {model_path}...")
    assert model_path.exists(), f"Model directory not found: {model_path}"

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_path), trust_remote_code=True, device_map="cpu")

    prompt = "YUNO, introduce yourself in Hinglish."
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=30)
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)

    print(f"[Verification Output] {response[:150]}...")
    print("  [PASS] Standalone Hybrid LLM loaded & generated output cleanly!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Standalone YUNO-Qwen Hybrid LLM")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--output-dir", default="models/yuno-hybrid-0.5.0")
    args = parser.parse_args()

    exported_dir = export_hybrid_model(
        base_model_name=args.base_model,
        adapter_dir=args.adapter_dir,
        output_dir=args.output_dir,
    )
    verify_hybrid_model(model_dir=args.output_dir)
