"""
YUNO-LLM: Supervised Fine-Tuning (SFT) with LoRA
===================================================
Full training script using TRL's SFTTrainer with PEFT/LoRA.

Pipeline:
    Dataset → Tokenizer → DataLoader → Forward Pass →
    Loss → Backpropagation → Optimizer → Checkpoint → Evaluation

Usage:
    # CPU (development/small experiment):
    python training/train_sft.py --config config/training_config.yaml --train datasets/train.jsonl

    # GPU (production):
    python training/train_sft.py --config config/training_config.yaml --train datasets/train.jsonl --bf16

    # With 4-bit quantization (QLoRA):
    python training/train_sft.py --config config/training_config.yaml --train datasets/train.jsonl --4bit

Before running:
    1. Download the base model: python scripts/download_model.py
    2. Prepare your dataset in JSONL format (see training/dataset_prep.py)
    3. Adjust config/training_config.yaml
"""

import sys
import numpy as np
np.complex = complex  # Monkeypatch for legacy librosa compatibility with modern numpy
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("yuno_llm.train")


def setup_experiment_dir(output_dir: str) -> Path:
    """Create timestamped experiment directory."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = ROOT / "experiments" / f"sft_{ts}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def train(args):
    import torch
    import yaml
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
    )
    from trl import SFTTrainer, SFTConfig
    from training.lora_config import get_lora_config, apply_lora
    from training.dataset_prep import prepare_dataset

    # ── Load training config ──────────────────────────────────────────────────
    with open(args.config, "r") as f:
        train_cfg = yaml.safe_load(f)

    main_cfg_path = ROOT / "config" / "yuno_config.yaml"
    with open(main_cfg_path, "r") as f:
        main_cfg = yaml.safe_load(f)

    model_id = main_cfg["base_model"]["name"]
    t_cfg = train_cfg["training"]
    lora_cfg_data = train_cfg.get("lora", {})

    logger.info(f"YUNO-LLM SFT Training")
    logger.info(f"Base model: {model_id}")
    logger.info(f"Train file: {args.train}")

    # ── Load tokenizer ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Prepare dataset ───────────────────────────────────────────────────────
    train_ds, eval_ds = prepare_dataset(
        train_file=args.train,
        eval_file=args.eval,
        model_id=model_id,
        max_seq_length=t_cfg.get("max_seq_length", 2048),
    )

    # ── Load model ────────────────────────────────────────────────────────────
    load_kwargs = {
        "trust_remote_code": True,
        "device_map": "cpu" if not torch.cuda.is_available() else "auto",
    }

    quant_cfg = train_cfg.get("quantization", {})
    if args.use_4bit or quant_cfg.get("load_in_4bit", False):
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["quantization_config"] = bnb
        load_kwargs["torch_dtype"] = torch.bfloat16
        logger.info("QLoRA mode: loading in 4-bit")
    elif args.bf16 or t_cfg.get("bf16", False):
        load_kwargs["torch_dtype"] = torch.bfloat16
    elif t_cfg.get("fp16", False):
        load_kwargs["torch_dtype"] = torch.float16
    else:
        load_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.config.use_cache = False  # Required for gradient checkpointing

    # ── Apply LoRA ────────────────────────────────────────────────────────────
    lora_preset = args.lora_preset or "attention_only"
    lora_config = get_lora_config(lora_preset)

    # Override with YAML config if present
    if lora_cfg_data.get("r"):
        lora_config.r = lora_cfg_data["r"]
        lora_config.lora_alpha = lora_cfg_data.get("lora_alpha", lora_config.lora_alpha)
        lora_config.lora_dropout = lora_cfg_data.get("lora_dropout", lora_config.lora_dropout)
        lora_config.target_modules = lora_cfg_data.get("target_modules", lora_config.target_modules)

    model = apply_lora(model, lora_config)

    # ── Training arguments ────────────────────────────────────────────────────
    output_dir = args.output_dir or t_cfg.get("output_dir", "checkpoints/sft_run")
    exp_dir = setup_experiment_dir(output_dir)

    training_args = SFTConfig(
        output_dir=str(ROOT / output_dir),
        num_train_epochs=t_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=t_cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=t_cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=t_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=t_cfg.get("learning_rate", 2e-4),
        weight_decay=t_cfg.get("weight_decay", 0.01),
        warmup_ratio=t_cfg.get("warmup_ratio", 0.03),
        lr_scheduler_type=t_cfg.get("lr_scheduler_type", "cosine"),
        max_seq_length=t_cfg.get("max_seq_length", 2048),
        fp16=args.fp16 or t_cfg.get("fp16", False),
        bf16=args.bf16 or t_cfg.get("bf16", False),
        logging_steps=t_cfg.get("logging_steps", 10),
        eval_strategy="steps",
        eval_steps=t_cfg.get("eval_steps", 100),
        save_steps=t_cfg.get("save_steps", 200),
        save_total_limit=t_cfg.get("save_total_limit", 3),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=t_cfg.get("report_to", "none"),
        dataloader_num_workers=t_cfg.get("dataloader_num_workers", 0),
        dataset_text_field="text",  # field name after tokenization
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )

    # ── Log experiment metadata ───────────────────────────────────────────────
    metadata = {
        "start_time": datetime.now().isoformat(),
        "base_model": model_id,
        "lora_preset": lora_preset,
        "lora_r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "lora_targets": lora_config.target_modules,
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds),
        "epochs": t_cfg.get("num_train_epochs", 3),
        "batch_size": t_cfg.get("per_device_train_batch_size", 2),
        "learning_rate": t_cfg.get("learning_rate", 2e-4),
    }
    with open(exp_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Experiment metadata saved to {exp_dir}/metadata.json")

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Starting training...")
    trainer.train()

    # ── Save final model ──────────────────────────────────────────────────────
    final_path = ROOT / output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    logger.info(f"✅ Training complete. Model saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YUNO-LLM SFT Training")
    parser.add_argument("--config", default="config/training_config.yaml")
    parser.add_argument("--train", required=True, help="Path to training JSONL")
    parser.add_argument("--eval", default=None, help="Path to eval JSONL")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--lora-preset", default="attention_only",
                        choices=["attention_only", "full", "minimal", "high_rank"])
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--4bit", dest="use_4bit", action="store_true",
                        help="QLoRA: load in 4-bit (requires CUDA + bitsandbytes)")
    args = parser.parse_args()
    train(args)
