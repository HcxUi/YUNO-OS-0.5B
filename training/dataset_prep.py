"""
YUNO-LLM: Dataset Preparation
================================
Loads, formats, deduplicates, and tokenizes datasets for SFT training.

Expected input format (JSONL):
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Or ShareGPT format:
    {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}

Usage:
    python training/dataset_prep.py --train datasets/train.jsonl --eval datasets/eval.jsonl --dedup --shuffle
"""

import sys
import json
import random
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def convert_sharegpt_to_messages(sample: dict) -> dict:
    """Convert ShareGPT format to OpenAI messages format."""
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    messages = []
    for turn in sample.get("conversations", []):
        role = role_map.get(turn.get("from", ""), "user")
        messages.append({"role": role, "content": turn.get("value", "")})
    return {"messages": messages}


def deduplicate_samples(samples: List[dict]) -> List[dict]:
    """
    Remove duplicate conversation samples by computing an MD5 hash
    over all turn roles and contents.
    """
    unique_samples = []
    seen_hashes = set()

    for s in samples:
        msg_str = json.dumps(s.get("messages", []), sort_keys=True)
        h = hashlib.md5(msg_str.encode("utf-8")).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_samples.append(s)

    return unique_samples


def load_jsonl(path: str, dedup: bool = True) -> List[dict]:
    """Load a JSONL file into a list of dicts with optional deduplication."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sample = json.loads(line)
                if "conversations" in sample and "messages" not in sample:
                    sample = convert_sharegpt_to_messages(sample)
                samples.append(sample)

    if dedup:
        before_cnt = len(samples)
        samples = deduplicate_samples(samples)
        after_cnt = len(samples)
        if before_cnt != after_cnt:
            print(f"  [Dedup] Removed {before_cnt - after_cnt} duplicate samples ({before_cnt} -> {after_cnt})")

    return samples


def format_sample(sample: dict, tokenizer) -> str:
    """
    Apply the chat template to a single sample.
    Returns the fully formatted string ready for tokenization.
    """
    messages = sample["messages"]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def prepare_dataset(
    train_file: Optional[str] = None,
    eval_file: Optional[str] = None,
    hf_dataset: Optional[str] = None,
    model_id: str = "Qwen/Qwen3-0.6B",
    max_seq_length: int = 2048,
    train_split_ratio: float = 0.95,
    dedup: bool = True,
    shuffle: bool = True,
    seed: int = 42,
):
    """
    Main dataset preparation pipeline.
    """
    from transformers import AutoTokenizer
    from datasets import Dataset

    print(f"\n  YUNO-LLM — Dataset Preparation")
    print(f"  {'='*40}")

    random.seed(seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print(f"  Tokenizer: {model_id} (vocab={tokenizer.vocab_size:,})")

    if hf_dataset:
        from datasets import load_dataset
        print(f"  Loading HF dataset: {hf_dataset}")
        ds = load_dataset(hf_dataset)
        train_data = list(ds["train"])
        eval_data = list(ds.get("test", ds.get("validation", [])))
    elif train_file:
        print(f"  Loading local train: {train_file}")
        train_data = load_jsonl(train_file, dedup=dedup)
        if eval_file:
            print(f"  Loading local eval:  {eval_file}")
            eval_data = load_jsonl(eval_file, dedup=dedup)
        else:
            if shuffle:
                random.shuffle(train_data)
            split_idx = int(len(train_data) * train_split_ratio)
            train_data, eval_data = train_data[:split_idx], train_data[split_idx:]
            print(f"  Auto-split: {len(train_data)} train, {len(eval_data)} eval")
    else:
        print("  ❌ Must provide --train or --hf-dataset")
        sys.exit(1)

    if shuffle and eval_file:
        random.shuffle(train_data)
        random.shuffle(eval_data)

    print(f"  Final Samples — train: {len(train_data):,}, eval: {len(eval_data):,}")

    def tokenize_fn(sample):
        """Format and tokenize a single sample."""
        text = format_sample(sample, tokenizer)
        result = tokenizer(
            text,
            max_length=max_seq_length,
            truncation=True,
            padding=False,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    print(f"  Tokenizing (max_seq_length={max_seq_length})...")
    train_ds = Dataset.from_list(train_data).map(tokenize_fn, batched=False, remove_columns=["messages"])
    eval_ds = Dataset.from_list(eval_data).map(tokenize_fn, batched=False, remove_columns=["messages"])

    lengths = [len(x["input_ids"]) for x in train_ds]
    print(f"  Token length — mean: {sum(lengths)/len(lengths):.0f}, max: {max(lengths)}, min: {min(lengths)}")
    print(f"  [OK] Dataset ready.")
    return train_ds, eval_ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=None, help="Path to train JSONL file")
    parser.add_argument("--eval", default=None, help="Path to eval JSONL file")
    parser.add_argument("--hf-dataset", default=None, help="HuggingFace dataset ID")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--no-dedup", action="store_true", help="Disable deduplication")
    parser.add_argument("--no-shuffle", action="store_true", help="Disable shuffling")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_dataset(
        train_file=args.train,
        eval_file=args.eval,
        hf_dataset=args.hf_dataset,
        model_id=args.model,
        max_seq_length=args.max_seq_length,
        dedup=not args.no_dedup,
        shuffle=not args.no_shuffle,
        seed=args.seed,
    )
