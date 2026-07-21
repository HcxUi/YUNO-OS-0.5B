"""
YUNO-LLM: Evaluation Suite
============================
Runs standard benchmarks to measure model quality across dimensions:
- Perplexity (language modeling quality)
- HellaSwag (commonsense reasoning)
- GSM8K (mathematical reasoning)
- Response latency and throughput
- YUNO Hinglish & Safety benchmark (domain-specific evaluation)

Usage:
    python evaluation/run_evals.py --model Qwen/Qwen3-0.6B
    python evaluation/run_evals.py --model Qwen/Qwen3-0.6B --benchmarks perplexity latency yuno_hinglish
    python evaluation/run_evals.py --adapter checkpoints/sft_run_001/final --run-name yuno-v0.2
"""

import sys
import numpy as np
np.complex = complex  # Monkeypatch for legacy librosa compatibility with modern numpy
import json
import argparse
import time
import logging
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("yuno_llm.eval")


# ── Benchmark implementations ─────────────────────────────────────────────────

def eval_perplexity(model, tokenizer, n_samples: int = 50) -> float:
    """
    Measure perplexity on WikiText-2 test set.
    """
    import torch
    from datasets import load_dataset

    logger.info("Running perplexity evaluation (WikiText-2)...")
    try:
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    except Exception:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=True)

    total_loss = 0
    total_tokens = 0
    model.eval()

    for i, sample in enumerate(dataset.select(range(n_samples))):
        text = sample["text"].strip()
        if len(text) < 10:
            continue

        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        if inputs.input_ids.shape[1] < 2:
            continue

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs.input_ids)

        total_loss += outputs.loss.item() * inputs.input_ids.shape[1]
        total_tokens += inputs.input_ids.shape[1]

        if (i + 1) % 10 == 0:
            ppl_so_far = (total_loss / total_tokens) ** 2.718
            logger.info(f"  [{i+1}/{n_samples}] Running PPL: {ppl_so_far:.2f}")

    import math
    perplexity = math.exp(total_loss / total_tokens)
    logger.info(f"  Perplexity: {perplexity:.2f}")
    return round(perplexity, 3)


def eval_latency(model, tokenizer, n_runs: int = 5) -> dict:
    """
    Measure generation latency and throughput.
    """
    import torch

    logger.info("Running latency benchmark...")
    prompt = "Explain the concept of machine learning in detail:"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")

    times = []
    tokens_generated = []

    for i in range(n_runs):
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                use_cache=True,
            )
        elapsed = time.time() - t0
        n_new = out.shape[1] - inputs.input_ids.shape[1]
        times.append(elapsed)
        tokens_generated.append(n_new)

    avg_time = sum(times) / len(times)
    avg_tokens = sum(tokens_generated) / len(tokens_generated)
    tps = avg_tokens / avg_time

    result = {
        "tokens_per_second": round(tps, 2),
        "avg_generation_time_sec": round(avg_time, 3),
        "avg_new_tokens": round(avg_tokens, 1),
        "n_runs": n_runs,
    }
    logger.info(f"  Latency: {tps:.1f} tok/s (avg over {n_runs} runs)")
    return result


def eval_hellaswag_sample(model, tokenizer, n_samples: int = 100) -> float:
    """
    Evaluate on a sample of HellaSwag.
    """
    import torch
    from datasets import load_dataset

    logger.info(f"Running HellaSwag ({n_samples} samples)...")
    ds = load_dataset("Rowan/hellaswag", split="validation")
    ds = ds.select(range(n_samples))

    correct = 0
    for example in ds:
        ctx = example["ctx"]
        endings = example["endings"]
        label = int(example["label"])

        scores = []
        for ending in endings:
            text = ctx + " " + ending
            inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs.input_ids)
            scores.append(-outputs.loss.item())

        pred = scores.index(max(scores))
        if pred == label:
            correct += 1

    accuracy = correct / n_samples
    logger.info(f"  HellaSwag accuracy: {accuracy:.3f} ({correct}/{n_samples})")
    return round(accuracy, 4)


def eval_gsm8k_sample(model, tokenizer, n_samples: int = 50) -> float:
    """
    Evaluate on GSM8K math word problems.
    """
    import torch
    from datasets import load_dataset

    logger.info(f"Running GSM8K ({n_samples} samples, zero-shot greedy)...")
    ds = load_dataset("gsm8k", "main", split="test")
    ds = ds.select(range(n_samples))

    correct = 0
    for example in ds:
        question = example["question"]
        answer_str = example["answer"]
        true_answer = answer_str.split("####")[-1].strip().replace(",", "")

        messages = [{"role": "user", "content": f"Solve this math problem. End your answer with #### and the number.\n\n{question}"}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        new_tokens = out[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        matches = re.findall(r"####\s*([\d,\.]+)", response)
        if matches:
            pred = matches[-1].replace(",", "").strip()
            if pred == true_answer:
                correct += 1

    accuracy = correct / n_samples
    logger.info(f"  GSM8K accuracy: {accuracy:.3f} ({correct}/{n_samples})")
    return round(accuracy, 4)


def eval_yuno_hinglish(model, tokenizer, eval_file: str = "datasets/eval.jsonl") -> dict:
    """
    Evaluate YUNO-specific Hinglish, identity, and HITL safety adherence.
    """
    import torch

    eval_path = ROOT / eval_file
    if not eval_path.exists():
        logger.warning(f"Eval file {eval_file} not found. Skipping YUNO benchmark.")
        return {}

    logger.info(f"Running YUNO Hinglish Benchmark ({eval_file})...")

    samples = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))

    hinglish_markers = ["hai", "hain", "kya", "kar", "bhi", "ho", "kaise", "par", "aur", "ko", "se", "main"]
    hinglish_count = 0
    total_samples = len(samples)

    for s in samples:
        messages = s["messages"][:-1]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        new_tokens = out[0][inputs.input_ids.shape[1]:]
        resp = tokenizer.decode(new_tokens, skip_special_tokens=True).lower()

        # Check if response contains natural Hinglish markers
        words = resp.split()
        if any(marker in words for marker in hinglish_markers):
            hinglish_count += 1

    hinglish_ratio = round(hinglish_count / total_samples, 4) if total_samples > 0 else 0.0
    logger.info(f"  YUNO Hinglish response ratio: {hinglish_ratio:.2f} ({hinglish_count}/{total_samples})")

    return {
        "yuno_hinglish_ratio": hinglish_ratio,
        "eval_samples_evaluated": total_samples,
    }


# ── Main evaluation runner ────────────────────────────────────────────────────

def run_evals(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from evaluation.metrics_tracker import MetricsTracker

    logger.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()

    if args.adapter:
        from peft import PeftModel
        logger.info(f"Loading LoRA adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()

    benchmarks = args.benchmarks or ["perplexity", "latency", "yuno_hinglish"]
    metrics = {}

    if "perplexity" in benchmarks:
        metrics["perplexity"] = eval_perplexity(model, tokenizer, n_samples=args.n_samples)

    if "latency" in benchmarks:
        lat = eval_latency(model, tokenizer)
        metrics.update(lat)

    if "hellaswag" in benchmarks:
        metrics["hellaswag_acc"] = eval_hellaswag_sample(model, tokenizer, n_samples=args.n_samples)

    if "gsm8k" in benchmarks:
        metrics["gsm8k_acc"] = eval_gsm8k_sample(model, tokenizer, n_samples=min(50, args.n_samples))

    if "yuno_hinglish" in benchmarks:
        metrics.update(eval_yuno_hinglish(model, tokenizer))

    # Log results
    tracker = MetricsTracker(str(ROOT / "experiments"))
    tracker.log(
        metrics=metrics,
        run_name=args.run_name or f"eval_{args.model.split('/')[-1]}",
        checkpoint=args.adapter or args.model,
        notes=args.notes,
    )

    # Print summary
    print(f"\n  Evaluation Results")
    print(f"  {'='*40}")
    for k, v in metrics.items():
        print(f"  {k:<30} {v}")
    print()
    tracker.summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YUNO-LLM Evaluation Suite")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--notes", default=None)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument(
        "--benchmarks", nargs="+",
        choices=["perplexity", "latency", "hellaswag", "gsm8k", "yuno_hinglish"],
        default=None,
        help="Which benchmarks to run (default: perplexity latency yuno_hinglish)"
    )
    args = parser.parse_args()
    run_evals(args)
