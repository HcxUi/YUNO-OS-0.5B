"""
YUNO-LLM: Evaluation Suite
============================
Runs standard benchmarks to measure model quality across dimensions:
- Perplexity (language modeling quality)
- HellaSwag (commonsense reasoning)
- GSM8K (mathematical reasoning)
- Response latency and throughput

Usage:
    python evaluation/run_evals.py --model Qwen/Qwen3-0.6B
    python evaluation/run_evals.py --model Qwen/Qwen3-0.6B --benchmarks perplexity latency
    python evaluation/run_evals.py --adapter checkpoints/sft_run_001/final --run-name yuno-v0.1
"""

import sys
import numpy as np
np.complex = complex  # Monkeypatch for legacy librosa compatibility with modern numpy
import argparse
import time
import logging
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("yuno_llm.eval")


# ── Benchmark implementations ─────────────────────────────────────────────────

def eval_perplexity(model, tokenizer, n_samples: int = 50) -> float:
    """
    Measure perplexity on WikiText-2 test set.

    Perplexity = exp(average cross-entropy loss).
    Lower is better. Random baseline ≈ vocab_size.
    A good 0.6B model should achieve PPL < 15 on WikiText-2.
    """
    import torch
    from datasets import load_dataset

    logger.info("Running perplexity evaluation (WikiText-2)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

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

    Returns:
        dict with tokens_per_second, time_to_first_token, total_time
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
    Evaluate on a sample of HellaSwag (commonsense NLI completion task).

    For each example, the model scores 4 completions and picks the highest.
    Accuracy = fraction of examples where the correct completion is chosen.

    Note: This is a fast approximation. Full HellaSwag has 10,042 validation examples.
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
            # Lower loss = higher probability
            scores.append(-outputs.loss.item())

        pred = scores.index(max(scores))
        if pred == label:
            correct += 1

    accuracy = correct / n_samples
    logger.info(f"  HellaSwag accuracy: {accuracy:.3f} ({correct}/{n_samples})")
    return round(accuracy, 4)


def eval_gsm8k_sample(model, tokenizer, n_samples: int = 50) -> float:
    """
    Evaluate on GSM8K math word problems (greedy generation, exact match).

    This is a zero-shot evaluation. The model must generate the correct
    numerical answer. Accuracy is very low for small models without fine-tuning.
    """
    import torch
    from datasets import load_dataset
    import re

    logger.info(f"Running GSM8K ({n_samples} samples, zero-shot greedy)...")
    ds = load_dataset("gsm8k", "main", split="test")
    ds = ds.select(range(n_samples))

    correct = 0
    for example in ds:
        question = example["question"]
        answer_str = example["answer"]
        # Extract numerical answer (after ####)
        true_answer = answer_str.split("####")[-1].strip().replace(",", "")

        messages = [{"role": "user", "content": f"Solve this math problem. End your answer with #### and the number.\n\n{question}"}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        new_tokens = out[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Extract numerical answer
        matches = re.findall(r"####\s*([\d,\.]+)", response)
        if matches:
            pred = matches[-1].replace(",", "").strip()
            if pred == true_answer:
                correct += 1

    accuracy = correct / n_samples
    logger.info(f"  GSM8K accuracy: {accuracy:.3f} ({correct}/{n_samples})")
    return round(accuracy, 4)


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

    benchmarks = args.benchmarks or ["perplexity", "latency", "hellaswag"]
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
        choices=["perplexity", "latency", "hellaswag", "gsm8k"],
        default=None,
        help="Which benchmarks to run (default: perplexity latency hellaswag)"
    )
    args = parser.parse_args()
    run_evals(args)
