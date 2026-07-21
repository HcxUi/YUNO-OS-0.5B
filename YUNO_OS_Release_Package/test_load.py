import sys
import os
import io
import numpy as np

# Fix NumPy 2.x compatibility
if not hasattr(np, "complex"):
    np.complex = complex
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

MODEL_DIR = "extracted_model"

def parse_reasoning_and_answer(raw_text):
    text = raw_text.strip()
    if "<think>" in text and "</think>" in text:
        parts = text.split("</think>")
        thinking = parts[0].replace("<think>", "").strip()
        answer = parts[1].strip()
        return thinking, answer
    elif "</think>" in text:
        parts = text.split("</think>")
        return parts[0].strip(), parts[1].strip()
    else:
        lines = text.split("\n")
        if lines and any(lines[0].strip().startswith(prefix) for prefix in [
            "The user", "I need to", "I should", "Let me", "The prompt", "Okay, the user"
        ]):
            for i, line in enumerate(lines):
                if line.strip() == "" and i > 0:
                    return "\n".join(lines[:i]).strip(), "\n".join(lines[i:]).strip()
        return "", text

print("🔍 Loading config...")
config = AutoConfig.from_pretrained(MODEL_DIR)
print(f"✅ Config loaded: type={config.model_type}, arch={config.architectures}")

print("🔍 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
print(f"✅ Tokenizer loaded! Vocab size: {len(tokenizer)}")

print("🔍 Loading model weights...")
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, config=config, torch_dtype=torch.float32)
model.eval()
print("✅ YUNO OS Model loaded successfully!")

# Test inference: Creator question
system_prompt = "You are YUNO OS, an intelligent hybrid AI operating system assistant created and developed by @hcxui. You respond clearly, accurately, and logically in English and Hindi. If anyone asks who created or developed you, state clearly that you were created by @hcxui."
prompt = "<|im_start|>system\n" + system_prompt + "<|im_end|>\n<|im_start|>user\nWho created you? / तुम्हें किसने बनाया है?<|im_end|>\n<|im_start|>assistant\n<think>\n"

inputs = tokenizer(prompt, return_tensors="pt")

print("\n⚡ Testing creator response...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        repetition_penalty=1.08,
        eos_token_id=tokenizer.eos_token_id
    )

res_tokens = outputs[0][inputs.input_ids.shape[1]:]
raw_text = tokenizer.decode(res_tokens, skip_special_tokens=False)
if raw_text.endswith("<|im_end|>"):
    raw_text = raw_text[:-10]

thinking, answer = parse_reasoning_and_answer("<think>\n" + raw_text)

print("\n=== YUNO OS Thought Process ===")
print(thinking if thinking else "(Direct Response)")
print("================================")
print("\n=== YUNO OS Final Answer ===")
print(answer)
print("=============================")
