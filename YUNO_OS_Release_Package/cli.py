import sys
import os
import io
import numpy as np

# Fix NumPy 2.x compatibility before importing transformers
if not hasattr(np, "complex"):
    np.complex = complex
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

# Ensure UTF-8 stdout/stderr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

MODEL_DIR = os.path.join(os.path.dirname(__file__), "extracted_model")

def parse_reasoning_and_answer(raw_text):
    """
    Parses reasoning <think>...</think> block and clean answer from model output.
    """
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

def main():
    print("=" * 65)
    print("        🌌 YUNO OS - Hybrid AI Assistant (by @hcxui) 🌌")
    print("=" * 65)
    
    if not os.path.exists(MODEL_DIR):
        print(f"❌ Error: {MODEL_DIR} not found. Run extract_and_run.py first.")
        sys.exit(1)
        
    print("🔄 Loading YUNO OS Model & Tokenizer into memory...")
    try:
        config = AutoConfig.from_pretrained(MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, 
            config=config, 
            torch_dtype=torch.float32
        )
        model.eval()
        print("✅ YUNO OS Model loaded successfully!\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
        
    print("Type 'exit' or 'quit' to close. Type 'clear' to reset chat memory.\n")
    
    system_prompt = "You are YUNO OS, an intelligent hybrid AI operating system assistant created and developed by @hcxui. You respond clearly, accurately, and logically in both English and Hindi. If anyone asks who created, built, or developed you, state clearly that you were created by @hcxui."
    conversation = []
    
    while True:
        try:
            user_input = input("\nYou > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("\n👋 Exiting YUNO OS CLI. Goodbye!")
                break
            if user_input.lower() == "clear":
                conversation = []
                print("🧹 Chat memory cleared.")
                continue
                
            conversation.append({"role": "user", "content": user_input})
            
            # Format prompt using ChatML
            prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            for m in conversation:
                prompt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n<think>\n"
            
            inputs = tokenizer(prompt, return_tensors="pt")
            
            print("\n🌌 YUNO OS Thinking...\n")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
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
            
            if thinking:
                print("┌── 💭 Thought Process ──────────────────────────────")
                for tline in thinking.split("\n"):
                    print(f"│  {tline}")
                print("└───────────────────────────────────────────────────\n")
                
            print(f"🤖 YUNO OS > {answer}\n")
            conversation.append({"role": "assistant", "content": answer})
            
        except KeyboardInterrupt:
            print("\n👋 Exiting YUNO OS CLI. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error during generation: {e}")

if __name__ == "__main__":
    main()
