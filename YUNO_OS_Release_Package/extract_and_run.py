import os
import json
import shutil
import sys
import io
import numpy as np

# Ensure UTF-8 output encoding for Windows terminal
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Fix NumPy 2.x compatibility before importing transformers/librosa
if not hasattr(np, "complex"):
    np.complex = complex
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int

RELEASE_DIR = "YUNO_OS_Release_Package"
MODEL_FILE = os.path.join(RELEASE_DIR, "yuno-os-v0.5.0.yuno") if os.path.exists(os.path.join(RELEASE_DIR, "yuno-os-v0.5.0.yuno")) else "yuno-os-v0.5.0.yuno"
MODEL_DIR = "extracted_model"

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if loop.first and message['role'] != 'system' %}"
    "<|im_start|>system\nYou are YUNO OS, an intelligent hybrid AI operating system assistant created and developed by @hcxui. If anyone asks who created or developed you, state clearly that you were created by @hcxui.<|im_end|>\n"
    "{% endif %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n{% endif %}"
)

XOR_KEY = 0x77

def extract_yuno_bin(bin_path, target_dir):
    print(f"[*] Unpacking official YUNO OS model '{bin_path}' into '{target_dir}'...")
    os.makedirs(target_dir, exist_ok=True)
    
    with open(bin_path, "rb") as f:
        magic = f.read(4)
        if magic != b"YUNO":
            raise ValueError(f"Invalid magic header: {magic}. Expected b'YUNO'.")
        
        length = int.from_bytes(f.read(4), "little")
        header_raw = f.read(length).decode("utf-8")
        header = json.loads(header_raw)
        
        print(f"[+] Model Name: {header.get('name', 'YUNO-OS-0.5B-Instruct')}")
        print(f"[+] Architecture: {header.get('architecture', 'YunoForCausalLM')}")
        print(f"[+] Format: {header.get('format', 'YUNO_LLM_PROPRIETARY_V1')}")
        print(f"[+] Creator / Developer: @hcxui")
        if "sha256_signature" in header:
            print(f"[+] SHA-256 Integrity Hash: {header['sha256_signature'][:16]}...")
        if header.get("anti_tamper"):
            print(f"[+] Anti-Tamper Protection: Active (XOR 0x77 Cipher Stream)")
        
        # Save config.json with clean architecture
        config = header["config"]
        with open(os.path.join(target_dir, "config.json"), "w", encoding="utf-8") as cfg_f:
            json.dump(config, cfg_f, indent=2)
        print("  - Saved config.json")
            
        # Save generation_config.json
        gen_config = header.get("generation_config", {
            "bos_token_id": 151643,
            "eos_token_id": [151645, 151643],
            "pad_token_id": 151643,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20
        })
        with open(os.path.join(target_dir, "generation_config.json"), "w", encoding="utf-8") as gen_f:
            json.dump(gen_config, gen_f, indent=2)
        print("  - Saved generation_config.json")
                
        # Save tokenizer.json
        if "tokenizer" in header:
            with open(os.path.join(target_dir, "tokenizer.json"), "w", encoding="utf-8") as tok_f:
                json.dump(header["tokenizer"], tok_f, indent=2)
            print("  - Saved tokenizer.json")

        # Save tokenizer_config.json
        tokenizer_config = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "bos_token": "<|endoftext|>",
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
            "chat_template": CHAT_TEMPLATE,
            "model_max_length": config.get("max_position_embeddings", 40960),
            "clean_up_tokenization_spaces": False,
            "added_tokens_decoder": {
                "151643": {"content": "<|endoftext|>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
                "151644": {"content": "<|im_start|>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
                "151645": {"content": "<|im_end|>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
                "151667": {"content": "<think>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
                "151668": {"content": "</think>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True}
            }
        }
        with open(os.path.join(target_dir, "tokenizer_config.json"), "w", encoding="utf-8") as tc_f:
            json.dump(tokenizer_config, tc_f, indent=2)
        print("  - Saved tokenizer_config.json")

        # Extract & Decrypt weights binary stream
        weights_path = os.path.abspath(os.path.join(target_dir, "pytorch_model.bin"))
        print(f"  - Decrypting & Extracting model weights binary to '{weights_path}'...")
        
        is_anti_tamper = header.get("anti_tamper", False)
        
        # Open output file in binary write mode
        out_weights = open(weights_path, "wb")
        try:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                if is_anti_tamper:
                    unmasked_chunk = bytes([b ^ XOR_KEY for b in chunk])
                    out_weights.write(unmasked_chunk)
                else:
                    out_weights.write(chunk)
        finally:
            out_weights.close()
            
        print(f"  [+] Weights decrypted & extracted successfully: {os.path.getsize(weights_path):,} bytes")

if __name__ == "__main__":
    target_path = MODEL_FILE if os.path.exists(MODEL_FILE) else "yuno-os-v0.5.0.yuno"
    if not os.path.exists(target_path):
        target_path = "yuno-v0.5.0-single.bin"
        
    extract_yuno_bin(target_path, MODEL_DIR)
    print("\n[SUCCESS] YUNO OS Model Extracted Successfully! Creator: @hcxui")
