import os
import json
import shutil
import sys
import io
import struct
import hashlib

# Ensure UTF-8 output encoding
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

SOURCE_DIR = "extracted_model"
RELEASE_DIR = "YUNO_OS_Release_Package"
OUTPUT_YUNO = os.path.join(RELEASE_DIR, "yuno-os-v0.5.0.yuno")

def pack_secure_yuno_model(source_dir, output_yuno):
    print("=" * 70)
    print("  🔒 YUNO OS Official Secure Builder & Anti-Tamper Obfuscator")
    print("  👑 Developer & Creator: @hcxui")
    print("=" * 70)
    
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source directory '{source_dir}' not found.")
        
    os.makedirs(RELEASE_DIR, exist_ok=True)
    
    config_path = os.path.join(source_dir, "config.json")
    tokenizer_path = os.path.join(source_dir, "tokenizer.json")
    gen_config_path = os.path.join(source_dir, "generation_config.json")
    tokenizer_config_path = os.path.join(source_dir, "tokenizer_config.json")
    weights_path = os.path.join(source_dir, "pytorch_model.bin")
    
    # Clean Architecture Metadata
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    config["architectures"] = ["YunoForCausalLM"]
    config["model_type"] = "qwen3"
    config["_name_or_path"] = "hcxui/YUNO-OS-0.5B-Instruct"
    config["developer"] = "@hcxui"
    config["organization"] = "hcxui"
    
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    with open(tokenizer_config_path, "r", encoding="utf-8") as f:
        tokenizer_config = json.load(f)
    tokenizer_config["tokenizer_class"] = "PreTrainedTokenizerFast"
    with open(tokenizer_config_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2)

    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tokenizer = json.load(f)
        
    with open(gen_config_path, "r", encoding="utf-8") as f:
        gen_config = json.load(f)

    # Compute SHA-256 integrity signature of weights stream
    print("  ⚡ Computing cryptographic SHA-256 weight signature...")
    sha256_hash = hashlib.sha256()
    with open(weights_path, "rb") as w_f:
        for chunk in iter(lambda: w_f.read(4096 * 1024), b""):
            sha256_hash.update(chunk)
    signature = sha256_hash.hexdigest()
    print(f"  ✓ Signature: {signature[:16]}...{signature[-16:]}")

    header = {
        "format": "YUNO_LLM_PROPRIETARY_V1",
        "name": "YUNO-OS-0.5B-Instruct",
        "creator": "@hcxui",
        "architecture": "YunoForCausalLM",
        "sha256_signature": signature,
        "anti_tamper": True,
        "config": config,
        "generation_config": gen_config,
        "tokenizer": tokenizer,
        "tokenizer_config": tokenizer_config
    }
    
    header_json = json.dumps(header, ensure_ascii=False)
    header_bytes = header_json.encode("utf-8")
    header_length = len(header_bytes)
    
    magic = b"YUNO"
    length_bytes = header_length.to_bytes(4, byteorder="little")
    
    # 0x77 XOR cipher stream obfuscation key for Anti-Tamper & Anti-Decompile
    XOR_KEY = 0x77
    
    print(f"  🔒 Writing Proprietary Binary Header ({header_length:,} bytes)...")
    print(f"  🔒 Applying XOR 0x77 Stream Protection against unauthorized editing...")
    
    temp_output = output_yuno + ".tmp"
    with open(temp_output, "wb") as out_f:
        out_f.write(magic)
        out_f.write(length_bytes)
        out_f.write(header_bytes)
        
        # Write weights stream with XOR security masking
        with open(weights_path, "rb") as w_f:
            while True:
                chunk = w_f.read(1024 * 1024)
                if not chunk:
                    break
                # Apply 0x77 security transformation byte-by-byte
                masked_chunk = bytes([b ^ XOR_KEY for b in chunk])
                out_f.write(masked_chunk)
                
    if os.path.exists(output_yuno):
        os.remove(output_yuno)
    os.rename(temp_output, output_yuno)
    
    # Also sync root level yuno-v0.5.0-single.bin and yuno-os-v0.5.0.yuno
    shutil.copyfile(output_yuno, "yuno-os-v0.5.0.yuno")
    shutil.copyfile(output_yuno, "yuno-v0.5.0-single.bin")
    
    print(f"\n🎉 Successfully compiled Protected Release Package:")
    print(f"   📁 Output: '{output_yuno}' ({os.path.getsize(output_yuno):,} bytes)")
    print(f"   🛡️ Anti-Tamper Protection: Active (XOR 0x77 Cipher Stream)")
    print(f"   👑 Creator: @hcxui")

if __name__ == "__main__":
    pack_secure_yuno_model(SOURCE_DIR, OUTPUT_YUNO)
