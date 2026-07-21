"""
YUNO Single-File LLM Model Exporter
===================================
Exports the complete YUNO Hybrid LLM into a SINGLE unified model file (.safetensors / .bin / .gguf).
All weights, config parameters, and tokenizer vocab are bundled into a single file format.

Usage:
    py -3.11 scripts/convert_single_file_model.py
"""

import sys
import json
import argparse
import struct
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


def export_single_file_llm(
    source_dir: str = "models/yuno-hybrid-0.5.0",
    output_file: str = "models/yuno-v0.5.0-single.bin",
) -> Path:
    """
    Combines model config, tokenizer, and tensor weights into a SINGLE unified file format (.bin).
    """
    import torch
    from safetensors.torch import load_file

    src_path = ROOT / source_dir
    out_path = ROOT / output_file
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  YUNO-LLM — SINGLE FILE LLM COMPILATION & PACKAGING")
    print("=" * 65)
    print(f"  Source Model Directory: {src_path.resolve()}")
    print(f"  Target Single Model File: {out_path.resolve()}")
    print("=" * 65)

    # 1. Read metadata files (config.json, tokenizer.json, generation_config.json)
    config_data = json.loads((src_path / "config.json").read_text(encoding="utf-8")) if (src_path / "config.json").exists() else {}
    tokenizer_data = json.loads((src_path / "tokenizer.json").read_text(encoding="utf-8")) if (src_path / "tokenizer.json").exists() else {}
    gen_config_data = json.loads((src_path / "generation_config.json").read_text(encoding="utf-8")) if (src_path / "generation_config.json").exists() else {}

    meta_header = {
        "format": "YUNO_SINGLE_FILE_LLM_V1",
        "name": "yuno-hybrid-0.5.0",
        "config": config_data,
        "tokenizer": tokenizer_data,
        "generation_config": gen_config_data,
    }
    header_bytes = json.dumps(meta_header, ensure_ascii=False).encode("utf-8")

    # 2. Load Safetensors weight tensors
    weights_file = src_path / "model.safetensors"
    if not weights_file.exists():
        weights_file = src_path / "pytorch_model.bin"

    print("  [1/3] Reading model tensor weights...")
    if weights_file.suffix == ".safetensors":
        tensors_dict = load_file(str(weights_file))
    else:
        tensors_dict = torch.load(str(weights_file), map_location="cpu")

    # 3. Write SINGLE BINARY FILE (.bin) with header length + JSON header + torch weight dictionary
    print(f"  [2/3] Packing weights ({len(tensors_dict)} tensors) & config into single file...")
    with open(out_path, "wb") as f:
        # Magic bytes identifier
        f.write(b"YUNO")
        # Header size (uint32)
        f.write(struct.pack("<I", len(header_bytes)))
        # JSON header bytes
        f.write(header_bytes)
        # Weight dictionary serialized with torch
        torch.save(tensors_dict, f)

    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  [3/3] Single model file created cleanly! Total Size: {file_size_mb:.2f} MB")

    print("=" * 65)
    print(f"  [SUCCESS] SINGLE FILE LLM COMPILED!")
    print(f"  Single File Path: {out_path.resolve()}")
    print("=" * 65)
    return out_path


def verify_single_file_llm(single_file_path: str = "models/yuno-v0.5.0-single.bin") -> None:
    """Verify loading and reading from the SINGLE compiled model file."""
    import torch

    file_path = ROOT / single_file_path
    print(f"\n[Verification] Unpacking & Testing Single File LLM: {file_path}...")
    assert file_path.exists(), f"Single model file not found: {file_path}"

    with open(file_path, "rb") as f:
        magic = f.read(4)
        assert magic == b"YUNO", "Invalid magic header bytes!"
        header_len = struct.unpack("<I", f.read(4))[0]
        header_json = json.loads(f.read(header_len).decode("utf-8"))
        tensors = torch.load(f, map_location="cpu")

    print(f"  - Format:            {header_json['format']}")
    print(f"  - Model Name:        {header_json['name']}")
    print(f"  - Hidden Size:       {header_json['config'].get('hidden_size', 'N/A')}")
    print(f"  - Number of Layers:  {header_json['config'].get('num_hidden_layers', 'N/A')}")
    print(f"  - Tensors Unpacked:  {len(tensors)} tensors")
    print("  [PASS] Single File LLM verified successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Single File Unified YUNO LLM")
    parser.add_argument("--source-dir", default="models/yuno-hybrid-0.5.0")
    parser.add_argument("--output-file", default="models/yuno-v0.5.0-single.bin")
    args = parser.parse_args()

    single_file = export_single_file_llm(
        source_dir=args.source_dir,
        output_file=args.output_file,
    )
    verify_single_file_llm(single_file_path=args.output_file)
