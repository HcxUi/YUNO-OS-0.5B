"""
YUNO-LLM Model Wrapper
========================
YunoForCausalLM wraps Qwen3ForCausalLM with:
- YUNO-LLM identity and versioning
- Improved logging and diagnostics
- Helper methods for parameter inspection
- Hook points for future architectural modifications

The underlying Qwen3 architecture is NOT changed in this version.
All modifications are additive wrappers and metadata.
"""

from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any

import torch

logger = logging.getLogger("yuno_llm.model")


class YunoForCausalLM:
    """
    YUNO-LLM causal language model.

    A thin wrapper around Qwen3ForCausalLM that adds:
    - YUNO identity metadata
    - Convenience methods for inspection
    - Logging hooks
    - Future: architectural modifications go here

    Usage:
        model = YunoForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
        print(model.info())
    """

    YUNO_VERSION = "0.1.0"

    def __init__(self, model, config=None):
        self._model = model
        self._config = config
        self._load_time = time.time()

        if config and config.logging.log_model_info:
            self._log_model_info()

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        config=None,
        device_map: str = "cpu",
        torch_dtype=None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        **kwargs,
    ) -> "YunoForCausalLM":
        """
        Load Qwen3 model and wrap as YunoForCausalLM.

        Args:
            model_id: HuggingFace model ID or local path
            config: YunoConfig instance
            device_map: "cpu" | "cuda" | "auto"
            torch_dtype: torch.float32 | torch.float16 | torch.bfloat16
            load_in_4bit: Enable 4-bit quantization (requires bitsandbytes + GPU)
            load_in_8bit: Enable 8-bit quantization (requires bitsandbytes + GPU)
        """
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        logger.info(f"Loading model: {model_id}")
        t0 = time.time()

        # Resolve dtype
        if torch_dtype is None:
            if config and config.hardware.dtype == "bfloat16":
                torch_dtype = torch.bfloat16
            elif config and config.hardware.dtype == "float16":
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32

        # Quantization config
        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif load_in_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
            quantization_config=bnb_config,
            **kwargs,
        )

        elapsed = time.time() - t0
        logger.info(f"Model loaded in {elapsed:.1f}s | dtype={torch_dtype} | device={device_map}")

        return cls(model, config)

    def _log_model_info(self) -> None:
        """Log model architecture summary."""
        total = self.total_parameters()
        trainable = self.trainable_parameters()
        logger.info(
            f"YUNO-LLM v{self.YUNO_VERSION} | "
            f"Parameters: {total:,} total, {trainable:,} trainable | "
            f"Base: {getattr(self._model.config, 'model_type', 'unknown')}"
        )

    def total_parameters(self) -> int:
        """Total number of model parameters."""
        return sum(p.numel() for p in self._model.parameters())

    def trainable_parameters(self) -> int:
        """Number of trainable parameters (unfrozen)."""
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)

    def info(self) -> str:
        """Return a formatted model info string."""
        total = self.total_parameters()
        trainable = self.trainable_parameters()
        cfg = self._model.config
        return (
            f"\n  YUNO-LLM Model Info\n"
            f"  {'='*40}\n"
            f"  Version:         {self.YUNO_VERSION}\n"
            f"  Base model:      {getattr(cfg, 'model_type', 'unknown')}\n"
            f"  Hidden size:     {getattr(cfg, 'hidden_size', 'N/A')}\n"
            f"  Num layers:      {getattr(cfg, 'num_hidden_layers', 'N/A')}\n"
            f"  Attention heads: {getattr(cfg, 'num_attention_heads', 'N/A')}\n"
            f"  KV heads:        {getattr(cfg, 'num_key_value_heads', 'N/A')}\n"
            f"  Vocab size:      {getattr(cfg, 'vocab_size', 'N/A'):,}\n"
            f"  Total params:    {total:,}\n"
            f"  Trainable:       {trainable:,}\n"
            f"  Frozen:          {total - trainable:,}\n"
            f"  Size (fp32):     ~{total * 4 / 1e9:.2f} GB\n"
        )

    def forward(self, *args, **kwargs):
        """Forward pass — delegates to Qwen3."""
        return self._model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """Text generation — delegates to Qwen3.generate()."""
        return self._model.generate(*args, **kwargs)

    def save_pretrained(self, path: str) -> None:
        """Save model weights to disk."""
        Path(path).mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(path)
        logger.info(f"Model saved to {path}")

    def eval(self) -> "YunoForCausalLM":
        """Set model to eval mode."""
        self._model.eval()
        return self

    def train(self) -> "YunoForCausalLM":
        """Set model to training mode."""
        self._model.train()
        return self

    @property
    def model(self):
        """Access the underlying Qwen3 model."""
        return self._model

    @property
    def config(self):
        """Access the model's HuggingFace config."""
        return self._model.config

    def __repr__(self) -> str:
        return (
            f"YunoForCausalLM("
            f"version={self.YUNO_VERSION!r}, "
            f"params={self.total_parameters():,})"
        )
