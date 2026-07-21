"""
YUNO-LLM Python Package
========================
Research-grade open-source LLM based on Qwen3.

Usage:
    from yuno_llm import YunoLLM
    model = YunoLLM.from_config("config/yuno_config.yaml")
    response = model.chat("What is quantum computing?")
"""

import numpy as np
np.complex = complex  # Monkeypatch for legacy librosa compatibility with modern numpy

from .config import YunoConfig
from .model import YunoForCausalLM
from .tokenizer import YunoTokenizer
from .generation import YunoGenerator

__version__ = "0.1.0"
__author__ = "YUNO-LLM Team"

__all__ = [
    "YunoConfig",
    "YunoForCausalLM",
    "YunoTokenizer",
    "YunoGenerator",
]
