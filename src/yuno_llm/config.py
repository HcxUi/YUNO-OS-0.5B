"""
YUNO-LLM Configuration
=======================
YunoConfig extends Qwen3Config with YUNO-LLM-specific settings:
- Custom system identity
- Generation defaults
- Logging preferences
- Version tracking

This is a safe first modification — we add fields without touching
any model architecture.
"""

from dataclasses import dataclass, field
from typing import Optional
import yaml
from pathlib import Path


@dataclass
class YunoIdentityConfig:
    """YUNO-LLM system identity settings."""
    name: str = "YUNO"
    version: str = "0.1"
    system_prompt: str = (
        "You are YUNO, a research-grade AI assistant built on the YUNO-LLM architecture. "
        "You are helpful, honest, and direct. You acknowledge uncertainty when you don't know something. "
        "You are built and owned by the YUNO-LLM team."
    )


@dataclass
class YunoGenerationConfig:
    """Default generation hyperparameters."""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    stream: bool = True


@dataclass
class YunoHardwareConfig:
    """Hardware and dtype settings."""
    device: str = "auto"        # auto | cpu | cuda | mps
    dtype: str = "float32"      # float32 | float16 | bfloat16
    load_in_4bit: bool = False
    load_in_8bit: bool = False


@dataclass
class YunoLoggingConfig:
    """Logging preferences."""
    level: str = "INFO"
    log_to_file: bool = True
    log_file: str = "logs/yuno_llm.log"
    log_model_info: bool = True
    log_generation: bool = False


@dataclass
class YunoConfig:
    """
    Master YUNO-LLM configuration.

    Extends the base model's config with project-specific settings.
    Loaded from config/yuno_config.yaml.

    Example:
        cfg = YunoConfig.from_yaml("config/yuno_config.yaml")
        print(cfg.identity.name)         # "YUNO"
        print(cfg.generation.temperature) # 0.7
    """
    base_model: str = "Qwen/Qwen3-0.6B"
    version: str = "0.1.0"

    identity: YunoIdentityConfig = field(default_factory=YunoIdentityConfig)
    generation: YunoGenerationConfig = field(default_factory=YunoGenerationConfig)
    hardware: YunoHardwareConfig = field(default_factory=YunoHardwareConfig)
    logging: YunoLoggingConfig = field(default_factory=YunoLoggingConfig)

    # Paths
    base_model_dir: str = "models/base"
    yuno_model_dir: str = "models/yuno"
    checkpoints_dir: str = "checkpoints"
    experiments_dir: str = "experiments"

    @classmethod
    def from_yaml(cls, path: str) -> "YunoConfig":
        """Load config from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        cfg = cls()

        # Base model
        if "base_model" in data:
            cfg.base_model = data["base_model"].get("name", cfg.base_model)

        # Version
        if "project" in data:
            cfg.version = data["project"].get("version", cfg.version)

        # Identity
        if "identity" in data:
            id_data = data["identity"]
            cfg.identity = YunoIdentityConfig(
                name=id_data.get("name", cfg.identity.name),
                version=id_data.get("version", cfg.identity.version),
                system_prompt=id_data.get("system_prompt", cfg.identity.system_prompt),
            )

        # Generation
        if "generation" in data:
            gen_data = data["generation"]
            cfg.generation = YunoGenerationConfig(
                max_new_tokens=gen_data.get("max_new_tokens", cfg.generation.max_new_tokens),
                temperature=gen_data.get("temperature", cfg.generation.temperature),
                top_p=gen_data.get("top_p", cfg.generation.top_p),
                top_k=gen_data.get("top_k", cfg.generation.top_k),
                repetition_penalty=gen_data.get("repetition_penalty", cfg.generation.repetition_penalty),
                do_sample=gen_data.get("do_sample", cfg.generation.do_sample),
                stream=gen_data.get("stream", cfg.generation.stream),
            )

        # Hardware
        if "hardware" in data:
            hw_data = data["hardware"]
            cfg.hardware = YunoHardwareConfig(
                device=hw_data.get("device", cfg.hardware.device),
                dtype=hw_data.get("dtype", cfg.hardware.dtype),
            )

        # Logging
        if "logging" in data:
            log_data = data["logging"]
            cfg.logging = YunoLoggingConfig(
                level=log_data.get("level", cfg.logging.level),
                log_to_file=log_data.get("log_to_file", cfg.logging.log_to_file),
                log_file=log_data.get("log_file", cfg.logging.log_file),
                log_model_info=log_data.get("log_model_info", cfg.logging.log_model_info),
                log_generation=log_data.get("log_generation", cfg.logging.log_generation),
            )

        # Paths
        if "paths" in data:
            paths = data["paths"]
            cfg.base_model_dir = paths.get("base_model_dir", cfg.base_model_dir)
            cfg.yuno_model_dir = paths.get("yuno_model_dir", cfg.yuno_model_dir)
            cfg.checkpoints_dir = paths.get("checkpoints_dir", cfg.checkpoints_dir)
            cfg.experiments_dir = paths.get("experiments_dir", cfg.experiments_dir)

        return cfg

    def to_dict(self) -> dict:
        """Serialize config to a plain dictionary."""
        return {
            "base_model": self.base_model,
            "version": self.version,
            "identity": {
                "name": self.identity.name,
                "version": self.identity.version,
                "system_prompt": self.identity.system_prompt,
            },
            "generation": {
                "max_new_tokens": self.generation.max_new_tokens,
                "temperature": self.generation.temperature,
                "top_p": self.generation.top_p,
                "top_k": self.generation.top_k,
                "repetition_penalty": self.generation.repetition_penalty,
                "do_sample": self.generation.do_sample,
            },
            "hardware": {
                "device": self.hardware.device,
                "dtype": self.hardware.dtype,
            },
        }

    def __repr__(self) -> str:
        return (
            f"YunoConfig(\n"
            f"  base_model={self.base_model!r},\n"
            f"  version={self.version!r},\n"
            f"  identity.name={self.identity.name!r},\n"
            f"  generation.temperature={self.generation.temperature},\n"
            f"  hardware.device={self.hardware.device!r},\n"
            f")"
        )
