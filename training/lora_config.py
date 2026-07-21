"""
YUNO-LLM: LoRA Configuration
==============================
Defines the LoRA (Low-Rank Adaptation) configuration for fine-tuning.

LoRA adds small trainable rank decomposition matrices to the frozen
base model weights. This allows fine-tuning with ~1% of the parameters.

Key concepts:
- r (rank): The bottleneck dimension of the low-rank matrices.
  Higher r → more capacity → more trainable params.
  Start with r=16, increase if the model underfits.

- alpha: Scaling factor. Effective scale = alpha / r.
  Common to set alpha = 2*r (e.g., r=16, alpha=32).

- target_modules: Which linear layers to apply LoRA to.
  Attention-only is safest to start.

Usage:
    from training.lora_config import get_lora_config
    lora_cfg = get_lora_config()
"""

from dataclasses import dataclass
from typing import List


@dataclass
class LoRAConfig:
    """LoRA hyperparameter configuration."""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"           # "none" | "all" | "lora_only"
    task_type: str = "CAUSAL_LM"
    target_modules: List[str] = None
    inference_mode: bool = False

    def __post_init__(self):
        if self.target_modules is None:
            # Attention-only (safe default)
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    @property
    def trainable_ratio(self) -> str:
        """Approximate trainable parameter ratio."""
        # Each LoRA adds 2 matrices of shape [hidden, r] and [r, hidden]
        # Very rough estimate
        return f"~{2 * self.r / 1024 * 100:.1f}% of each attention layer"


# ── Preset configurations ─────────────────────────────────────────────────────

def get_lora_config(preset: str = "attention_only") -> LoRAConfig:
    """
    Return a preset LoRA configuration.

    Presets:
        "attention_only": LoRA on Q, K, V, O projections only (safest, fastest)
        "full":           LoRA on attention + all FFN layers (more capacity)
        "minimal":        LoRA on V, O only (fewest params, quick experiments)

    Args:
        preset: One of "attention_only", "full", "minimal"

    Returns:
        LoRAConfig instance
    """
    presets = {
        "attention_only": LoRAConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
        "full": LoRAConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        ),
        "minimal": LoRAConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["v_proj", "o_proj"],
        ),
        "high_rank": LoRAConfig(
            r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    }
    if preset not in presets:
        raise ValueError(f"Unknown preset {preset!r}. Choose from: {list(presets.keys())}")
    return presets[preset]


def apply_lora(model, lora_config: LoRAConfig):
    """
    Apply LoRA to the model using PEFT.

    Args:
        model: HuggingFace model (AutoModelForCausalLM)
        lora_config: LoRAConfig instance

    Returns:
        PEFT model with LoRA applied
    """
    from peft import LoraConfig, get_peft_model, TaskType

    peft_config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_config.target_modules,
        inference_mode=lora_config.inference_mode,
    )

    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()
    return peft_model


if __name__ == "__main__":
    # Show all presets
    for name in ["attention_only", "full", "minimal", "high_rank"]:
        cfg = get_lora_config(name)
        print(f"\n[{name}]")
        print(f"  r={cfg.r}, alpha={cfg.lora_alpha}, dropout={cfg.lora_dropout}")
        print(f"  target_modules={cfg.target_modules}")
