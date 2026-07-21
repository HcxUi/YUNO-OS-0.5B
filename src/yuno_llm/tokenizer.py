"""
YUNO-LLM Tokenizer Wrapper
============================
Wraps the Qwen3 tokenizer with YUNO-specific defaults
and chat template formatting.
"""

from typing import List, Dict, Optional, Union
import logging

logger = logging.getLogger("yuno_llm.tokenizer")


class YunoTokenizer:
    """
    Wrapper around the Qwen3 tokenizer.

    Adds:
    - Automatic chat template formatting
    - Consistent padding/truncation defaults
    - Logging of tokenization

    Example:
        tok = YunoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        tokens = tok.encode_chat([{"role": "user", "content": "Hello"}])
        text = tok.decode(tokens)
    """

    def __init__(self, tokenizer, config=None):
        self._tokenizer = tokenizer
        self._config = config
        self.vocab_size = tokenizer.vocab_size
        self.eos_token = tokenizer.eos_token
        self.pad_token = tokenizer.pad_token
        self.eos_token_id = tokenizer.eos_token_id

    @classmethod
    def from_pretrained(cls, model_id: str, config=None, **kwargs) -> "YunoTokenizer":
        """Load tokenizer from HuggingFace Hub or local path."""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            padding_side="right",
            **kwargs,
        )
        logger.info(f"Loaded tokenizer from {model_id} (vocab_size={tokenizer.vocab_size:,})")
        return cls(tokenizer, config)

    def encode_chat(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
        tokenize: bool = True,
        return_tensors: Optional[str] = "pt",
    ):
        """
        Format messages with the chat template and tokenize.

        Args:
            messages: [{"role": "user"/"assistant"/"system", "content": "..."}]
            add_generation_prompt: If True, adds the assistant prefix token
            tokenize: If True, return token IDs. If False, return raw text.
            return_tensors: "pt" for PyTorch tensors, None for lists

        Returns:
            Tokenizer output (BatchEncoding) or raw formatted text
        """
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        if not tokenize:
            return text
        return self._tokenizer(
            text,
            return_tensors=return_tensors,
            padding=True,
            truncation=True,
            max_length=getattr(self._config, "max_length", 2048) if self._config else 2048,
        )

    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        """Decode token IDs back to text."""
        return self._tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

    def encode(self, text: str, **kwargs):
        """Encode plain text (not chat formatted)."""
        return self._tokenizer(text, **kwargs)

    def add_special_tokens(self, extra_tokens: Optional[List[str]] = None) -> int:
        """Add YUNO special tokens to underlying tokenizer vocabulary."""
        tokens = ["<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]
        if extra_tokens:
            tokens.extend(extra_tokens)
        num_added = self._tokenizer.add_special_tokens({"additional_special_tokens": tokens})
        self.vocab_size = len(self._tokenizer)
        logger.info(f"[YunoTokenizer] Added {num_added} special tokens. New vocab size: {self.vocab_size:,}")
        return num_added

    def __call__(self, *args, **kwargs):
        """Pass-through to underlying tokenizer."""
        return self._tokenizer(*args, **kwargs)

    def __repr__(self) -> str:
        return f"YunoTokenizer(vocab_size={self.vocab_size:,}, eos={self.eos_token!r})"
