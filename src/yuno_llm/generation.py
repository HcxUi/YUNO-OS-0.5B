"""
YUNO-LLM Generation
====================
Custom generation logic with YUNO defaults:
- System prompt injection via YunoIdentity
- Streaming output
- Configurable decoding strategy
- Token/sec measurement

Usage:
    gen = YunoGenerator(model, tokenizer, config)
    response = gen.chat("What is consciousness?")
"""

from __future__ import annotations
import logging
import time
from typing import Optional, List, Dict, Generator

import torch

logger = logging.getLogger("yuno_llm.generation")


class YunoGenerator:
    """
    Handles text generation for YUNO-LLM.

    Wraps the model's generate() with:
    - YUNO identity system prompt injection
    - Chat history management
    - Streaming via TextIteratorStreamer
    - Generation metrics logging

    Example (non-streaming):
        gen = YunoGenerator(model, tokenizer, config)
        reply = gen.chat("Explain transformers in one paragraph.")
        print(reply)

    Example (streaming):
        for chunk in gen.stream("Tell me a story"):
            print(chunk, end="", flush=True)
    """

    def __init__(self, model, tokenizer, config=None):
        self._model = model if not hasattr(model, "_model") else model._model
        
        from .tokenizer import YunoTokenizer
        if isinstance(tokenizer, YunoTokenizer):
            self._tokenizer = tokenizer._tokenizer
        else:
            self._tokenizer = tokenizer
        self._config = config

        # Import identity here to avoid circular imports
        from .identity import YunoIdentity
        self._identity = YunoIdentity(config)

        # History for multi-turn conversations
        self._history: List[Dict[str, str]] = []

    @property
    def _gen_kwargs(self) -> dict:
        """Default generation kwargs from config."""
        if self._config:
            g = self._config.generation
            return {
                "max_new_tokens": g.max_new_tokens,
                "temperature": g.temperature,
                "top_p": g.top_p,
                "top_k": g.top_k,
                "repetition_penalty": g.repetition_penalty,
                "do_sample": g.do_sample,
            }
        return {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "do_sample": True,
        }

    def chat(
        self,
        user_message: str,
        system_override: Optional[str] = None,
        **generation_kwargs,
    ) -> str:
        """
        Single-turn or multi-turn chat.

        Args:
            user_message: The user's input text
            system_override: Override the system prompt for this turn
            **generation_kwargs: Override any generation parameter

        Returns:
            The assistant's response as a string
        """
        # Build messages
        messages = self._identity.apply_system_prompt(
            self._history + [{"role": "user", "content": user_message}],
            override_system=system_override,
        )

        # Format with chat template
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt")

        # Move to model device
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Merge generation kwargs
        gen_kwargs = {**self._gen_kwargs, **generation_kwargs}
        gen_kwargs["eos_token_id"] = self._tokenizer.eos_token_id
        gen_kwargs["pad_token_id"] = self._tokenizer.eos_token_id

        # Generate
        t0 = time.time()
        with torch.no_grad():
            output = self._model.generate(**inputs, **gen_kwargs)
        elapsed = time.time() - t0

        # Decode only the new tokens
        input_len = inputs["input_ids"].shape[1]
        new_tokens = output[0][input_len:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        tps = len(new_tokens) / elapsed
        logger.info(f"Generated {len(new_tokens)} tokens in {elapsed:.2f}s ({tps:.1f} tok/s)")

        # Update history (without system prompt to avoid duplication)
        self._history.append({"role": "user", "content": user_message})
        self._history.append({"role": "assistant", "content": response})

        return response

    def stream(
        self,
        user_message: str,
        system_override: Optional[str] = None,
        **generation_kwargs,
    ) -> Generator[str, None, None]:
        """
        Streaming chat — yields text chunks as they are generated.

        Usage:
            for chunk in gen.stream("Tell me a story"):
                print(chunk, end="", flush=True)
        """
        from transformers import TextIteratorStreamer
        from threading import Thread

        messages = self._identity.apply_system_prompt(
            self._history + [{"role": "user", "content": user_message}],
            override_system=system_override,
        )
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        gen_kwargs = {
            **self._gen_kwargs,
            **generation_kwargs,
            **inputs,
            "streamer": streamer,
            "eos_token_id": self._tokenizer.eos_token_id,
            "pad_token_id": self._tokenizer.eos_token_id,
        }

        thread = Thread(target=self._model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()

        full_response = []
        for chunk in streamer:
            full_response.append(chunk)
            yield chunk

        thread.join()

        # Update history
        self._history.append({"role": "user", "content": user_message})
        self._history.append({"role": "assistant", "content": "".join(full_response)})

    def reset_history(self) -> None:
        """Clear conversation history."""
        self._history = []
        logger.info("Conversation history cleared.")

    def __repr__(self) -> str:
        return (
            f"YunoGenerator("
            f"identity={self._identity.name!r}, "
            f"history_turns={len(self._history)//2})"
        )
