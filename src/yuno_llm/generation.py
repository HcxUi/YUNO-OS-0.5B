"""
YUNO-LLM Generation Orchestration
==================================
Manages the orchestrating loop connecting planning, memory, and safe tool use.
"""

from __future__ import annotations
import logging
import time
from typing import Optional, List, Dict, Generator

import torch

logger = logging.getLogger("yuno_llm.generation")


class YunoGenerator:
    """
    Handles text generation for YUNO-LLM Vision OS.

    Integrates:
    - YunoMemory: short, long, and project databases.
    - YunoPlanner: plans parsing.
    - YunoToolRegistry: HITL tool safe execution loop.
    - YunoIdentity: custom identity and system prompts.
    """

    def __init__(self, model, tokenizer, config=None):
        self._model = model if not hasattr(model, "_model") else model._model
        
        from .tokenizer import YunoTokenizer
        if isinstance(tokenizer, YunoTokenizer):
            self._tokenizer = tokenizer._tokenizer
        else:
            self._tokenizer = tokenizer
        self._config = config

        # Core OS components
        from .identity import YunoIdentity
        from .memory import YunoMemory
        from .tools import YunoToolRegistry
        from .planner import YunoPlanner

        self._identity = YunoIdentity(config)
        self.memory = YunoMemory(config)
        self.tools = YunoToolRegistry(config)
        self.planner = YunoPlanner(config)

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
        Runs the full YUNO OS chat loop:
          1. Intent classification (TOOL_CALL / MEMORY_QUERY / PLAN / CHAT)
          2. Dynamic system prompt with memory context
          3. LLM generation with streaming support
          4. Thought-block stripping (<think>...</think>)
          5. Episodic memory storage for future recall
        """
        from .planner import IntentType

        # 1. Classify intent
        intent, tool_name, tool_args = self.planner.parse_intent(user_message)

        # ── Tool call: execute immediately ────────────────────────────────
        if intent == IntentType.TOOL_CALL and tool_name:
            tool_output = self.tools.run_tool(tool_name, **tool_args)
            self.memory.add_chat_turn("user", user_message)
            self.memory.add_chat_turn("assistant", tool_output)
            return tool_output

        # ── Memory query: search episodic memory and return summary ───────
        if intent == IntentType.MEMORY_QUERY:
            query = tool_args.get("query", user_message)
            hits = self.memory.search_memory(query)
            if hits:
                lines = [f"[{h.timestamp}] {h.content}" for h in hits]
                result = "Yeh raha maine jo yaad rakha:\n" + "\n".join(lines)
            else:
                result = "Mujhe koi relevant memory nahi mili. Kya aap aur details de sakte hain?"
            self.memory.add_chat_turn("user", user_message)
            self.memory.add_chat_turn("assistant", result)
            return result

        # ── Chat / Plan: run through LLM ──────────────────────────────────
        self.memory.add_chat_turn("user", user_message)

        # 2. Build dynamic system prompt (injects date, user name, memory)
        messages = self._identity.apply_system_prompt(
            self.memory.get_chat_history(),
            override_system=system_override,
            memory=self.memory,
            last_user_message=user_message,
        )

        # Format with chat template
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt")

        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs = {**self._gen_kwargs, **generation_kwargs}
        gen_kwargs["eos_token_id"] = self._tokenizer.eos_token_id
        gen_kwargs["pad_token_id"] = self._tokenizer.eos_token_id

        # 3. Generate
        t0 = time.time()
        with torch.no_grad():
            output = self._model.generate(**inputs, **gen_kwargs)
        elapsed = time.time() - t0

        input_len = inputs["input_ids"].shape[1]
        new_tokens = output[0][input_len:]
        raw_response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        tps = len(new_tokens) / elapsed
        logger.info(f"Generated {len(new_tokens)} tokens in {elapsed:.2f}s ({tps:.1f} tok/s)")

        # 4. Strip <think> blocks from the visible response
        thought, response = self.planner.extract_thought_steps(raw_response)
        if thought:
            logger.debug(f"[Planner] Thought block: {thought[:200]}")

        # 5. Save to short-term memory and episodic store
        self.memory.add_chat_turn("assistant", response)
        # Store meaningful turns as episodic memories for future recall
        if len(user_message) > 20:
            self.memory.add_episodic_entry(
                content=f"User: {user_message[:300]} | YUNO: {response[:300]}",
                tags="conversation",
                source="chat",
            )

        return response

    def stream(
        self,
        user_message: str,
        system_override: Optional[str] = None,
        **generation_kwargs,
    ) -> Generator[str, None, None]:
        """
        Streaming chat with integrated planning, memory context, and thought stripping.
        """
        from transformers import TextIteratorStreamer
        from threading import Thread
        from .planner import IntentType

        # 1. Intent classification
        intent, tool_name, tool_args = self.planner.parse_intent(user_message)

        if intent == IntentType.TOOL_CALL and tool_name:
            tool_output = self.tools.run_tool(tool_name, **tool_args)
            yield tool_output
            self.memory.add_chat_turn("user", user_message)
            self.memory.add_chat_turn("assistant", tool_output)
            return

        if intent == IntentType.MEMORY_QUERY:
            query = tool_args.get("query", user_message)
            hits = self.memory.search_memory(query)
            if hits:
                result = "Yeh raha maine jo yaad rakha:\n" + "\n".join(
                    f"[{h.timestamp}] {h.content}" for h in hits
                )
            else:
                result = "Mujhe koi relevant memory nahi mili."
            yield result
            self.memory.add_chat_turn("user", user_message)
            self.memory.add_chat_turn("assistant", result)
            return

        # 2. Chat / Plan path
        self.memory.add_chat_turn("user", user_message)

        messages = self._identity.apply_system_prompt(
            self.memory.get_chat_history(),
            override_system=system_override,
            memory=self.memory,
            last_user_message=user_message,
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

        # Stream tokens, but suppress <think>...</think> blocks from output
        full_response: List[str] = []
        in_think_block = False
        buffer = ""

        for chunk in streamer:
            buffer += chunk
            full_response.append(chunk)

            # Detect start of think block
            if "<think>" in buffer and not in_think_block:
                in_think_block = True
                # Yield anything before <think>
                before = buffer[: buffer.index("<think>")]
                if before:
                    yield before
                buffer = buffer[buffer.index("<think>"):]
                continue

            # Detect end of think block
            if in_think_block and "</think>" in buffer:
                in_think_block = False
                after = buffer[buffer.index("</think>") + len("</think>"):]
                buffer = after
                continue

            # Yield normally if not in a think block
            if not in_think_block:
                yield buffer
                buffer = ""

        # Yield any remaining buffer
        if buffer and not in_think_block:
            yield buffer

        thread.join()

        # Save to memory
        raw = "".join(full_response)
        _, clean_response = self.planner.extract_thought_steps(raw)
        self.memory.add_chat_turn("assistant", clean_response)
        if len(user_message) > 20:
            self.memory.add_episodic_entry(
                content=f"User: {user_message[:300]} | YUNO: {clean_response[:300]}",
                tags="conversation",
                source="stream",
            )

    def reset_history(self) -> None:
        """Clear conversation history."""
        self.memory.clear_short_term()
        logger.info("Conversation history cleared.")

    def __repr__(self) -> str:
        return (
            f"YunoGenerator("
            f"identity={self._identity.name!r}, "
            f"history_turns={len(self.memory.get_chat_history())//2})"
        )


if __name__ == "__main__":
    import sys
    from pathlib import Path
    root_dir = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(root_dir / "src"))

    from yuno_llm.config import YunoConfig
    from yuno_llm.identity import YunoIdentity
    from yuno_llm.memory import YunoMemory
    from yuno_llm.tools import YunoToolRegistry
    from yuno_llm.planner import YunoPlanner
    from yuno_llm.automation import YunoAutomation

    cfg = YunoConfig.from_yaml("config/yuno_config.yaml")
    identity = YunoIdentity(cfg)
    memory = YunoMemory(cfg)
    tools = YunoToolRegistry(cfg)
    planner = YunoPlanner(cfg)
    automation = YunoAutomation(cfg)

    print("=" * 65)
    print("  YUNO-LLM v0.5.0 PERSONAL AI OPERATING SYSTEM")
    print("=" * 65)
    print(f"  Persona:             {identity.name}")
    print(f"  Short-Term Window:   {cfg.memory.max_short_term_turns} turns")
    print(f"  Episodic Database:   SQLite FTS5")
    print(f"  Registered Tools:    {len(tools.list_tools())} tools")
    print("=" * 65)
    print("  Type 'exit' or 'quit' to exit.\n")

    user_name = memory.get_personal_fact("user_name", "User")
    while True:
        try:
            prompt = input(f"{user_name} > ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("exit", "quit"):
                print("YUNO: Alvida! Phir milenge.")
                break

            intent, tool_name, tool_args = planner.parse_intent(prompt)
            if intent.name == "TOOL_CALL" and tool_name:
                print(f"[YUNO Planning] Tool Call Detected: {tool_name} with args {tool_args}")
                result = tools.run_tool(tool_name, **tool_args)
                safe_out = result.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8")
                print(f"[YUNO Tool Output]\n{safe_out}")
            else:
                print(f"YUNO: Main aapki baat samajh gaya. How can I help you further with '{prompt}'?")
        except (KeyboardInterrupt, EOFError):
            print("\nYUNO: Session ended.")
            break
