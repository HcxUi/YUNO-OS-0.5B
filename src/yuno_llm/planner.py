"""
YUNO Planner & Intent Parser Module
====================================
Interprets prompts, extracts plans, and decides if tools should be triggered.
"""

import re
import json
import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger("yuno_llm.planner")


class YunoPlanner:
    """
    Parses user inputs, generates step-by-step plans, and identifies
    when tools (e.g. read_file, write_file) should be executed.
    """

    def __init__(self, config=None):
        self.config = config

    def parse_intent(self, user_input: str) -> Tuple[Optional[str], Dict[str, Any]]:
        r"""
        Parses intent from the user input.
        Returns:
            (tool_name, tool_args) or (None, {}) if it is a general chat turn.
        
        Example format: 
            "read F:\llm\config\yuno_config.yaml" -> ("read_file", {"file_path": "F:\\llm\\config\\yuno_config.yaml"})
            "write hello.txt with content 'Hi'" -> ("write_file", {"file_path": "hello.txt", "content": "Hi"})
        """
        # 1. Simple parser for read_file
        read_match = re.match(
            r"^(?:read|show|cat)\s+([a-zA-Z]:\\[^\s]+|[^\s]+)", 
            user_input, 
            re.IGNORECASE
        )
        if read_match:
            file_path = read_match.group(1).strip()
            return "read_file", {"file_path": file_path}

        # 2. Simple parser for write_file
        write_match = re.match(
            r"^(?:write|create)\s+([a-zA-Z]:\\[^\s]+|[^\s]+)\s+with\s+content\s+['\"](.*)['\"]", 
            user_input, 
            re.IGNORECASE
        )
        if write_match:
            file_path = write_match.group(1).strip()
            content = write_match.group(2).strip()
            return "write_file", {"file_path": file_path, "content": content}

        return None, {}

    def extract_thought_steps(self, raw_llm_response: str) -> Tuple[str, str]:
        """
        Extracts reasoning/thought steps (enclosed in <think> tags) and the 
        final model output.
        
        Returns:
            (thought_block, response_block)
        """
        think_match = re.search(r"<think>(.*?)</think>", raw_llm_response, re.DOTALL)
        if think_match:
            thought = think_match.group(1).strip()
            response = re.sub(r"<think>.*?</think>", "", raw_llm_response, flags=re.DOTALL).strip()
            return thought, response
        return "", raw_llm_response.strip()
