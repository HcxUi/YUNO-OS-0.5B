"""
YUNO-LLM Python Package
========================
Personal AI Operating System — Natural, Private, Offline-first.

Core OS Modules:
    YunoConfig          — Master configuration loader (from YAML)
    YunoIdentity        — System persona and dynamic prompt builder
    YunoMemory          — Short-term, long-term, project, and episodic memory
    YunoPlanner         — Intent classification and multi-step plan extraction
    YunoToolRegistry    — HITL tool registry with permission enforcement
    YunoUpdater         — Safe update checker (check → notify → approve → apply)
    YunoVision          — Image understanding, OCR, screenshot & document processing
    YunoVoice           — Text-to-Speech (TTS) and Speech-to-Text (STT) processing
    YunoAutomation      — File sorting, reminders, project scaffolding
    YunoGenerator       — Main orchestration loop (chat + streaming)
    YunoForCausalLM     — Base CausalLM model loader
    YunoTokenizer       — Tokenizer wrapper with special tokens

New in v0.5.0 (Automation Enabled):
    YunoAutomation      — Core Automation engine
    YunoAutomationConfig— Automation configuration dataclass
    YunoVoice           — Core Voice engine (pyttsx3 & SpeechRecognition)
    YunoVoiceConfig     — Voice configuration dataclass
    YunoVision          — Core Vision engine
    YunoVisionConfig    — Vision configuration dataclass
    IntentType          — Enum: CHAT / TOOL_CALL / MEMORY_QUERY / PLAN
    EpisodicMemory      — SQLite FTS5 searchable episodic memory
    ToolPermissionLevel — Enum: AUTO / CONFIRM / EXPLICIT

Usage:
    from yuno_llm import YunoConfig, YunoGenerator, YunoAutomation
    cfg = YunoConfig.from_yaml("config/yuno_config.yaml")

    # Organize workspace files
    auto = YunoAutomation(cfg)
    auto.organize_directory("workspace")
"""

import numpy as np
np.complex = complex  # Monkeypatch for legacy librosa compatibility with modern numpy

# ── Core config ───────────────────────────────────────────────────────────────
from .config import (
    YunoConfig,
    YunoIdentityConfig,
    YunoGenerationConfig,
    YunoHardwareConfig,
    YunoLoggingConfig,
    YunoMemoryConfig,
    YunoToolsConfig,
    YunoLanguageConfig,
    YunoUpdaterConfig,
    YunoVisionConfig,
    YunoVoiceConfig,
    YunoAutomationConfig,
    YunoBaseModelConfig,
)

# ── Core OS modules ───────────────────────────────────────────────────────────
from .model import YunoForCausalLM
from .tokenizer import YunoTokenizer
from .identity import YunoIdentity
from .memory import YunoMemory, EpisodicMemory, EpisodicEntry
from .planner import YunoPlanner, IntentType, YunoPlan, PlanStep
from .tools import YunoToolRegistry, YunoTool, ToolPermissionLevel
from .updater import YunoUpdater, UpdateInfo
from .vision import YunoVision, VisionAnalysisResult
from .voice import YunoVoice
from .automation import YunoAutomation
from .generation import YunoGenerator
from .cli import main

__version__ = "0.5.0"
__author__ = "YUNO-LLM Team"

__all__ = [
    # Config
    "YunoConfig",
    "YunoIdentityConfig",
    "YunoGenerationConfig",
    "YunoHardwareConfig",
    "YunoLoggingConfig",
    "YunoMemoryConfig",
    "YunoToolsConfig",
    "YunoLanguageConfig",
    "YunoUpdaterConfig",
    "YunoVisionConfig",
    "YunoVoiceConfig",
    "YunoAutomationConfig",
    "YunoBaseModelConfig",
    # Core OS
    "YunoForCausalLM",
    "YunoTokenizer",
    "YunoIdentity",
    "YunoMemory",
    "EpisodicMemory",
    "EpisodicEntry",
    "YunoPlanner",
    "IntentType",
    "YunoPlan",
    "PlanStep",
    "YunoToolRegistry",
    "YunoTool",
    "ToolPermissionLevel",
    "YunoUpdater",
    "UpdateInfo",
    "YunoVision",
    "VisionAnalysisResult",
    "YunoVoice",
    "YunoAutomation",
    "YunoGenerator",
]
