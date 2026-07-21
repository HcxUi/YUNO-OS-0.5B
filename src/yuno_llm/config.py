"""
YUNO-LLM Configuration
=======================
YunoConfig is the master configuration object for the entire YUNO OS.
All modules (identity, memory, tools, updater, generation, hardware) read
their settings from this single, validated config object.

Loaded from:  config/yuno_config.yaml

Example:
    cfg = YunoConfig.from_yaml("config/yuno_config.yaml")
    print(cfg.identity.name)              # "YUNO"
    print(cfg.generation.temperature)    # 0.7
    print(cfg.updater.require_user_approval)  # True
    print(cfg.memory.max_short_term_turns)    # 15
"""

from dataclasses import dataclass, field
from typing import Optional
import yaml
from pathlib import Path


# ── Sub-configs ───────────────────────────────────────────────────────────────

@dataclass
class YunoIdentityConfig:
    """YUNO's system persona and identity settings."""
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
    """Hardware device and dtype settings."""
    device: str = "auto"        # auto | cpu | cuda | mps
    dtype: str = "float32"      # float32 | float16 | bfloat16
    load_in_4bit: bool = False
    load_in_8bit: bool = False


@dataclass
class YunoLoggingConfig:
    """Logging preferences."""
    level: str = "INFO"           # DEBUG | INFO | WARNING | ERROR
    log_to_file: bool = True
    log_file: str = "logs/yuno_llm.log"
    log_model_info: bool = True
    log_generation: bool = False


@dataclass
class YunoMemoryConfig:
    """Memory system settings."""
    enabled: bool = True
    long_term_file: str = "datasets/long_term_memory.json"
    project_file: str = "datasets/project_memory.json"
    episodic_db_file: str = "datasets/episodic_memory.db"
    max_short_term_turns: int = 15
    auto_save_interval_turns: int = 5
    max_episodic_results: int = 5


@dataclass
class YunoToolsConfig:
    """Tool registry and HITL permission settings."""
    enabled: bool = True
    sandbox_dir: str = "workspace"
    auto_approve_reads: bool = True       # Read-only tools execute automatically
    require_permission: bool = True       # Write/execute tools need console approval
    script_timeout_seconds: int = 30      # Max time for run_script tool
    audit_log_file: str = "logs/tool_audit.log"


@dataclass
class YunoLanguageConfig:
    """Language and locale settings."""
    default_mode: str = "hinglish"   # hinglish | english | hindi | mixed
    allow_switching: bool = True
    primary_locale: str = "en-IN"


@dataclass
class YunoUpdaterConfig:
    """Safe self-update system settings."""
    enabled: bool = True
    check_on_startup: bool = True
    require_user_approval: bool = True    # ALWAYS ask the user before applying
    auto_apply: bool = False              # NEVER apply silently
    backup_before_update: bool = True     # Git stash before any change
    verify_signatures: bool = False       # Future: GPG signature verification
    rollback_on_failure: bool = True      # Auto-rollback if smoke test fails
    check_model_updates: bool = False     # Check HF Hub for model weight updates
    timeout_seconds: int = 10


@dataclass
class YunoVisionConfig:
    """Vision module settings for image analysis and OCR."""
    enabled: bool = True
    ocr_engine: str = "tesseract"       # tesseract | fallback
    max_image_dim: int = 1024
    supported_formats: list = field(default_factory=lambda: [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".pdf"])


@dataclass
class YunoVoiceConfig:
    """Voice module settings for Text-to-Speech (TTS) and Speech-to-Text (STT)."""
    enabled: bool = True
    tts_engine: str = "pyttsx3"          # pyttsx3 | piper
    stt_engine: str = "speech_recognition"
    speech_rate: int = 175
    volume: float = 1.0
    auto_speak: bool = False             # Speak assistant responses automatically


@dataclass
class YunoAutomationConfig:
    """Automation module settings for file sorting, reminders, and scaffolding."""
    enabled: bool = True
    default_organize_dir: str = "workspace"
    reminders_file: str = "workspace/reminders.json"


@dataclass
class YunoBaseModelConfig:
    """Base LLM model settings."""
    name: str = "Qwen/Qwen3-0.6B"
    revision: str = "main"
    trust_remote_code: bool = True


# ── Master Config ─────────────────────────────────────────────────────────────

@dataclass
class YunoConfig:
    """
    Master YUNO-LLM configuration.

    All sub-configs are populated from config/yuno_config.yaml.
    Fields fall back to sensible defaults if the YAML key is missing.

    Usage:
        cfg = YunoConfig.from_yaml("config/yuno_config.yaml")
    """
    version: str = "0.1.0"

    base_model: YunoBaseModelConfig = field(default_factory=YunoBaseModelConfig)
    identity: YunoIdentityConfig = field(default_factory=YunoIdentityConfig)
    generation: YunoGenerationConfig = field(default_factory=YunoGenerationConfig)
    hardware: YunoHardwareConfig = field(default_factory=YunoHardwareConfig)
    logging: YunoLoggingConfig = field(default_factory=YunoLoggingConfig)
    memory: YunoMemoryConfig = field(default_factory=YunoMemoryConfig)
    tools: YunoToolsConfig = field(default_factory=YunoToolsConfig)
    language: YunoLanguageConfig = field(default_factory=YunoLanguageConfig)
    updater: YunoUpdaterConfig = field(default_factory=YunoUpdaterConfig)
    vision: YunoVisionConfig = field(default_factory=YunoVisionConfig)
    voice: YunoVoiceConfig = field(default_factory=YunoVoiceConfig)
    automation: YunoAutomationConfig = field(default_factory=YunoAutomationConfig)

    # Paths (kept as flat strings for backward compat)
    base_model_dir: str = "models/base"
    yuno_model_dir: str = "models/yuno"
    checkpoints_dir: str = "checkpoints"
    experiments_dir: str = "experiments"

    # ── Loader ───────────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str) -> "YunoConfig":
        """Load and validate config from a YAML file."""
        yaml_path = Path(path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        cfg = cls()

        # Project version
        if "project" in data:
            cfg.version = data["project"].get("version", cfg.version)

        # Base model
        if "base_model" in data:
            bm = data["base_model"]
            cfg.base_model = YunoBaseModelConfig(
                name=bm.get("name", cfg.base_model.name),
                revision=bm.get("revision", cfg.base_model.revision),
                trust_remote_code=bm.get("trust_remote_code", cfg.base_model.trust_remote_code),
            )

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
            gen = data["generation"]
            cfg.generation = YunoGenerationConfig(
                max_new_tokens=gen.get("max_new_tokens", cfg.generation.max_new_tokens),
                temperature=gen.get("temperature", cfg.generation.temperature),
                top_p=gen.get("top_p", cfg.generation.top_p),
                top_k=gen.get("top_k", cfg.generation.top_k),
                repetition_penalty=gen.get("repetition_penalty", cfg.generation.repetition_penalty),
                do_sample=gen.get("do_sample", cfg.generation.do_sample),
                stream=gen.get("stream", cfg.generation.stream),
            )

        # Hardware
        if "hardware" in data:
            hw = data["hardware"]
            cfg.hardware = YunoHardwareConfig(
                device=hw.get("device", cfg.hardware.device),
                dtype=hw.get("dtype", cfg.hardware.dtype),
                load_in_4bit=hw.get("load_in_4bit", cfg.hardware.load_in_4bit),
                load_in_8bit=hw.get("load_in_8bit", cfg.hardware.load_in_8bit),
            )

        # Logging
        if "logging" in data:
            log = data["logging"]
            cfg.logging = YunoLoggingConfig(
                level=log.get("level", cfg.logging.level),
                log_to_file=log.get("log_to_file", cfg.logging.log_to_file),
                log_file=log.get("log_file", cfg.logging.log_file),
                log_model_info=log.get("log_model_info", cfg.logging.log_model_info),
                log_generation=log.get("log_generation", cfg.logging.log_generation),
            )

        # Memory (previously ignored — now fully parsed)
        if "memory" in data:
            mem = data["memory"]
            cfg.memory = YunoMemoryConfig(
                enabled=mem.get("enabled", cfg.memory.enabled),
                long_term_file=mem.get("long_term_file", cfg.memory.long_term_file),
                project_file=mem.get("project_file", cfg.memory.project_file),
                episodic_db_file=mem.get("episodic_db_file", cfg.memory.episodic_db_file),
                max_short_term_turns=mem.get("max_short_term_turns", cfg.memory.max_short_term_turns),
                auto_save_interval_turns=mem.get("auto_save_interval_turns", cfg.memory.auto_save_interval_turns),
                max_episodic_results=mem.get("max_episodic_results", cfg.memory.max_episodic_results),
            )

        # Tools (previously ignored — now fully parsed)
        if "tools" in data:
            tl = data["tools"]
            cfg.tools = YunoToolsConfig(
                enabled=tl.get("enabled", cfg.tools.enabled),
                sandbox_dir=tl.get("sandbox_dir", cfg.tools.sandbox_dir),
                auto_approve_reads=tl.get("auto_approve_reads", cfg.tools.auto_approve_reads),
                require_permission=tl.get("require_permission", cfg.tools.require_permission),
                script_timeout_seconds=tl.get("script_timeout_seconds", cfg.tools.script_timeout_seconds),
                audit_log_file=tl.get("audit_log_file", cfg.tools.audit_log_file),
            )

        # Language (previously ignored — now fully parsed)
        if "language" in data:
            lang = data["language"]
            cfg.language = YunoLanguageConfig(
                default_mode=lang.get("default_mode", cfg.language.default_mode),
                allow_switching=lang.get("allow_switching", cfg.language.allow_switching),
                primary_locale=lang.get("primary_locale", cfg.language.primary_locale),
            )

        # Updater (previously ignored — now fully parsed with safe defaults)
        if "updater" in data:
            upd = data["updater"]
            cfg.updater = YunoUpdaterConfig(
                enabled=upd.get("enabled", cfg.updater.enabled),
                check_on_startup=upd.get("check_on_startup", cfg.updater.check_on_startup),
                require_user_approval=upd.get("require_user_approval", True),   # default: True
                auto_apply=upd.get("auto_apply", False),                        # default: False
                backup_before_update=upd.get("backup_before_update", True),
                verify_signatures=upd.get("verify_signatures", False),
                rollback_on_failure=upd.get("rollback_on_failure", True),
                check_model_updates=upd.get("check_model_updates", False),
                timeout_seconds=upd.get("timeout_seconds", cfg.updater.timeout_seconds),
            )

        # Vision
        if "vision" in data:
            vis = data["vision"]
            cfg.vision = YunoVisionConfig(
                enabled=vis.get("enabled", cfg.vision.enabled),
                ocr_engine=vis.get("ocr_engine", cfg.vision.ocr_engine),
                max_image_dim=vis.get("max_image_dim", cfg.vision.max_image_dim),
            )

        # Voice
        if "voice" in data:
            vc = data["voice"]
            cfg.voice = YunoVoiceConfig(
                enabled=vc.get("enabled", cfg.voice.enabled),
                tts_engine=vc.get("tts_engine", cfg.voice.tts_engine),
                stt_engine=vc.get("stt_engine", cfg.voice.stt_engine),
                speech_rate=vc.get("speech_rate", cfg.voice.speech_rate),
                volume=vc.get("volume", cfg.voice.volume),
                auto_speak=vc.get("auto_speak", cfg.voice.auto_speak),
            )

        # Automation
        if "automation" in data:
            auto = data["automation"]
            cfg.automation = YunoAutomationConfig(
                enabled=auto.get("enabled", cfg.automation.enabled),
                default_organize_dir=auto.get("default_organize_dir", cfg.automation.default_organize_dir),
                reminders_file=auto.get("reminders_file", cfg.automation.reminders_file),
            )

        # Paths
        if "paths" in data:
            paths = data["paths"]
            cfg.base_model_dir = paths.get("base_model_dir", cfg.base_model_dir)
            cfg.yuno_model_dir = paths.get("yuno_model_dir", cfg.yuno_model_dir)
            cfg.checkpoints_dir = paths.get("checkpoints_dir", cfg.checkpoints_dir)
            cfg.experiments_dir = paths.get("experiments_dir", cfg.experiments_dir)

        return cfg

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize config to a plain dictionary."""
        return {
            "version": self.version,
            "base_model": {
                "name": self.base_model.name,
                "revision": self.base_model.revision,
            },
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
            "memory": {
                "enabled": self.memory.enabled,
                "max_short_term_turns": self.memory.max_short_term_turns,
            },
            "tools": {
                "enabled": self.tools.enabled,
                "require_permission": self.tools.require_permission,
            },
            "updater": {
                "enabled": self.updater.enabled,
                "require_user_approval": self.updater.require_user_approval,
                "auto_apply": self.updater.auto_apply,
            },
        }

    def __repr__(self) -> str:
        return (
            f"YunoConfig(\n"
            f"  base_model={self.base_model.name!r},\n"
            f"  version={self.version!r},\n"
            f"  identity.name={self.identity.name!r},\n"
            f"  generation.temperature={self.generation.temperature},\n"
            f"  hardware.device={self.hardware.device!r},\n"
            f"  memory.enabled={self.memory.enabled},\n"
            f"  tools.require_permission={self.tools.require_permission},\n"
            f"  updater.require_user_approval={self.updater.require_user_approval},\n"
            f")"
        )
