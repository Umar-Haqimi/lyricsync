"""User configuration persisted as JSON (no database anywhere in this app)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from lyricsync.utils.paths import config_dir, default_model_dir


@dataclass
class AppConfig:
    # Separation
    separation_tier: int = 2          # 0 = skip, 1..3 = built-in tiers, 4 = custom
    custom_model_filename: str = ""   # used when separation_tier == 4
    normalize_loudness: bool = False

    # Transcription
    whisper_model: str = "small"      # tiny|base|small|medium|large-v3|turbo
    device: str = "auto"              # auto|cpu|cuda
    compute_type: str = "auto"        # auto|int8|int8_float16|float16|float32
    language: str = ""                # empty = autodetect
    vad_min_silence_ms: int = 500

    # Editor
    confidence_warn_threshold: float = 0.60
    confidence_bad_threshold: float = 0.35

    # Syllables
    syllable_backend: str = "hybrid"  # hybrid|pyphen
    syllable_lang: str = "en"

    # Storage — defaults to persistent user data dir, never the project tree
    model_dir: str = field(default_factory=lambda: str(default_model_dir()))

    # Batch
    max_parallel_jobs: int = 1

    def save(self, path: Path | None = None) -> None:
        p = path or (config_dir() / "config.json")
        p.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | None = None) -> "AppConfig":
        p = path or (config_dir() / "config.json")
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})
