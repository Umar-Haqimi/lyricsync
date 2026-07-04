"""Cross-platform user data/config/cache locations via platformdirs.

The model cache directory deliberately defaults to the persistent user data
dir (NOT anywhere inside the project/build tree) so packaging or
redeployment can never wipe downloaded model weights. It can be overridden
in config (see lyricsync.config).
"""

from __future__ import annotations

from pathlib import Path

from platformdirs import user_cache_dir, user_config_dir, user_data_dir

from lyricsync import APP_AUTHOR, APP_NAME


def config_dir() -> Path:
    p = Path(user_config_dir(APP_NAME, APP_AUTHOR))
    p.mkdir(parents=True, exist_ok=True)
    return p


def data_dir() -> Path:
    p = Path(user_data_dir(APP_NAME, APP_AUTHOR))
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir() -> Path:
    p = Path(user_cache_dir(APP_NAME, APP_AUTHOR))
    p.mkdir(parents=True, exist_ok=True)
    return p


def default_model_dir() -> Path:
    p = data_dir() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def work_dir() -> Path:
    """Scratch space for converted WAVs and separation stems."""
    p = cache_dir() / "work"
    p.mkdir(parents=True, exist_ok=True)
    return p


def log_dir() -> Path:
    p = data_dir() / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p
