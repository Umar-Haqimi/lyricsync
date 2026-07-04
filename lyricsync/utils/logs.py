"""App-wide logging: console + rotating file in the user data dir.

Background workers log through here so separation/transcription/download
failures are diagnosable after the fact, not silent.
"""

from __future__ import annotations

import logging
import logging.handlers

from lyricsync.utils.paths import log_dir

_CONFIGURED = False


def setup_logging(verbose: bool = False) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    root = logging.getLogger("lyricsync")
    root.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(console)

    logfile = log_dir() / "lyricsync.log"
    file_handler = logging.handlers.RotatingFileHandler(
        logfile, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"lyricsync.{name}")
