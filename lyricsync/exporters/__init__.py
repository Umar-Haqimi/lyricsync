"""Exporters — pure functions LyricDocument → str (or file side-effects).

Every format is generated from the canonical data model, never from raw
Whisper output.
"""

from lyricsync.exporters.timed_text import (
    export_ass,
    export_elrc,
    export_lrc,
    export_srt,
    export_vtt,
)

EXPORTERS = {
    "lrc": (export_lrc, ".lrc"),
    "elrc": (export_elrc, ".elrc"),
    "srt": (export_srt, ".srt"),
    "vtt": (export_vtt, ".vtt"),
    "ass": (export_ass, ".ass"),
}

__all__ = ["EXPORTERS", "export_lrc", "export_elrc", "export_srt", "export_vtt", "export_ass"]
