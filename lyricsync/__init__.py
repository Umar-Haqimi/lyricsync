"""LyricSync — local lyrics timing & karaoke generator.

Pipeline: audio preprocessing (ffmpeg) → vocal separation (audio-separator)
→ transcription (faster-whisper) → canonical timed-lyrics data model
→ exporters (LRC / eLRC / SRT / VTT / ASS) and metadata embedding (mutagen).
"""

__version__ = "0.1.0"

APP_NAME = "LyricSync"
APP_AUTHOR = "LyricSync"
