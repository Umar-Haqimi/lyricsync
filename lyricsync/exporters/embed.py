"""Embed synced lyrics into audio file metadata via mutagen.

MP3   : ID3 SYLT frame (true synced lyrics) + USLT fallback (plain text).
FLAC/ : Vorbis comments LYRICS (LRC text) + UNSYNCEDLYRICS. There is no
OGG     true synced-lyrics standard for Vorbis comments — most players
        that support anything here parse LRC-formatted text out of the
        LYRICS tag. The UI surfaces this limitation to the user.
"""

from __future__ import annotations

from pathlib import Path

from mutagen.flac import FLAC
from mutagen.id3 import ID3, SYLT, USLT, Encoding, ID3NoHeaderError
from mutagen.oggopus import OggOpus
from mutagen.oggvorbis import OggVorbis

from lyricsync.core.model import LyricDocument
from lyricsync.exporters.timed_text import export_lrc
from lyricsync.utils.logs import get_logger

log = get_logger("embed")

VORBIS_SYNC_NOTE = (
    "FLAC/OGG have no true synced-lyrics standard; LRC-formatted text was "
    "written to the LYRICS tag, which many players parse for timing."
)


class EmbedResult:
    def __init__(self, ok: bool, message: str):
        self.ok = ok
        self.message = message


def embed_lyrics(doc: LyricDocument, audio_path: str | Path | None = None) -> EmbedResult:
    """Write synced lyrics into the audio file's tags. Returns a result whose
    message notes any format limitations (shown in the UI)."""
    path = Path(audio_path or doc.audio_source)
    if not path.exists():
        return EmbedResult(False, f"Audio file not found: {path}")

    ext = path.suffix.lower()
    try:
        if ext == ".mp3":
            return _embed_mp3(doc, path)
        if ext == ".flac":
            return _embed_vorbis(doc, FLAC(str(path)), path)
        if ext == ".ogg":
            try:
                return _embed_vorbis(doc, OggVorbis(str(path)), path)
            except Exception:
                return _embed_vorbis(doc, OggOpus(str(path)), path)
        if ext == ".opus":
            return _embed_vorbis(doc, OggOpus(str(path)), path)
        return EmbedResult(False, f"Embedding not supported for {ext} files "
                                  "(supported: .mp3, .flac, .ogg, .opus).")
    except Exception as e:  # mutagen raises many exception types
        log.exception("embedding failed for %s", path)
        return EmbedResult(False, f"Embedding failed: {e}")


def _plain_text(doc: LyricDocument) -> str:
    return "\n".join(line.text for line in doc.lines if line.words)


def _embed_mp3(doc: LyricDocument, path: Path) -> EmbedResult:
    try:
        tags = ID3(str(path))
    except ID3NoHeaderError:
        tags = ID3()

    lang = (doc.language or "eng")[:3].ljust(3, "x")
    # SYLT sync payload: [(text, time_ms)], type 1 = lyrics, format 2 = ms.
    sync = [
        (line.text, int(round(line.start * 1000)))
        for line in doc.lines
        if line.words
    ]
    tags.delall("SYLT")
    tags.add(SYLT(encoding=Encoding.UTF8, lang=lang, format=2, type=1,
                  desc="LyricSync", text=sync))
    tags.delall("USLT")
    tags.add(USLT(encoding=Encoding.UTF8, lang=lang, desc="LyricSync",
                  text=_plain_text(doc)))
    tags.save(str(path))
    log.info("embedded SYLT (%d lines) + USLT into %s", len(sync), path.name)
    return EmbedResult(True, f"Embedded synced lyrics (SYLT, {len(sync)} lines) "
                             f"and plain fallback (USLT) into {path.name}.")


def _embed_vorbis(doc: LyricDocument, audio, path: Path) -> EmbedResult:
    audio["LYRICS"] = export_lrc(doc)
    audio["UNSYNCEDLYRICS"] = _plain_text(doc)
    audio.save()
    log.info("embedded LYRICS (LRC) into %s", path.name)
    return EmbedResult(True, f"Wrote LRC text into {path.name}'s LYRICS tag. "
                             + VORBIS_SYNC_NOTE)
