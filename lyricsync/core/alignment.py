"""Lyrics alignment mode: align official lyrics text to Whisper's timing.

This is additive — the plain transcription pipeline works completely
without it. The `AlignmentEngine` interface exists so stronger backends
(stable-ts, wav2vec2-CTC forced alignment à la WhisperX) can be dropped in
later without touching the rest of the app; the built-in engine performs
word-level dynamic-programming alignment between the official text and the
Whisper transcript, transferring Whisper's timestamps onto the official
words. That fixes wording (Whisper mishears) while keeping timing, with no
extra model downloads.

Also includes a best-effort LRCLIB lookup as a convenience.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from difflib import SequenceMatcher

from lyricsync.core.model import Line, LyricDocument, Word
from lyricsync.utils.logs import get_logger

log = get_logger("alignment")


class AlignmentEngine(ABC):
    """Swappable backend: reconcile official lyrics with transcribed timing."""

    name: str = "base"

    @abstractmethod
    def align(self, doc: LyricDocument, official_text: str) -> LyricDocument:
        """Return a new document whose words are the official lyrics with
        timestamps derived from `doc` (the Whisper transcription)."""


def _norm(token: str) -> str:
    return re.sub(r"[^\w']", "", token.lower())


class DPTextAligner(AlignmentEngine):
    """Word-level DP alignment (difflib) between official text and transcript.

    Matched official words inherit the transcript word's timing and
    confidence. Unmatched runs of official words are interpolated across
    the gap between their timed neighbours. Line structure follows the
    official text's line breaks — which is what you want in an LRC.
    """

    name = "dp-text"

    def align(self, doc: LyricDocument, official_text: str) -> LyricDocument:
        trans_words = [w for _, _, w in doc.iter_words()]
        if not trans_words:
            raise ValueError("Transcription is empty — nothing to align against.")

        official_lines = [l for l in (s.strip() for s in official_text.splitlines()) if l]
        tokens: list[tuple[int, str]] = []  # (official line index, token)
        for li, line_text in enumerate(official_lines):
            for tok in line_text.split():
                tokens.append((li, tok))
        if not tokens:
            raise ValueError("Official lyrics text is empty.")

        a = [_norm(t) for _, t in tokens]
        b = [_norm(w.text) for w in trans_words]
        matcher = SequenceMatcher(a=a, b=b, autojunk=False)

        # For each official token, the matched transcript word (or None).
        matched: list[Word | None] = [None] * len(tokens)
        for block in matcher.get_matching_blocks():
            for k in range(block.size):
                matched[block.a + k] = trans_words[block.b + k]

        timed = self._interpolate(tokens, matched, trans_words)

        aligned = LyricDocument(
            audio_source=doc.audio_source,
            language=doc.language,
            metadata={**doc.metadata, "alignment_engine": self.name},
        )
        current: list[Word] = []
        current_li = 0
        for (li, _), word in zip(tokens, timed):
            if li != current_li and current:
                aligned.lines.append(self._make_line(current))
                current = []
            current_li = li
            current.append(word)
        if current:
            aligned.lines.append(self._make_line(current))

        match_rate = sum(1 for m in matched if m) / len(matched)
        aligned.metadata["alignment_match_rate"] = round(match_rate, 3)
        log.info("aligned %d official words, match rate %.0f%%", len(tokens), match_rate * 100)
        return aligned

    @staticmethod
    def _make_line(words: list[Word]) -> Line:
        line = Line(words=words)
        line.recompute_bounds()
        return line

    @staticmethod
    def _interpolate(
        tokens: list[tuple[int, str]],
        matched: list[Word | None],
        trans_words: list[Word],
    ) -> list[Word]:
        """Produce a timed Word for every official token.

        Unmatched tokens get timing linearly interpolated between the
        nearest matched neighbours (song start / end at the extremes) and
        confidence 0 so the editor flags them for review.
        """
        n = len(tokens)
        out: list[Word] = [None] * n  # type: ignore[list-item]
        for i, m in enumerate(matched):
            if m is not None:
                out[i] = Word(text=tokens[i][1], start=m.start, end=m.end,
                              confidence=m.confidence)

        song_start = trans_words[0].start
        song_end = trans_words[-1].end
        i = 0
        while i < n:
            if out[i] is not None:
                i += 1
                continue
            gap_start = i
            while i < n and out[i] is None:
                i += 1
            gap_end = i  # exclusive
            left_t = out[gap_start - 1].end if gap_start > 0 else song_start
            right_t = out[gap_end].start if gap_end < n else song_end
            if right_t <= left_t:
                right_t = left_t + 0.3 * (gap_end - gap_start)
            span = (right_t - left_t) / (gap_end - gap_start)
            for k in range(gap_start, gap_end):
                s = left_t + span * (k - gap_start)
                out[k] = Word(text=tokens[k][1], start=round(s, 3),
                              end=round(s + span, 3), confidence=0.0)
        return out


def make_engine(name: str = "dp-text") -> AlignmentEngine:
    # Future backends (stable-ts, wav2vec2 forced alignment) register here.
    return DPTextAligner()


def fetch_lrclib_lyrics(
    artist: str, title: str, album: str = "", duration: float = 0.0, timeout: float = 10.0
) -> str | None:
    """Best-effort plain-lyrics lookup from LRCLIB. Returns None on any failure —
    this is a convenience, never a hard dependency."""
    import httpx

    params = {"artist_name": artist, "track_name": title}
    if album:
        params["album_name"] = album
    if duration:
        params["duration"] = str(int(duration))
    try:
        r = httpx.get("https://lrclib.net/api/get", params=params, timeout=timeout,
                      headers={"User-Agent": "LyricSync/0.1 (local tool)"})
        if r.status_code == 200:
            return r.json().get("plainLyrics") or None
        log.info("LRCLIB lookup: HTTP %s", r.status_code)
    except httpx.HTTPError as e:
        log.info("LRCLIB lookup failed: %s", e)
    return None
