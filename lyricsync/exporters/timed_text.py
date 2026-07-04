"""LRC / eLRC / SRT / VTT / ASS exporters.

All are pure functions over LyricDocument. Timestamp formatting helpers
are kept per-format since each has its own quirks (LRC centiseconds,
SRT commas, VTT dots, ASS centisecond \\k karaoke tags).
"""

from __future__ import annotations

from lyricsync.core.model import Line, LyricDocument


# --- timestamp helpers -------------------------------------------------

def _lrc_ts(t: float) -> str:
    t = max(0.0, t)
    m = int(t // 60)
    s = t - m * 60
    return f"{m:02d}:{s:05.2f}"


def _srt_ts(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = round((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _vtt_ts(t: float) -> str:
    return _srt_ts(t).replace(",", ".")


def _ass_ts(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    cs = round((t - int(t)) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _metadata_header_lines(doc: LyricDocument) -> list[str]:
    meta = doc.metadata
    out = []
    if meta.get("artist"):
        out.append(f"[ar:{meta['artist']}]")
    if meta.get("title"):
        out.append(f"[ti:{meta['title']}]")
    if meta.get("album"):
        out.append(f"[al:{meta['album']}]")
    out.append("[re:LyricSync]")
    return out


# --- LRC / eLRC ---------------------------------------------------------

def export_lrc(doc: LyricDocument) -> str:
    """Line-by-line LRC: [mm:ss.xx]text"""
    lines = _metadata_header_lines(doc)
    for line in doc.lines:
        if line.words:
            lines.append(f"[{_lrc_ts(line.start)}]{line.text}")
    return "\n".join(lines) + "\n"


def export_elrc(doc: LyricDocument, per_syllable: bool = True) -> str:
    """Enhanced LRC: [mm:ss.xx]<mm:ss.xx>unit <mm:ss.xx>unit …

    Units are syllables when available (karaoke-style), else words.
    """
    lines = _metadata_header_lines(doc)
    for line in doc.lines:
        if not line.words:
            continue
        parts = [f"[{_lrc_ts(line.start)}]"]
        for wi, word in enumerate(line.words):
            if per_syllable and word.syllables:
                for si, syl in enumerate(word.syllables):
                    sep = "" if si < len(word.syllables) - 1 else " "
                    parts.append(f"<{_lrc_ts(syl.start)}>{syl.text}{sep}")
            else:
                parts.append(f"<{_lrc_ts(word.start)}>{word.text} ")
        lines.append("".join(parts).rstrip())
    return "\n".join(lines) + "\n"


# --- SRT / VTT ----------------------------------------------------------

def export_srt(doc: LyricDocument) -> str:
    blocks = []
    for i, line in enumerate(l for l in doc.lines if l.words):
        blocks.append(f"{i + 1}\n{_srt_ts(line.start)} --> {_srt_ts(line.end)}\n{line.text}\n")
    return "\n".join(blocks)


def export_vtt(doc: LyricDocument) -> str:
    blocks = ["WEBVTT", ""]
    for line in doc.lines:
        if line.words:
            blocks.append(f"{_vtt_ts(line.start)} --> {_vtt_ts(line.end)}\n{line.text}\n")
    return "\n".join(blocks)


# --- ASS (karaoke) --------------------------------------------------------

_ASS_HEADER = """[Script Info]
Title: {title}
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,Arial,64,&H00FFFFFF,&H000088EF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,0,2,60,60,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def export_ass(doc: LyricDocument) -> str:
    """ASS with per-syllable \\k color-wipe karaoke tags.

    \\k durations are in centiseconds. Gaps between units (breaths, pauses)
    are absorbed into the preceding unit so the wipe stays continuous.
    """
    title = doc.metadata.get("title") or "Karaoke"
    events = []
    for line in doc.lines:
        if not line.words:
            continue
        units: list[tuple[str, float, float]] = []  # (text, start, end)
        for wi, word in enumerate(line.words):
            trailing = " " if wi < len(line.words) - 1 else ""
            if word.syllables:
                for si, syl in enumerate(word.syllables):
                    text = syl.text + (trailing if si == len(word.syllables) - 1 else "")
                    units.append((text, syl.start, syl.end))
            else:
                units.append((word.text + trailing, word.start, word.end))

        parts = []
        for ui, (text, start, end) in enumerate(units):
            # Absorb the gap to the next unit for a continuous wipe.
            unit_end = units[ui + 1][1] if ui < len(units) - 1 else end
            dur_cs = max(1, round((unit_end - start) * 100))
            parts.append(f"{{\\k{dur_cs}}}{_ass_escape(text)}")
        events.append(
            f"Dialogue: 0,{_ass_ts(line.start)},{_ass_ts(line.end)},Karaoke,,0,0,0,,"
            + "".join(parts)
        )
    return _ASS_HEADER.format(title=title) + "\n".join(events) + "\n"


def _ass_escape(text: str) -> str:
    return text.replace("{", "(").replace("}", ")").replace("\n", " ")
