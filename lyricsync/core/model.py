"""Canonical in-memory / JSON representation of timed lyrics.

Every feature in the app reads from or writes to this structure:
transcription populates it, the correction UI mutates it, the syllable
splitter enriches it, and every exporter / the metadata embedder consumes
it as a pure function. Nothing downstream ever touches raw Whisper output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass
class Syllable:
    text: str
    start: float
    end: float

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "start": round(self.start, 3), "end": round(self.end, 3)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Syllable":
        return cls(text=d["text"], start=float(d["start"]), end=float(d["end"]))


@dataclass
class Word:
    text: str
    start: float
    end: float
    confidence: float = 1.0
    syllables: list[Syllable] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "text": self.text,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "confidence": round(self.confidence, 4),
        }
        if self.syllables:
            d["syllables"] = [s.to_dict() for s in self.syllables]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Word":
        return cls(
            text=d["text"],
            start=float(d["start"]),
            end=float(d["end"]),
            confidence=float(d.get("confidence", 1.0)),
            syllables=[Syllable.from_dict(s) for s in d.get("syllables", [])],
        )


@dataclass
class Line:
    words: list[Word] = field(default_factory=list)
    # Explicit bounds so a line can extend past its words (e.g. after manual
    # edits or for instrumental lead-in); kept in sync by recompute_bounds().
    start: float = 0.0
    end: float = 0.0

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def recompute_bounds(self) -> None:
        if self.words:
            self.start = min(w.start for w in self.words)
            self.end = max(w.end for w in self.words)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "words": [w.to_dict() for w in self.words],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Line":
        line = cls(
            words=[Word.from_dict(w) for w in d.get("words", [])],
            start=float(d.get("start", 0.0)),
            end=float(d.get("end", 0.0)),
        )
        if line.start == 0.0 and line.end == 0.0:
            line.recompute_bounds()
        return line


@dataclass
class LyricDocument:
    audio_source: str = ""
    language: str = ""
    lines: list[Line] = field(default_factory=list)
    # Free-form provenance: model sizes, separation tier, normalization flag…
    metadata: dict[str, Any] = field(default_factory=dict)

    def iter_words(self) -> Iterator[tuple[int, int, Word]]:
        """Yield (line_index, word_index, word) across the document."""
        for li, line in enumerate(self.lines):
            for wi, word in enumerate(line.words):
                yield li, wi, word

    @property
    def duration(self) -> float:
        return self.lines[-1].end if self.lines else 0.0

    def sort_lines(self) -> None:
        self.lines.sort(key=lambda l: l.start)

    def validate(self) -> list[str]:
        """Return human-readable consistency problems (empty = OK)."""
        problems: list[str] = []
        for li, line in enumerate(self.lines):
            if line.end < line.start:
                problems.append(f"line {li + 1}: end before start")
            for wi, word in enumerate(line.words):
                if word.end < word.start:
                    problems.append(f"line {li + 1} word {wi + 1} ({word.text!r}): end before start")
                for syl in word.syllables:
                    if syl.end < syl.start:
                        problems.append(
                            f"line {li + 1} word {wi + 1} ({word.text!r}): syllable {syl.text!r} end before start"
                        )
        return problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio_source": self.audio_source,
            "language": self.language,
            "metadata": self.metadata,
            "lines": [l.to_dict() for l in self.lines],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LyricDocument":
        return cls(
            audio_source=d.get("audio_source", ""),
            language=d.get("language", ""),
            lines=[Line.from_dict(l) for l in d.get("lines", [])],
            metadata=d.get("metadata", {}),
        )

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "LyricDocument":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
