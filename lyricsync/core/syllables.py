"""Syllable splitting behind a swappable SyllableSplitter interface.

Two backends:

* PyphenSplitter — plain pyphen hyphenation.
* HybridSplitter — ported from the user's existing syllabification script
  (pyphen + word-override dictionary + a vowel-cluster rule-based fallback
  that handles contraction tails, silent '-ed' and silent trailing 'e').
  This is the default: pyphen alone refuses to split many short words
  ("Mama", "only") and mis-splits others ("times" → "ti|mes").

`split_document` interpolates per-syllable timestamps across each word's
duration, weighted by syllable length.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

import pyphen

from lyricsync.core.model import LyricDocument, Syllable, Word

VOWELS_SET = set("aeiouy")

# Words where pyphen gives wrong results — keyed by lowercase.
WORD_OVERRIDES = {
    "times": "times",          # monosyllabic; pyphen splits as ti|mes
    "every": "ev|ery",         # pyphen: e|ve|ry (3) instead of ev|ery (2)
    "finally": "fi|nal|ly",    # pyphen: fi|nally (2) instead of fi|nal|ly (3)
    "chance": "chance",        # monosyllabic; pyphen splits as chan|ce
    "whenever": "when|ev|er",  # pyphen: whenev|er (2) instead of when|ev|er (3)
}


class SyllableSplitter(ABC):
    """Backend interface: split a single word into syllable strings."""

    name: str = "base"

    @abstractmethod
    def split(self, word: str) -> list[str]:
        """Return syllables of `word` (concatenation must equal the input)."""


class PyphenSplitter(SyllableSplitter):
    """Plain pyphen hyphenation-based estimation."""

    name = "pyphen"

    def __init__(self, lang: str = "en"):
        self._dic = pyphen.Pyphen(lang=lang)

    def split(self, word: str) -> list[str]:
        core = word.strip()
        if not core:
            return [word]
        return self._dic.inserted(core, hyphen="|").split("|")


class HybridSplitter(SyllableSplitter):
    """Pyphen + overrides + rule-based fallback (ported user script)."""

    name = "hybrid"

    def __init__(self, lang: str = "en"):
        self._dic = pyphen.Pyphen(lang=lang)

    def split(self, word: str) -> list[str]:
        result = self._syllabify_word(word)
        parts = result.split("|")
        # Safety: the pattern must reassemble to the original token.
        return parts if "".join(parts) == word else [word]

    # --- ported logic ------------------------------------------------

    @staticmethod
    def _apply_case(original: str, pattern: str) -> str:
        """Re-apply the casing from `original` onto a syllabified pattern."""
        result = []
        orig_idx = 0
        for ch in pattern:
            if ch == "|":
                result.append("|")
            else:
                result.append(original[orig_idx] if orig_idx < len(original) else ch)
                orig_idx += 1
        return "".join(result)

    @staticmethod
    def _rule_based(word: str) -> str:
        """Vowel-cluster fallback for words pyphen refuses to split.

        Handles contraction tails (I've, don't), silent '-ed' (walked, but
        not wanted/needed) and silent trailing 'e' (love, make, one).
        """
        core = word.lower()
        effective = core  # only ever chopped from the end → stays a prefix

        apos = effective.find("'")
        if apos != -1:
            effective = effective[:apos]
        elif len(effective) > 3 and effective.endswith("ed") and effective[-3] not in "td":
            effective = effective[:-2]
        elif (
            len(effective) >= 3
            and effective[-1] == "e"
            and effective[-2] not in VOWELS_SET
            and effective[-3] in VOWELS_SET
        ):
            effective = effective[:-1]

        in_vowel, vowel_groups, start = False, [], None
        for i, c in enumerate(effective):
            if c in VOWELS_SET:
                if not in_vowel:
                    start, in_vowel = i, True
            elif in_vowel:
                vowel_groups.append((start, i))
                in_vowel = False
        if in_vowel:
            vowel_groups.append((start, len(effective)))

        if len(vowel_groups) <= 1:
            return word  # monosyllabic — leave as-is

        splits = []
        for idx in range(len(vowel_groups) - 1):
            v_end = vowel_groups[idx][1]
            gap = vowel_groups[idx + 1][0] - v_end
            splits.append(v_end if gap <= 1 else v_end + 1)

        parts, prev = [], 0
        for pos in splits:
            parts.append(word[prev:pos])
            prev = pos
        parts.append(word[prev:])  # trailing tail attaches to last syllable
        return "|".join(parts)

    def _syllabify_word(self, word: str) -> str:
        m = re.match(r"^([^a-zA-Z']*)([a-zA-Z'\-]+)([^a-zA-Z']*)$", word)
        if not m:
            return word  # numbers, ellipses, etc. — leave alone

        prefix, core, suffix = m.groups()
        while core.endswith("-"):
            suffix = "-" + suffix
            core = core[:-1]
        if not core:
            return word

        override = WORD_OVERRIDES.get(core.lower())
        if override is not None:
            return prefix + self._apply_case(core, override) + suffix

        # Mid-word contractions: syllabify only the base, reattach the tail
        # (prevents pyphen from splitting "There's" → "The|re's").
        apos_pos = core.find("'")
        if apos_pos > 0:
            base, tail = core[:apos_pos], core[apos_pos:]
            result = self._dic.inserted(base, hyphen="|")
            if result == base:
                result = self._rule_based(base)
            return prefix + result + tail + suffix

        result = self._dic.inserted(core, hyphen="|")
        if result == core:
            result = self._rule_based(core)
        return prefix + result + suffix


def make_splitter(backend: str = "hybrid", lang: str = "en") -> SyllableSplitter:
    if backend == "pyphen":
        return PyphenSplitter(lang)
    return HybridSplitter(lang)


def split_word_timed(word: Word, splitter: SyllableSplitter) -> list[Syllable]:
    """Split a word and interpolate timestamps across its duration.

    Time is distributed proportionally to syllable character length —
    crude but stable, and always exactly covers [word.start, word.end].
    """
    parts = [p for p in splitter.split(word.text) if p]
    if not parts:
        return []
    total_chars = sum(len(p) for p in parts)
    duration = word.duration
    syllables: list[Syllable] = []
    t = word.start
    for i, part in enumerate(parts):
        frac = len(part) / total_chars if total_chars else 1.0 / len(parts)
        end = word.end if i == len(parts) - 1 else min(word.end, t + duration * frac)
        syllables.append(Syllable(text=part, start=round(t, 3), end=round(end, 3)))
        t = end
    return syllables


def split_document(doc: LyricDocument, splitter: SyllableSplitter) -> None:
    """Populate `syllables` on every word in the document (in place)."""
    for _, _, word in doc.iter_words():
        word.syllables = split_word_timed(word, splitter)
    doc.metadata["syllable_backend"] = splitter.name
