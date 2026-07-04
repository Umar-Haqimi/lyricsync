"""Syllable splitter tests — including the ported custom hybrid rules."""

import pytest

from lyricsync.core.model import Word
from lyricsync.core.syllables import (
    HybridSplitter,
    PyphenSplitter,
    make_splitter,
    split_word_timed,
)


@pytest.fixture(scope="module")
def hybrid():
    return HybridSplitter("en")


def test_override_words(hybrid):
    # pyphen alone gets these wrong; the override table fixes them
    assert hybrid.split("times") == ["times"]
    assert hybrid.split("chance") == ["chance"]
    assert hybrid.split("every") == ["ev", "ery"]
    assert hybrid.split("finally") == ["fi", "nal", "ly"]
    assert hybrid.split("whenever") == ["when", "ev", "er"]


def test_override_preserves_case(hybrid):
    assert hybrid.split("Every") == ["Ev", "ery"]


def test_rule_based_fallback_short_words(hybrid):
    # pyphen refuses these; the vowel-cluster fallback splits them
    assert hybrid.split("Mama") == ["Ma", "ma"]
    assert hybrid.split("only") == ["on", "ly"]


def test_silent_e_monosyllabic(hybrid):
    assert hybrid.split("love") == ["love"]
    assert hybrid.split("make") == ["make"]
    assert hybrid.split("one") == ["one"]


def test_contractions_not_split(hybrid):
    assert hybrid.split("don't") == ["don't"]
    assert hybrid.split("I've") == ["I've"]


def test_concatenation_invariant(hybrid):
    for word in ["hello", "beautiful", "don't", "Mama", "extraordinary", "a", "1234"]:
        assert "".join(hybrid.split(word)) == word


def test_pyphen_backend():
    splitter = PyphenSplitter("en")
    assert "".join(splitter.split("beautiful")) == "beautiful"
    assert len(splitter.split("beautiful")) >= 2


def test_make_splitter():
    assert make_splitter("pyphen").name == "pyphen"
    assert make_splitter("hybrid").name == "hybrid"
    assert make_splitter("anything-else").name == "hybrid"


def test_split_word_timed_covers_duration(hybrid):
    word = Word(text="hello", start=10.0, end=11.0, confidence=0.9)
    syls = split_word_timed(word, hybrid)
    assert len(syls) == 2
    assert syls[0].start == 10.0
    assert syls[-1].end == 11.0
    assert syls[0].end == syls[1].start          # contiguous
    assert syls[0].end < syls[1].end


def test_split_word_timed_monosyllable(hybrid):
    word = Word(text="love", start=1.0, end=2.0)
    syls = split_word_timed(word, hybrid)
    assert len(syls) == 1
    assert (syls[0].start, syls[0].end) == (1.0, 2.0)
