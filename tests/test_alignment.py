"""DP text alignment tests: official wording + Whisper timing."""

from lyricsync.core.alignment import DPTextAligner
from lyricsync.core.model import Line, LyricDocument, Word


def make_transcription() -> LyricDocument:
    # Whisper misheard "hold me closer" as "hold me colder"
    return LyricDocument(
        audio_source="song.mp3",
        language="en",
        lines=[
            Line(words=[Word("hold", 10.0, 10.4, 0.9),
                        Word("me", 10.4, 10.6, 0.95),
                        Word("colder", 10.6, 11.2, 0.4)]),
            Line(words=[Word("tiny", 20.0, 20.5, 0.9),
                        Word("dancer", 20.5, 21.1, 0.92)]),
        ],
    )


def test_matched_words_inherit_timing():
    doc = DPTextAligner().align(make_transcription(), "Hold me closer\nTiny dancer")
    assert len(doc.lines) == 2
    hold = doc.lines[0].words[0]
    assert hold.text == "Hold"
    assert (hold.start, hold.end) == (10.0, 10.4)
    dancer = doc.lines[1].words[1]
    assert (dancer.start, dancer.end) == (20.5, 21.1)


def test_unmatched_words_interpolated_and_flagged():
    doc = DPTextAligner().align(make_transcription(), "Hold me closer\nTiny dancer")
    closer = doc.lines[0].words[2]
    assert closer.text == "closer"
    assert closer.confidence == 0.0            # flagged for review
    assert 10.6 <= closer.start < closer.end <= 20.0  # inside the gap


def test_line_structure_follows_official_text():
    doc = DPTextAligner().align(
        make_transcription(), "Hold me\ncloser Tiny dancer")
    assert [l.text for l in doc.lines] == ["Hold me", "closer Tiny dancer"]


def test_match_rate_metadata():
    doc = DPTextAligner().align(make_transcription(), "Hold me closer\nTiny dancer")
    assert 0.5 < doc.metadata["alignment_match_rate"] <= 1.0
