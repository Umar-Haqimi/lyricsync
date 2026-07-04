"""Data model round-trip and validation tests."""

import json

from lyricsync.core.model import Line, LyricDocument, Syllable, Word


def make_doc() -> LyricDocument:
    return LyricDocument(
        audio_source="song.mp3",
        language="en",
        metadata={"whisper_model": "small"},
        lines=[
            Line(
                start=12.34,
                end=15.10,
                words=[
                    Word(
                        text="hello",
                        start=12.34,
                        end=12.80,
                        confidence=0.91,
                        syllables=[
                            Syllable("hel", 12.34, 12.55),
                            Syllable("lo", 12.55, 12.80),
                        ],
                    ),
                    Word(text="world", start=12.90, end=15.10, confidence=0.42),
                ],
            )
        ],
    )


def test_round_trip(tmp_path):
    doc = make_doc()
    path = tmp_path / "doc.json"
    doc.save_json(path)
    loaded = LyricDocument.load_json(path)
    assert loaded.audio_source == "song.mp3"
    assert loaded.language == "en"
    assert len(loaded.lines) == 1
    line = loaded.lines[0]
    assert line.text == "hello world"
    assert line.words[0].syllables[1].text == "lo"
    assert abs(line.words[0].confidence - 0.91) < 1e-6
    assert line.start == 12.34 and line.end == 15.10


def test_json_shape_matches_spec(tmp_path):
    doc = make_doc()
    path = tmp_path / "doc.json"
    doc.save_json(path)
    raw = json.loads(path.read_text())
    word = raw["lines"][0]["words"][0]
    assert set(word) == {"text", "start", "end", "confidence", "syllables"}
    assert word["syllables"][0] == {"text": "hel", "start": 12.34, "end": 12.55}


def test_recompute_bounds():
    line = Line(words=[Word("a", 5.0, 5.5), Word("b", 6.0, 7.0)])
    line.recompute_bounds()
    assert line.start == 5.0 and line.end == 7.0


def test_validate_flags_inverted_times():
    doc = make_doc()
    doc.lines[0].words[0].end = 1.0  # before start
    problems = doc.validate()
    assert any("hello" in p for p in problems)


def test_iter_words():
    doc = make_doc()
    words = [w.text for _, _, w in doc.iter_words()]
    assert words == ["hello", "world"]
