"""Exporter format tests — all pure functions over the data model."""

from lyricsync.core.model import Line, LyricDocument, Syllable, Word
from lyricsync.exporters import export_ass, export_elrc, export_lrc, export_srt, export_vtt


def make_doc() -> LyricDocument:
    return LyricDocument(
        audio_source="song.mp3",
        metadata={"artist": "Artist", "title": "Title"},
        lines=[
            Line(start=12.34, end=15.10, words=[
                Word("hello", 12.34, 12.80, 0.9, syllables=[
                    Syllable("hel", 12.34, 12.55), Syllable("lo", 12.55, 12.80)]),
                Word("world", 12.90, 15.10, 0.8),
            ]),
            Line(start=61.5, end=63.0, words=[Word("again", 61.5, 63.0, 0.95)]),
        ],
    )


def test_lrc():
    out = export_lrc(make_doc())
    assert "[ar:Artist]" in out and "[ti:Title]" in out
    assert "[00:12.34]hello world" in out
    assert "[01:01.50]again" in out


def test_elrc_per_syllable():
    out = export_elrc(make_doc())
    assert "[00:12.34]" in out
    assert "<00:12.34>hel" in out          # first syllable
    assert "<00:12.55>lo" in out           # second syllable
    assert "<00:12.90>world" in out        # word without special handling
    # syllables of one word must not be space-separated
    line = [l for l in out.splitlines() if "hel" in l][0]
    assert "hel<" in line.replace(" ", "").replace("<00:12.55>", "<")


def test_elrc_word_level_fallback():
    doc = make_doc()
    for _, _, w in doc.iter_words():
        w.syllables = []
    out = export_elrc(doc)
    assert "<00:12.34>hello <00:12.90>world" in out


def test_srt():
    out = export_srt(make_doc())
    assert out.startswith("1\n00:00:12,340 --> 00:00:15,100\nhello world")
    assert "2\n00:01:01,500 --> 00:01:03,000\nagain" in out


def test_vtt():
    out = export_vtt(make_doc())
    assert out.startswith("WEBVTT")
    assert "00:00:12.340 --> 00:00:15.100" in out


def test_ass_karaoke_tags():
    out = export_ass(make_doc())
    assert "[Script Info]" in out and "Dialogue:" in out
    # hel spans 12.34→12.55 = 21cs; lo 12.55→12.80 = 25cs;
    # gap absorption extends 'lo' to world's start 12.90 = 35cs
    assert "{\\k21}hel" in out
    assert "{\\k35}lo " in out
    assert out.count("Dialogue:") == 2


def test_ass_escapes_braces():
    doc = make_doc()
    doc.lines[0].words[1].text = "wor{ld}"
    doc.lines[0].words[1].syllables = []
    assert "{\\k" in export_ass(doc) and "wor(ld)" in export_ass(doc)
