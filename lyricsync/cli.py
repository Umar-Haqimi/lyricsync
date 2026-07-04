"""Headless CLI: run the full pipeline on one or more files.

Examples:
    lyricsync-cli song.mp3                        # JSON next to the song
    lyricsync-cli song.mp3 --tier 3 --model medium --lrc --elrc --ass
    lyricsync-cli song.mp3 --skip-separation --json out.json
    lyricsync-cli *.mp3 --lrc --embed             # batch + embed tags
    lyricsync-cli song.mp3 --align-text official_lyrics.txt --elrc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lyricsync.config import AppConfig
from lyricsync.core.alignment import make_engine
from lyricsync.core.model import LyricDocument
from lyricsync.core.pipeline import PipelineOptions, run_pipeline
from lyricsync.core.separation import TIERS
from lyricsync.core.transcription import COMPUTE_TYPES, WHISPER_SIZES
from lyricsync.exporters import EXPORTERS
from lyricsync.exporters.embed import embed_lyrics
from lyricsync.utils.logs import get_logger, setup_logging

log = get_logger("cli")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lyricsync-cli",
        description="Generate time-stamped lyrics (LRC/eLRC/SRT/VTT/ASS) from audio files.",
    )
    p.add_argument("inputs", nargs="+", help="Audio file(s): mp3/wav/flac/m4a/ogg…")

    sep = p.add_argument_group("separation")
    tier_help = "; ".join(f"{t.tier}={t.label}" for t in TIERS.values())
    sep.add_argument("--tier", type=int, default=None, choices=[1, 2, 3, 4],
                     help=f"Vocal separation tier ({tier_help}; 4=custom)")
    sep.add_argument("--skip-separation", action="store_true",
                     help="Transcribe the mix directly (already-clean tracks)")
    sep.add_argument("--custom-model", default="",
                     help="Model filename for --tier 4 (must exist in the model dir)")
    sep.add_argument("--normalize", action="store_true",
                     help="Loudness-normalize (EBU R128) before processing")

    tr = p.add_argument_group("transcription")
    tr.add_argument("--model", default=None, choices=WHISPER_SIZES,
                    help="faster-whisper model size")
    tr.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    tr.add_argument("--compute-type", default=None, choices=COMPUTE_TYPES)
    tr.add_argument("--language", default=None,
                    help="ISO language code (default: autodetect)")
    tr.add_argument("--min-silence-ms", type=int, default=None,
                    help="VAD min silence duration (default 500)")

    al = p.add_argument_group("alignment (optional)")
    al.add_argument("--align-text", metavar="FILE",
                    help="Align official lyrics text file to the audio timing")

    out = p.add_argument_group("output")
    out.add_argument("--json", metavar="FILE", help="Output JSON path "
                     "(default: <input>.lyricsync.json next to the input)")
    for fmt in EXPORTERS:
        out.add_argument(f"--{fmt}", action="store_true", help=f"Also write .{fmt}")
    out.add_argument("--embed", action="store_true",
                     help="Embed synced lyrics into the audio file's tags")
    out.add_argument("--no-syllables", action="store_true",
                     help="Skip syllable splitting")

    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(verbose=args.verbose)
    cfg = AppConfig.load()

    options = PipelineOptions.from_config(cfg)
    if args.skip_separation:
        options.separation_tier = 0
    elif args.tier is not None:
        options.separation_tier = args.tier
    if args.custom_model:
        options.custom_model_filename = args.custom_model
    if args.normalize:
        options.normalize_loudness = True
    if args.model:
        options.whisper_model = args.model
    if args.device:
        options.device = args.device
    if args.compute_type:
        options.compute_type = args.compute_type
    if args.language is not None:
        options.language = args.language
    if args.min_silence_ms is not None:
        options.vad_min_silence_ms = args.min_silence_ms
    options.split_syllables = not args.no_syllables

    official_text = None
    if args.align_text:
        official_text = Path(args.align_text).read_text(encoding="utf-8")

    failures = 0
    for source in args.inputs:
        source = Path(source)
        if not source.exists():
            print(f"error: {source} does not exist", file=sys.stderr)
            failures += 1
            continue
        try:
            doc = _process_one(source, options, official_text, args)
        except Exception as e:
            log.exception("pipeline failed for %s", source)
            print(f"error: {source.name}: {e}", file=sys.stderr)
            failures += 1
            continue
        print(f"ok: {source.name} — {len(doc.lines)} lines "
              f"({doc.language or 'unknown language'})")
    return 1 if failures else 0


def _progress(msg: str, frac: float) -> None:
    print(f"\r[{frac * 100:5.1f}%] {msg:<60}", end="", flush=True)
    if frac >= 1.0:
        print()


def _process_one(
    source: Path, options: PipelineOptions, official_text: str | None, args
) -> LyricDocument:
    doc = run_pipeline(source, options, progress=_progress)

    if official_text:
        doc = make_engine().align(doc, official_text)
        if not args.no_syllables:
            from lyricsync.core.syllables import make_splitter, split_document
            split_document(doc, make_splitter(options.syllable_backend,
                                              options.syllable_lang))

    json_path = Path(args.json) if args.json and len(args.inputs) == 1 \
        else source.with_suffix(source.suffix + ".lyricsync.json")
    doc.save_json(json_path)
    print(f"wrote {json_path}")

    for fmt, (exporter, ext) in EXPORTERS.items():
        if getattr(args, fmt):
            out_path = source.with_suffix(ext)
            out_path.write_text(exporter(doc), encoding="utf-8")
            print(f"wrote {out_path}")

    if args.embed:
        result = embed_lyrics(doc, source)
        print(("embedded: " if result.ok else "embed failed: ") + result.message)

    return doc


if __name__ == "__main__":
    sys.exit(main())
