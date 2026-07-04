# LyricSync

A local desktop app that takes a song and produces accurately time-stamped
lyrics — line-by-line (`.lrc`) or word-by-word / karaoke-style (`.elrc`),
with SRT/VTT/ASS export and synced-lyrics embedding straight into the audio
file's tags.

Single-user, fully local: no web server, no database. Persistence is plain
JSON files.

**Pipeline:** ffmpeg preprocessing → vocal separation (audio-separator) →
transcription (faster-whisper, word timestamps + VAD) → canonical timed-lyrics
model → manual correction UI → exporters / metadata embedding.

## Requirements

- **Python 3.11 or 3.12** (a `.python-version` pin for 3.12 is included; `uv` will fetch it automatically)
- **ffmpeg** on PATH (external binary):
  - Arch: `sudo pacman -S ffmpeg` · Debian/Ubuntu: `sudo apt install ffmpeg` · macOS: `brew install ffmpeg` · Windows: [ffmpeg.org](https://ffmpeg.org/download.html)
- Optional: an NVIDIA GPU. CUDA-enabled torch + onnxruntime-gpu are installed by
  default on Linux; everything degrades gracefully to CPU (the app auto-falls-back
  if CUDA init fails).

## Setup

```bash
cd "Lyrics & Karaoke Generator"
uv sync            # creates .venv with Python 3.12 + all dependencies
```

(or classic pip: `python3.12 -m venv .venv && .venv/bin/pip install -e .`)

## Usage

### GUI

```bash
uv run lyricsync
```

- **Transcribe tab** — pick/drop an audio file (MP3/WAV/FLAC/M4A/OGG/Opus…),
  choose a separation tier and Whisper model size, optionally paste official
  lyrics (or fetch from LRCLIB) to align them to the audio timing, hit
  *Transcribe*. Dropping multiple files sends them to the Queue.
- **Editor tab** — waveform with click-to-seek and playback sync; drag the
  green/orange markers to fix line timing (dashed markers = selected word);
  double-click words to correct text; amber/red words are low-confidence and
  worth double-checking (thresholds configurable in Settings). Export buttons
  and tag embedding live at the bottom.
- **Queue tab** — batch processing with per-job status, progress and cancel.
- **Models tab** — per-model disk usage, manual delete, and custom separation
  model install from a direct URL, Hugging Face repo+filename, or local file
  (Roformer/MDXC checkpoints often need a sidecar YAML config — there's a
  field for its URL). Downloads stream with a progress bar and resume from
  `.partial` files after interruption.
- **Settings tab** — model directory, confidence thresholds, syllable
  backend, batch parallelism.

### CLI (headless)

```bash
uv run lyricsync-cli song.mp3 --tier 2 --model small --lrc --elrc --embed
uv run lyricsync-cli song.mp3 --skip-separation --align-text lyrics.txt --ass
uv run lyricsync-cli *.mp3 --lrc            # batch
```

Every run writes `<input>.lyricsync.json` (the canonical document) which the
GUI's *Open project JSON…* re-opens for later correction.

## Separation tiers

| Tier | Model | Notes |
|------|-------|-------|
| 1 — Potato | UVR-MDX-NET Inst HQ 3 | CPU-friendly, fast |
| 2 — Balanced | htdemucs_ft | Good default |
| 3 — Max Quality | MelBand Roformer INSTV8 (Gabox) | GPU-heavy, best quality |
| Custom | any audio-separator-compatible model | via Models tab |
| Skip | — | for already-clean/acapella tracks |

## First-run model downloads

Nothing is bundled. On first use:

- **Whisper weights** download from Hugging Face into its standard cache
  (`~/.cache/huggingface`). `tiny` ≈ 75 MB … `large-v3` ≈ 3 GB.
- **Separation weights** are fetched by audio-separator into the LyricSync
  model directory (`~/.local/share/LyricSync/models` on Linux; see the Models
  tab). Tier 1 ≈ 60 MB, tier 2 ≈ 320 MB×4, tier 3 ≈ 550 MB.

The model directory is persistent user data, deliberately outside the project
tree, and is never auto-evicted — delete models manually from the Models tab.

## Formats

- `.lrc` — `[mm:ss.xx]line`
- `.elrc` — enhanced LRC, `<mm:ss.xx>` per syllable (or per word if syllable
  splitting is disabled)
- `.srt` / `.vtt` — standard subtitles, one cue per line
- `.ass` — karaoke `\k` color-wipe tags per syllable
- **Embedding** — MP3 gets a true synced ID3 `SYLT` frame plus `USLT`
  fallback; FLAC/OGG/Opus get LRC text in the `LYRICS` Vorbis comment.
  There is **no true synced-lyrics standard for Vorbis comments** — many
  players parse LRC text out of `LYRICS`, but it's a convention, not a spec
  (the app tells you this when embedding).

## Notes & limitations

- Timing quality depends heavily on separation tier + Whisper size. For real
  songs, tier 2 + `small`/`medium` is a good starting point; `tiny` is only
  for smoke tests.
- Syllable timing is interpolated within each word (proportional to syllable
  length), then hand-adjustable in the editor.
- The lyrics-alignment mode uses word-level dynamic-programming matching
  between the official text and Whisper's transcript (unmatched words are
  interpolated and flagged at confidence 0). The `AlignmentEngine` interface
  is designed so a forced-alignment backend (stable-ts / wav2vec2-CTC à la
  WhisperX) can be dropped in later.
- Syllable splitting is behind a `SyllableSplitter` interface with two
  backends: `hybrid` (default — pyphen + rule-based fallback + override
  table, ported from prior custom syllabification work) and plain `pyphen`.

## Development

```bash
uv sync --extra dev
uv run pytest              # unit tests (model, exporters, syllables, alignment, manifest)
```

Module layout:

```
lyricsync/
  core/        pipeline logic: audio, separation, transcription, alignment,
               syllables, canonical data model (model.py)
  models/      manifest + resumable downloader + catalog bridge
  exporters/   LRC/eLRC/SRT/VTT/ASS + mutagen embedding (pure functions)
  gui/         PySide6: main window, waveform editor, queue, models, settings
  cli.py       headless entry point
```

Packaging for distribution: `uvx pyinstaller --windowed --name LyricSync -p . lyricsync/app.py`
(model downloads stay in user data dirs, so packaged builds share the same cache).

Logs: `~/.local/share/LyricSync/logs/lyricsync.log`.
