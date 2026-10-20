"""Transcribe your own audio files with the published model, on a Mac (MPS), an NVIDIA GPU or the CPU.

Any format ffmpeg reads is decoded to 16 kHz mono, cut at the quietest 100 ms between 15 and 25 s
into each piece (the same chunking as run_open_models.py and the model card), and each piece is
transcribed with Finnish forced. No pipeline chunk_length_s and no return_timestamps: the model card
measured both as harmful for this model.

torch and transformers are not project dependencies (the GPU stacks are pinned in lumi/setup.sh), so
add them for the run with --with:

    uv run --with torch --with transformers python scripts/transcribe_local.py interview.mp3
    uv run --with torch --with transformers python scripts/transcribe_local.py *.wav --out transcripts/
    uv run --with torch --with transformers python scripts/transcribe_local.py talk.m4a --device cpu

Output: each piece as "[start s] text" on stdout, or <out>/<file stem>.txt with --out. Needs ffmpeg on PATH.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_open_models import SR, chunk_bounds  # noqa: E402

MODEL = "GAIK-project/whisper-large-v3-fi-dialects"


def pick_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def decode(path: Path):
    """Any audio file to 16 kHz mono float32 with ffmpeg."""
    import numpy as np  # comes with transformers

    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype="float32")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("audio", nargs="+", type=Path)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=None, help="a Hugging Face revision, e.g. v1.0")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--fp32", action="store_true", help="float32 on a GPU too (if float16 misbehaves on MPS)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None, help="a folder for <stem>.txt instead of stdout")
    args = ap.parse_args()

    import torch
    from transformers import pipeline

    dev = pick_device(args.device)
    dtype = torch.float32 if args.fp32 or dev == "cpu" else torch.float16
    print(f"model={args.model} device={dev} dtype={dtype}", file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    asr = pipeline("automatic-speech-recognition", model=args.model, revision=args.revision, dtype=dtype, device=dev)
    print(f"loaded in {time.perf_counter() - t0:.1f} s", file=sys.stderr, flush=True)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
    for path in args.audio:
        audio = decode(path)
        bounds = chunk_bounds(audio)
        t0 = time.perf_counter()
        out = asr(
            [{"raw": audio[a:b], "sampling_rate": SR} for a, b in bounds],
            batch_size=args.batch_size,
            generate_kwargs={"language": "fi", "task": "transcribe"},
        )
        took = time.perf_counter() - t0
        lines = [f"[{a / SR:7.1f} s] {o['text'].strip()}" for (a, _), o in zip(bounds, out, strict=True)]
        secs = len(audio) / SR
        print(f"{path.name}: {secs:.0f} s audio in {took:.1f} s ({secs / took:.1f}x real time)", file=sys.stderr)
        if args.out:
            (args.out / f"{path.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        else:
            print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
