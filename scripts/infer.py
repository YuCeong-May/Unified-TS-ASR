#!/usr/bin/env python3
"""Run single-utterance inference with a released checkpoint directory."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unified_ts_asr import ModelBundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--audio", required=True, help="16 kHz mixture waveform")
    parser.add_argument("--text-cue", default=None)
    parser.add_argument("--enrollment-audio", default=None)
    parser.add_argument("--enrollment-max-seconds", type=float, default=5.0)
    parser.add_argument("--device", default=None, help="cuda:0, cpu, or automatic")
    parser.add_argument("--allow-partial-load", action="store_true")
    args = parser.parse_args()

    bundle = ModelBundle(
        args.model_dir,
        device=args.device,
        strict=not args.allow_partial_load,
    )
    result = bundle.transcribe(
        args.audio,
        text_cue=args.text_cue,
        enrollment_audio=args.enrollment_audio,
        enrollment_max_seconds=args.enrollment_max_seconds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
