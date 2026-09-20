#!/usr/bin/env python3
"""Evaluate a released checkpoint on one JSONL manifest."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


BACKENDS = {
    "KwsAsrConformerDae": "evaluate_text.py",
    "SpeakerConformerCTCOnlineEnroll": "evaluate_speaker.py",
    "SpeakerConformerCTCEmbeddingFree": "evaluate_speaker.py",
    "DualCueConformerCTC": "evaluate_unified.py",
}


def resolve_audio(value, base: Path):
    if not value:
        return value
    path = Path(str(value)).expanduser()
    return str(path if path.is_absolute() else (base / path).resolve())


def normalize_manifest(source: Path, destination: Path, dataset_root: Path) -> None:
    base = dataset_root / "ts-aishell" if (dataset_root / "ts-aishell").is_dir() else dataset_root
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open(encoding="utf-8") as reader, destination.open("w", encoding="utf-8") as writer:
        for line_no, line in enumerate(reader, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "source" not in row:
                raise ValueError(f"{source}:{line_no} has no source field")
            row["source"] = resolve_audio(row["source"], base)
            if row.get("enroll_source"):
                row["enroll_source"] = resolve_audio(row["enroll_source"], base)
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cue", choices=("both", "text", "enroll", "none"), default="both")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    import yaml

    model_dir = Path(args.model_dir).resolve()
    config = model_dir / "config.yaml"
    checkpoint = model_dir / "model.pt"
    with config.open(encoding="utf-8") as source:
        model_name = str(yaml.safe_load(source)["model"])
    try:
        backend = Path(__file__).resolve().parent / "backends" / BACKENDS[model_name]
    except KeyError as exc:
        raise SystemExit(f"Unsupported released model: {model_name}") from exc

    output_dir = Path(args.output_dir).resolve()
    normalized = output_dir / "resolved_manifest.jsonl"
    normalize_manifest(
        Path(args.manifest).resolve(), normalized, Path(args.dataset_root).resolve()
    )
    command = [
        sys.executable,
        str(backend),
        "--config", str(config),
        "--init_param", str(checkpoint),
        "--test_sets", f"evaluation:{normalized}",
        "--output_dir", str(output_dir),
        "--gpus", args.gpus,
        "--batch_size", str(args.batch_size),
        "--limit", str(args.limit),
    ]
    if model_name == "DualCueConformerCTC":
        command += ["--cue", args.cue, "--cue_text_field", "cue_text"]
    environment = dict(os.environ)
    root = str(Path(__file__).resolve().parents[1])
    environment["PYTHONPATH"] = root + os.pathsep + environment.get("PYTHONPATH", "")
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=environment)


if __name__ == "__main__":
    main()
