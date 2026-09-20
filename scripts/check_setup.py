#!/usr/bin/env python3
"""Check dependencies and the downloaded artifact layout without loading a model."""

import argparse
import importlib.util
from pathlib import Path


REQUIRED = ("torch", "torchaudio", "soundfile", "omegaconf", "hydra", "yaml")
MODEL_DIRS = (
    "speaker-cue-ts-asr-embedding",
    "speaker-cue-ts-asr-embedding-free",
    "speaker-cue-ts-asr-embedding-free-low-frame",
    "text-cue-ts-asr",
    "text-cue-ts-asr-neg-0p3",
    "text-cue-ts-asr-neg-0p3-sv-head",
    "text-cue-ts-asr-neg-0p3-sv-head-0p1-reg",
    "text-cue-ts-asr-neg-0p3-sv-head-0p5-reg-0p01",
    "text-cue-ts-asr-neg-0p5",
    "unified-ts-asr-concat",
    "unified-ts-asr-parallel",
)
DATASET_SUBSETS = (
    "aishell1-ts",
    "aishell-ios-ts",
    "aishell-android-ts",
    "aishell-mic-ts",
    "aishell-mix-ts",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()
    missing = [name for name in REQUIRED if importlib.util.find_spec(name) is None]
    root = Path(args.artifacts).resolve()
    checkpoint_root = root / "models" / "ckpt"
    dataset_root = root / "dataset" / "ts-aishell"
    checks = {
        "model checkpoint tree": checkpoint_root,
        "token list": checkpoint_root / "config" / "tokens_kws8408.json",
        "segmentation dictionary": checkpoint_root / "config" / "seg_dict_ext",
        "evaluation manifest tree": dataset_root,
    }
    for model_name in MODEL_DIRS:
        checks[f"{model_name}/config.yaml"] = checkpoint_root / model_name / "config.yaml"
        checks[f"{model_name}/model.pt"] = checkpoint_root / model_name / "model.pt"
    for subset in DATASET_SUBSETS:
        checks[f"dataset subset {subset}"] = dataset_root / subset
    failed = False
    if missing:
        failed = True
        print("Missing Python modules:", ", ".join(missing))
    for label, path in checks.items():
        state = "OK" if path.exists() else "MISSING"
        failed |= state != "OK"
        print(f"{state:7} {label}: {path}")
    manifests = list(dataset_root.glob("*/*.jsonl")) if dataset_root.is_dir() else []
    if manifests:
        print(f"OK      evaluation manifests: {len(manifests)} JSONL files")
    else:
        failed = True
        print(f"MISSING evaluation manifests: no JSONL files below {dataset_root}")
    if failed:
        raise SystemExit(1)
    print("Environment and artifact layout are ready.")


if __name__ == "__main__":
    main()
