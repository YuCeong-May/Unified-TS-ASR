#!/usr/bin/env python3
"""Download anonymous-review model and evaluation archives."""

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


ARCHIVES = {
    "models": "https://anonymous-hf.com/api/a/x3aljjy7y3vi/download/",
    "dataset": "https://anonymous-hf.com/api/a/fftzfmfq4pq3/download/",
}


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Unified-TS-ASR/1.0"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as sink:
        total = int(response.headers.get("Content-Length", 0))
        copied = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            sink.write(chunk)
            copied += len(chunk)
            if total:
                print(f"\r{destination.name}: {copied / total:6.1%}", end="", flush=True)
        if total:
            print()


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = (destination / member.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        source.extractall(destination)


def flatten_single_root(destination: Path, marker: str) -> None:
    if (destination / marker).exists():
        return
    children = [path for path in destination.iterdir() if path.name != ".DS_Store"]
    candidates = [path for path in children if path.is_dir() and (path / marker).exists()]
    if len(candidates) != 1:
        return
    nested = candidates[0]
    for child in nested.iterdir():
        shutil.move(str(child), str(destination / child.name))
    nested.rmdir()


def fetch(name: str, root: Path, force: bool) -> Path:
    destination = root / name
    marker = "ckpt" if name == "models" else "ts-aishell"
    if (destination / marker).exists() and not force:
        print(f"{name}: already available at {destination}")
        return destination
    if destination.exists() and force:
        shutil.rmtree(destination)
    archive = root / f".{name}.zip.part"
    print(f"Downloading {name} from the anonymous review mirror ...")
    download(ARCHIVES[name], archive)
    print(f"Extracting {archive.name} ...")
    safe_extract(archive, destination)
    archive.unlink()
    flatten_single_root(destination, marker)
    if not (destination / marker).exists():
        raise RuntimeError(f"Downloaded {name}, but expected {marker}/ was not found")
    print(f"{name}: ready at {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", action="store_true", help="download model checkpoints")
    parser.add_argument("--dataset", action="store_true", help="download evaluation data")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.models and not args.dataset:
        parser.error("select --models, --dataset, or both")
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name, enabled in (("models", args.models), ("dataset", args.dataset)):
        if enabled:
            fetch(name, root, args.force)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

