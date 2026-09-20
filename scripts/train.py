#!/usr/bin/env python3
"""Launch reproducible multi-GPU training from a released model config."""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def select_deepspeed_config(config_values: dict, repository_root: Path) -> Path:
    """Map the released config's DeepSpeed recipe to the bundled portable copy."""
    original = str(config_values.get("deepspeed_config") or "")
    if "accum2" in Path(original).name:
        accumulation = 2
    elif "accum1" in Path(original).name:
        accumulation = 1
    else:
        accumulation = int(
            (config_values.get("train_conf") or {}).get("accum_grad", 1)
        )
    if accumulation not in (1, 2):
        raise ValueError(
            "No bundled DeepSpeed recipe for gradient accumulation "
            f"{accumulation}; pass --deepspeed-config explicitly"
        )
    return (
        repository_root
        / "configs"
        / "deepspeed"
        / f"stage1-bf16-accum{accumulation}.json"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="released config.yaml")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--valid-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpus", default="0", help="comma-separated CUDA device ids")
    parser.add_argument(
        "--deepspeed-config",
        default=None,
        help="defaults to the bundled copy of the recipe named by config.yaml",
    )
    parser.add_argument("--master-port", type=int, default=29501)
    parser.add_argument("--init-param", default=None, help="optional warm-start model.pt")
    parser.add_argument(
        "--speaker-encoder-init",
        default=None,
        help="standalone CAM++ state for training speaker-cue-ts-asr-embedding from scratch",
    )
    parser.add_argument(
        "--embedding-store",
        default=None,
        help="offline CAM++ utterance store required by AudioDatasetSpeakerOfflineEnroll",
    )
    parser.add_argument(
        "--speaker-map",
        default=None,
        help="corpus-scoped speaker-to-class JSON required by speaker-CE configs",
    )
    parser.add_argument("overrides", nargs="*", help="additional Hydra ++key=value overrides")
    args = parser.parse_args()

    config = Path(args.config).resolve()
    train_manifest = Path(args.train_manifest).resolve()
    valid_manifest = Path(args.valid_manifest).resolve()
    import yaml

    if not config.is_file():
        parser.error(f"file not found: {config}")
    with config.open(encoding="utf-8") as source:
        loaded = yaml.safe_load(source)
    if args.deepspeed_config:
        deepspeed_config = Path(args.deepspeed_config).resolve()
    else:
        try:
            deepspeed_config = select_deepspeed_config(
                loaded, Path(__file__).resolve().parents[1]
            )
        except ValueError as exc:
            parser.error(str(exc))
    required_files = [config, train_manifest, valid_manifest, deepspeed_config]
    if args.init_param:
        required_files.append(Path(args.init_param).resolve())
    if args.speaker_encoder_init:
        required_files.append(Path(args.speaker_encoder_init).resolve())
    for path in required_files:
        if not path.is_file():
            parser.error(f"file not found: {path}")
    embedding_store = None
    dataset_name = str(loaded.get("dataset"))
    if dataset_name == "AudioDatasetSpeakerOfflineEnroll":
        configured_store = str(
            (loaded.get("dataset_conf") or {}).get("emb_store", "")
        )
        candidates = []
        if configured_store:
            configured_path = Path(configured_store).expanduser()
            if configured_path.is_absolute():
                candidates.append(configured_path)
            else:
                candidates.extend(
                    [
                        (Path.cwd() / configured_path).resolve(),
                        (config.parent / configured_path).resolve(),
                    ]
                )
        if args.embedding_store:
            embedding_store = Path(args.embedding_store).resolve()
        else:
            embedding_store = next((path for path in candidates if path.is_dir()), None)
        if embedding_store is None or not embedding_store.is_dir():
            parser.error(
                "AudioDatasetSpeakerOfflineEnroll requires --embedding-store "
                "pointing to the offline CAM++ utterance store"
            )
        for filename in ("spk_emb.meta.json", "spk_emb.keys", "spk_emb.f32"):
            if not (embedding_store / filename).is_file():
                parser.error(f"embedding-store file not found: {embedding_store / filename}")
    elif args.embedding_store:
        parser.error(
            "--embedding-store is only valid for AudioDatasetSpeakerOfflineEnroll"
        )
    speaker_map = None
    configured_map = str((loaded.get("dataset_conf") or {}).get("spk2id", ""))
    if configured_map:
        configured_path = Path(configured_map).expanduser()
        candidates = (
            [configured_path]
            if configured_path.is_absolute()
            else [
                (Path.cwd() / configured_path).resolve(),
                (config.parent / configured_path).resolve(),
            ]
        )
        if args.speaker_map:
            speaker_map = Path(args.speaker_map).resolve()
        else:
            speaker_map = next((path for path in candidates if path.is_file()), None)
        expected_speakers = int(
            (((loaded.get("model_conf") or {}).get("spk_ce_conf") or {}).get(
                "num_speakers", 0
            ))
        )
        if speaker_map is None:
            from scripts.make_speaker_map import collect

            speaker_field = str(
                (loaded.get("dataset_conf") or {}).get("speaker_field", "speaker")
            )
            counts, _ = collect([str(train_manifest)], speaker_field)
            mapping = {
                name: index for index, name in enumerate(sorted(counts.keys()))
            }
            bootstrap_dir = Path(args.output_dir).resolve() / "bootstrap"
            bootstrap_dir.mkdir(parents=True, exist_ok=True)
            speaker_map = bootstrap_dir / "spk2id.json"
            with speaker_map.open("w", encoding="utf-8") as sink:
                json.dump(mapping, sink, ensure_ascii=False, indent=1, sort_keys=True)
            print(
                f"Generated corpus-scoped speaker map: {speaker_map} "
                f"({len(mapping)} speakers)",
                flush=True,
            )
        else:
            with speaker_map.open(encoding="utf-8") as source:
                mapping = json.load(source)
        if len(mapping) != expected_speakers:
            parser.error(
                f"speaker map has {len(mapping)} entries but config expects "
                f"{expected_speakers}; pass the original mapping with --speaker-map "
                "or use training manifests matching the released corpus"
            )
    elif args.speaker_map:
        parser.error("--speaker-map is only valid when config.yaml sets dataset_conf.spk2id")
    launcher = shutil.which("deepspeed")
    if not launcher:
        parser.error("deepspeed is not installed; run `pip install -e '.[train]'`")
    spec = importlib.util.find_spec("funasr.bin.train_ds")
    if spec is None or not spec.origin:
        parser.error("the bundled funasr package is not importable")

    model_root = config.parent
    shared = model_root.parent / "config"
    token_list = shared / "tokens_kws8408.json"
    seg_dict = shared / "seg_dict_ext"
    if not token_list.is_file() or not seg_dict.is_file():
        parser.error(
            "the config must stay inside the downloaded ckpt/<model>/ directory "
            "with ckpt/config available"
        )
    speaker_encoder_init = None
    if str(loaded.get("model")) == "SpeakerConformerCTCOnlineEnroll":
        configured = str(
            ((loaded.get("model_conf") or {}).get("spk_encoder_conf") or {}).get(
                "init_param", ""
            )
        )
        configured_path = Path(configured).expanduser()
        if configured and not configured_path.is_absolute():
            configured_path = (config.parent / configured_path).resolve()
        if args.speaker_encoder_init:
            speaker_encoder_init = Path(args.speaker_encoder_init).resolve()
        elif configured and configured_path.is_file():
            speaker_encoder_init = configured_path
        elif args.init_param:
            from unified_ts_asr.runtime import (
                _write_embedded_campplus,
                load_checkpoint_state,
            )

            bootstrap_dir = Path(args.output_dir).resolve() / "bootstrap"
            bootstrap_dir.mkdir(parents=True, exist_ok=True)
            speaker_encoder_init = Path(
                _write_embedded_campplus(
                    load_checkpoint_state(str(Path(args.init_param).resolve())),
                    str(bootstrap_dir),
                )
            )
            print(
                "Recovered CAM++ initialization from the released checkpoint: "
                f"{speaker_encoder_init}",
                flush=True,
            )
        else:
            parser.error(
                "SpeakerConformerCTCOnlineEnroll needs CAM++ initialization: pass "
                "--init-param with the released model.pt or --speaker-encoder-init "
                "with a standalone CAM++ state"
            )
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpu_ids:
        parser.error("--gpus must contain at least one CUDA device id")
    command = [
        launcher,
        "--num_gpus", str(len(gpu_ids)),
        "--master_port", str(args.master_port),
        spec.origin,
        "--config-path", str(config.parent),
        "--config-name", config.stem,
        "++use_deepspeed=true",
        f"++deepspeed_config={deepspeed_config}",
        f"++train_data_set_list={train_manifest}",
        f"++valid_data_set_list={valid_manifest}",
        f"++tokenizer_conf.token_list={token_list}",
        f"++tokenizer_conf.seg_dict={seg_dict}",
        f"++output_dir={Path(args.output_dir).resolve()}",
    ]
    if speaker_encoder_init is not None:
        command.append(
            f"++model_conf.spk_encoder_conf.init_param={speaker_encoder_init}"
        )
    if embedding_store is not None:
        command.append(f"++dataset_conf.emb_store={embedding_store}")
    if speaker_map is not None:
        command.append(f"++dataset_conf.spk2id={speaker_map}")
    if args.init_param:
        command += [f"++init_param={Path(args.init_param).resolve()}", "++ignore_init_mismatch=true"]
    command.extend(args.overrides)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    root = str(Path(__file__).resolve().parents[1])
    environment["PYTHONPATH"] = root + os.pathsep + environment.get("PYTHONPATH", "")
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=environment)


if __name__ == "__main__":
    main()
