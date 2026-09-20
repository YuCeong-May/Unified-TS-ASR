import json
import zipfile
from pathlib import Path

import pytest

from scripts.download_assets import safe_extract
from scripts.evaluate import normalize_manifest
from scripts.train import select_deepspeed_config
from unified_ts_asr.runtime import resolve_config_paths


def test_released_config_paths_are_relative_to_config(tmp_path):
    model_dir = tmp_path / "ckpt" / "model"
    model_dir.mkdir(parents=True)
    config_path = model_dir / "config.yaml"
    config_path.write_text("model: test\n", encoding="utf-8")
    config = {
        "tokenizer_conf": {
            "token_list": "../config/tokens.json",
            "seg_dict": "../config/seg_dict",
        },
        "model_conf": {"spk_encoder_conf": {"init_param": "pretrained/camp.bin"}},
    }
    resolved = resolve_config_paths(config, str(config_path))
    assert resolved["tokenizer_conf"]["token_list"] == str(
        (tmp_path / "ckpt" / "config" / "tokens.json").resolve()
    )
    assert resolved["model_conf"]["spk_encoder_conf"]["init_param"] == str(
        (model_dir / "pretrained" / "camp.bin").resolve()
    )


def test_manifest_audio_paths_resolve_from_ts_aishell(tmp_path):
    dataset = tmp_path / "dataset"
    base = dataset / "ts-aishell"
    base.mkdir(parents=True)
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "key": "one",
                "source": "aishell1-ts/wav/one.wav",
                "enroll_source": "../aishell/enroll.wav",
                "target": "测试",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "resolved.jsonl"
    normalize_manifest(source, output, dataset)
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["source"] == str((base / "aishell1-ts/wav/one.wav").resolve())
    assert row["enroll_source"] == str((base / "../aishell/enroll.wav").resolve())


def test_zip_path_traversal_is_rejected(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../escape.txt", "unsafe")
    with pytest.raises(ValueError, match="Unsafe archive member"):
        safe_extract(archive, tmp_path / "out")


def test_deepspeed_recipe_follows_released_config_name(tmp_path):
    config = {
        "deepspeed_config": "deepspeed_conf/ds_stage1_autocast_accum2.json",
        # This field is used by non-DeepSpeed training code and can legitimately
        # differ from the DeepSpeed engine's own accumulation setting.
        "train_conf": {"accum_grad": 1},
    }
    selected = select_deepspeed_config(config, tmp_path)
    assert selected == (
        tmp_path / "configs" / "deepspeed" / "stage1-bf16-accum2.json"
    )


def test_deepspeed_recipe_falls_back_to_train_conf(tmp_path):
    selected = select_deepspeed_config(
        {"train_conf": {"accum_grad": 2}}, tmp_path
    )
    assert selected.name == "stage1-bf16-accum2.json"
