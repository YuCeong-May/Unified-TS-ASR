# Unified Target-Speaker ASR

This repository contains the training and inference implementation for Mandarin target-speaker automatic speech recognition (TS-ASR) with text cues, enrollment-speech cues, or both. It is organized so that a released checkpoint can be evaluated without reconstructing the original experiment filesystem.

## Anonymous review resources

- [Code](https://anonymous.4open.science/r/Unifed-TS-ASR-41B8)
- [Model checkpoints](https://anonymous-hf.com/a/x3aljjy7y3vi/)
- [Evaluation dataset](https://anonymous-hf.com/a/fftzfmfq4pq3/)

## What is included

```text
.
├── funasr/                   # FunASR 1.3.22 research fork and TS-ASR models
├── unified_ts_asr/           # portable checkpoint loader and inference API
├── scripts/
│   ├── download_assets.py    # anonymous model/dataset downloader
│   ├── infer.py              # one-mixture inference
│   ├── evaluate.py           # benchmark evaluation dispatcher
│   ├── train.py              # portable DeepSpeed training launcher
│   ├── extract_embeddings.py # offline CAM++ store builder
│   ├── make_speaker_map.py   # speaker-CE class mapping builder
│   └── check_setup.py
├── configs/deepspeed/
├── docs/training_data.md
└── tests/
```

The loader handles the released `model.pt` files directly. In particular, it unwraps their DeepSpeed `module` state, resolves tokenizer resources relative to each `config.yaml`, and reconstructs the CAM++ initialization embedded in the speaker-embedding checkpoint.

## CAM++ dependency

The speaker-embedding model uses the official
[3D-Speaker CAM++ implementation](https://github.com/modelscope/3D-Speaker/blob/main/speakerlab/models/campplus/DTDNN.py)
and the official Chinese 16 kHz
[`iic/speech_campplus_sv_zh-cn_16k-common` checkpoint](https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/summary).
The compatible implementation and strict weight adapter needed by this release
are included locally, so released-model inference does not depend on cloning
another repository. See [docs/campplus.md](docs/campplus.md) for standalone and
embedded-weight loading examples.

## Environment

Linux, an NVIDIA GPU, and Python 3.10 or 3.11 are recommended. Install a PyTorch and torchaudio build compatible with the local CUDA driver first, then install this repository:

```bash
python -m venv .venv
source .venv/bin/activate

# Select the CUDA wheel appropriate for the machine.
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -e .
```

For training:

```bash
pip install -e '.[train]'
```

## Download released artifacts

The following command uses only the anonymous review mirrors:

```bash
python scripts/download_assets.py --models --dataset --output-dir artifacts
python scripts/check_setup.py --artifacts artifacts
```

After extraction, the important paths are:

```text
artifacts/models/ckpt/<model-name>/{config.yaml,model.pt}
artifacts/models/ckpt/config/{tokens_kws8408.json,seg_dict_ext}
artifacts/dataset/ts-aishell/<subset>/...
```

## Single-mixture inference

Text-cue model:

```bash
python scripts/infer.py \
  --model-dir artifacts/models/ckpt/text-cue-ts-asr-neg-0p3 \
  --audio /path/to/mixture.wav \
  --text-cue "目标线索"
```

Speaker-cue model:

```bash
python scripts/infer.py \
  --model-dir artifacts/models/ckpt/speaker-cue-ts-asr-embedding-free \
  --audio /path/to/mixture.wav \
  --enrollment-audio /path/to/target-speaker-enrollment.wav
```

Unified model with both cues:

```bash
python scripts/infer.py \
  --model-dir artifacts/models/ckpt/unified-ts-asr-parallel \
  --audio /path/to/mixture.wav \
  --text-cue "目标线索" \
  --enrollment-audio /path/to/target-speaker-enrollment.wav
```

For a unified checkpoint, either cue may be omitted to run the corresponding ablation. The result is emitted as JSON.

## Evaluation

Text-cue evaluation using the fixed four-character cues:

```bash
python scripts/evaluate.py \
  --model-dir artifacts/models/ckpt/text-cue-ts-asr-neg-0p3 \
  --dataset-root artifacts/dataset \
  --manifest artifacts/dataset/ts-aishell/aishell1-ts/aishell1-ts-test-cue4.jsonl \
  --output-dir outputs/text-cue-cue4 \
  --gpus 0
```

Unified evaluation with both cues:

```bash
python scripts/evaluate.py \
  --model-dir artifacts/models/ckpt/unified-ts-asr-parallel \
  --dataset-root artifacts/dataset \
  --manifest artifacts/dataset/ts-aishell/aishell1-ts/aishell1-ts-test-cue4-spkemb.jsonl \
  --output-dir outputs/unified-both \
  --cue both \
  --gpus 0
```

Use `--cue text`, `--cue enroll`, or `--cue none` with the same unified checkpoint for cue ablations. Evaluation writes utterance-level JSONL, alignment details, and aggregate metrics to the output directory.

## Training

Use a released configuration as the canonical architecture and optimization configuration. The launcher replaces machine-specific paths with explicit command-line paths:

```bash
python scripts/train.py \
  --config artifacts/models/ckpt/unified-ts-asr-parallel/config.yaml \
  --train-manifest /path/to/train.jsonl \
  --valid-manifest /path/to/dev.jsonl \
  --output-dir exp/unified-ts-asr-parallel \
  --gpus 0,1,2,3
```

To warm-start from a released checkpoint, add:

```bash
  --init-param artifacts/models/ckpt/unified-ts-asr-parallel/model.pt
```

The launcher selects the bundled DeepSpeed recipe corresponding to the
`deepspeed_config` recorded in the released YAML. Use `--deepspeed-config` only
when intentionally changing that optimization setting.

For `speaker-cue-ts-asr-embedding`, the original CAM++ initialization is
embedded in the released `model.pt`. Supplying that file through `--init-param`
recovers the CAM++ state automatically before model construction. A from-scratch
run may instead pass `--speaker-encoder-init /path/to/campplus_cn_common.bin`.
This checkpoint's faithful training recipe also uses precomputed utterance
embeddings; provide them with `--embedding-store /path/to/campplus_utterance_emb`.
The required store files are described in [docs/training_data.md](docs/training_data.md).

The released benchmark is an evaluation set and must not be used as training data. The required training-manifest schema and cue-sampling behavior are documented in [docs/training_data.md](docs/training_data.md).

## Checkpoints

| Family | Released directories | Required inference inputs |
| --- | --- | --- |
| Text cue | `text-cue-ts-asr*` | mixture waveform and a 2-5 character cue |
| Speaker cue | `speaker-cue-ts-asr*` | mixture waveform and enrollment waveform |
| Unified | `unified-ts-asr-{parallel,concat}` | mixture plus text cue, enrollment waveform, or both |

Every checkpoint directory must remain paired with its own `config.yaml`. Shared tokenizer files stay in `ckpt/config/`.

## Reproducibility notes

- Audio is converted to mono 16 kHz before feature extraction.
- Recognition uses greedy CTC decoding, matching the supplied evaluation backends.
- Text-cue benchmark manifests provide fixed `cue_text` values; cues are not resampled during evaluation.
- Speaker enrollment is a different utterance from the target speaker.
- The unified evaluator supports identical text-only, enrollment-only, both-cue, and no-cue conditions from one checkpoint.

## License

Code in this release is provided under Apache License 2.0. The repository includes and modifies components from FunASR; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model and dataset use is additionally subject to the terms stated on their anonymous resource pages and the source-corpus licenses.
