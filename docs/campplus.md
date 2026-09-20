# CAM++ implementation and checkpoint loading

The speaker-embedding arm uses CAM++ from the official
[3D-Speaker toolkit](https://github.com/modelscope/3D-Speaker). The upstream
architecture is implemented in
[`speakerlab/models/campplus/DTDNN.py`](https://github.com/modelscope/3D-Speaker/blob/main/speakerlab/models/campplus/DTDNN.py),
and the Chinese 16 kHz checkpoint is published on the official ModelScope page
[`iic/speech_campplus_sv_zh-cn_16k-common`](https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/summary).
Both are released under Apache License 2.0.

This repository carries the compatible implementation required by the TS-ASR
model in `funasr/models/campplus/` and the strict checkpoint adapter in
`funasr/models/tsasr/spk_encoder.py`. The adapter expects the official
`campplus_cn_common.bin` state dictionary and verifies every CAM++ parameter:

```python
from funasr.models.tsasr.spk_encoder import CAMPPlusSpeakerEncoder

encoder = CAMPPlusSpeakerEncoder(
    init_param="/path/to/campplus_cn_common.bin",
    embedding_size=192,
    feat_dim=80,
    freeze=True,
).eval()
```

Input features follow the official recipe: mono 16 kHz audio, 80-bin Kaldi
filterbanks, and utterance-level mean normalization. The encoder returns one
192-dimensional vector per utterance.

## Loading from this release

The released
`speaker-cue-ts-asr-embedding/model.pt` already contains the frozen CAM++
parameters under the ASR model's `spk_encoder` namespace. No separate download
is needed for released-checkpoint inference:

```bash
python scripts/infer.py \
  --model-dir artifacts/models/ckpt/speaker-cue-ts-asr-embedding \
  --audio /path/to/mixture.wav \
  --enrollment-audio /path/to/enrollment.wav
```

`ModelBundle` extracts the embedded CAM++ state into a temporary file, constructs
the official-compatible encoder, and then loads the complete TS-ASR state.

The same mechanism is available for training and offline embedding extraction:

```bash
# Warm-start training; CAM++ is recovered automatically from model.pt.
python scripts/train.py \
  --config artifacts/models/ckpt/speaker-cue-ts-asr-embedding/config.yaml \
  --init-param artifacts/models/ckpt/speaker-cue-ts-asr-embedding/model.pt \
  --embedding-store /path/to/campplus_utterance_emb \
  --train-manifest /path/to/train.jsonl \
  --valid-manifest /path/to/dev.jsonl \
  --output-dir exp/speaker-cue-ts-asr-embedding

# Build the offline embedding store directly from the released model.
python scripts/extract_embeddings.py \
  --jsonl /path/to/train.jsonl /path/to/dev.jsonl \
  --out-dir /path/to/campplus_utterance_emb \
  --released-model artifacts/models/ckpt/speaker-cue-ts-asr-embedding/model.pt
```

For a from-scratch run, download `campplus_cn_common.bin` from the official
ModelScope page and pass it through `--speaker-encoder-init` during training or
`--campplus-ckpt` during embedding extraction.
