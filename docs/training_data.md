# Training data format

Training and validation inputs are UTF-8 JSONL files with one utterance per line. Paths may be absolute or resolved by the storage backend configured in the YAML.

## Required fields

```json
{
  "key": "unique-utterance-id",
  "source": "/path/to/audio.wav",
  "target": "目标转写",
  "source_len": 420,
  "target_len": 8,
  "speaker": "speaker-id",
  "corpus": "corpus-name",
  "kw_text": "文本线索",
  "cmd_start": 0.0
}
```

| Field | Meaning |
| --- | --- |
| `key` | Stable utterance identifier. If omitted, an identifier is derived from the file and line number. |
| `source` | Waveform path. Audio is expected to be readable and is resampled to 16 kHz. |
| `target` | Target-speaker transcription. |
| `source_len` | Approximate number of 10 ms frames used by dynamic batching. |
| `target_len` | Number of target tokenizer units used by dynamic batching. |
| `speaker` | Corpus-local speaker identifier. Required for excluding same-speaker interference and selecting enrollment speech. |
| `corpus` | Corpus namespace. Speaker identity is interpreted as `corpus:speaker`. |
| `kw_text` | Optional known text prefix/cue. Empty for ordinary open-domain utterances. |
| `cmd_start` | Start time in seconds of the command segment when a preceding cue is present; otherwise `0.0`. |

## Model-family requirements

Text-cue training samples a contiguous cue from the target transcription according to the selected YAML. Speaker-cue and unified training require at least two different-sentence utterances for each usable speaker so that enrollment speech cannot leak the target sentence. Negative enrollment sampling additionally requires at least two speakers.

The released `speaker-cue-ts-asr-embedding` YAML uses
`AudioDatasetSpeakerOfflineEnroll`. Its `--embedding-store` directory must
contain `spk_emb.f32`, `spk_emb.keys`, and `spk_emb.meta.json`. The raw file is a
row-major float32 matrix; `spk_emb.keys` has one utterance key per row, and the
metadata records at least `num` and `dim`. Every utterance key eligible for
enrollment must be present. The embedding-free speaker models and unified models
instead derive enrollment features online and do not need this store.

Create the store from every training and validation manifest that can supply an
enrollment utterance. For one GPU, extraction and merge are:

```bash
python scripts/extract_embeddings.py \
  --jsonl /path/to/train.jsonl /path/to/dev.jsonl \
  --out-dir data/campplus_utterance_emb \
  --released-model artifacts/models/ckpt/speaker-cue-ts-asr-embedding/model.pt \
  --device cuda:0

python scripts/extract_embeddings.py \
  --merge \
  --out-dir data/campplus_utterance_emb
```

For multiple GPUs, run one extraction process per `--shard` using the same
`--num-shards`, assign a different CUDA device to each process, then run the merge
command once. Input `key` values must be globally unique.
If starting without the released model, replace `--released-model` with
`--campplus-ckpt /path/to/campplus_cn_common.bin`.

The three `text-cue-ts-asr-*-sv-head*` configurations also use a corpus-scoped
speaker-class mapping for their auxiliary CE head. `scripts/train.py` generates
it automatically from `--train-manifest` when needed. To build or inspect it
separately:

```bash
python scripts/make_speaker_map.py \
  --jsonl /path/to/train.jsonl \
  --out data/spk2id.json

```

The generated mapping size must equal `model_conf.spk_ce_conf.num_speakers` in
the selected released YAML. Validation speakers absent from the training mapping
are ignored by the auxiliary head, as intended. Pass `--speaker-map` only when
reusing an existing mapping whose exact class order must be preserved.

The online mixer chooses a different-speaker interferer and applies the SIR and overlap distributions from `dataset_conf.mix_conf`. Cue-presence and negative-cue probabilities are also controlled by `dataset_conf`; the released YAML is the authoritative record for each checkpoint.

## Reproduction boundary

The anonymous evaluation archive is held out and must not be merged into training or validation manifests. Full-scale training requires the appropriately licensed source corpora and any synthetic speech described by the experiment protocol. This repository provides the model, loader, online mixing logic, optimization configuration, and launcher, but does not redistribute third-party training corpora.
