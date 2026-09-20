"""Standalone online-enrollment dataset for speaker-embedding TS-ASR.

No keyword/text cue is created here.  A target waveform is mixed only with a
different speaker; a clean, different utterance of the target speaker becomes
the online CAM++ registration input.
"""

import json
import os
import random
import unicodedata

import numpy as np
import torch
import torchaudio.compliance.kaldi as Kaldi

from funasr.datasets.audio_datasets.bos_robust_read import BosRobustReadMixin
from funasr.datasets.audio_datasets.datasets import AudioDataset
from funasr.register import tables
from funasr.utils.load_utils import extract_fbank


def _sentence_id(text):
    """Canonical transcript id used to prevent same-sentence enrollment.

    AISHELL-2 may contain the same prompted sentence for one speaker in its
    iOS, Android, and microphone recordings.  They are distinct files but not
    independent enrollment utterances.  Ignore harmless formatting variation
    so these copies are always treated as the same sentence.
    """
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return "".join(ch for ch in text if not ch.isspace() and not unicodedata.category(ch).startswith("P"))


@tables.register("index_ds_classes", "IndexDSJsonlSpeakerCTC")
class IndexDSJsonlSpeakerCTC(torch.utils.data.Dataset):
    """JSONL index retaining canonical speaker ids for online enrollment."""

    def __init__(self, path, **kwargs):
        self.contents = []
        speaker_field = str(kwargs.get("speaker_field", "speaker"))
        min_src = int(kwargs.get("min_source_length", 0))
        max_src = int(kwargs.get("max_source_length", 1000000))
        min_tgt = int(kwargs.get("min_target_length", 0))
        max_tgt = int(kwargs.get("max_target_length", 1000000))
        max_tok = int(kwargs.get("max_token_length", 1000000))
        files = self._resolve_file_list(path, **kwargs)
        for file_path in files:
            with open(file_path, encoding="utf-8") as source:
                for line_no, line in enumerate(source):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    corpus = str(row.get("corpus", "default"))
                    speaker = row.get(speaker_field, row.get("speaker_id"))
                    if speaker in (None, ""):
                        raise ValueError(
                            f"Missing {speaker_field!r} at {file_path}:{line_no + 1}"
                        )
                    # V4 TTS rows store only the command in ``target`` because
                    # the text-cue path may crop the wake word.  This dataset
                    # always feeds the complete waveform, so its CTC target
                    # must include that spoken prefix as well.
                    target = f"{row.get('kw_text', '')}{row['target']}"
                    src_len = int(row.get("source_len", 1)) + int(
                        round(float(row.get("cmd_start", 0.0)) * 100)
                    )
                    tgt_len = len(target)
                    if not (min_src <= src_len <= max_src and min_tgt <= tgt_len <= max_tgt):
                        continue
                    if src_len + tgt_len > max_tok:
                        continue
                    self.contents.append({
                        "key": row.get("key", f"{file_path}:{line_no}"),
                        "source": row["source"], "target": target,
                        "sentence_id": _sentence_id(target),
                        "source_len": src_len, "target_len": tgt_len,
                        "speaker": f"{corpus}:{speaker}",
                    })
        counts = {}
        speaker_texts = {}
        for row in self.contents:
            counts[row["speaker"]] = counts.get(row["speaker"], 0) + 1
            speaker_texts.setdefault(row["speaker"], set()).add(row["sentence_id"])
        # “different utterance” is insufficient for enrollment: augmented or
        # duplicated recordings of the same sentence would leak lexical content.
        # Keep only targets that have a same-speaker enrollment with different text.
        self.contents = [
            row for row in self.contents
            if counts[row["speaker"]] >= 2
            and len(speaker_texts[row["speaker"]] - {row["sentence_id"]}) > 0
        ]
        if not self.contents:
            raise ValueError("No usable speakers with at least two utterances")

    @staticmethod
    def _resolve_file_list(path, **kwargs):
        """Apply distributed file-list sharding like the V4 text-cue index."""
        if path.endswith((".jsonl", ".json")):
            return [path]
        num = int(kwargs.get("data_split_num", 1))
        idx = int(kwargs.get("data_split_i", 0))
        if not kwargs.get("is_training", True):
            num, idx = 1, 0
        with open(path, encoding="utf-8") as source:
            files = [line.strip() for line in source if line.strip()]
        per = (len(files) - 1) // num + 1
        return files[idx * per: (idx + 1) * per]

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, index):
        return self.contents[index]

    def get_source_len(self, item):
        return item["source_len"]

    def get_target_len(self, item):
        return item["target_len"]


@tables.register("dataset_classes", "AudioDatasetSpeakerOnlineEnroll")
class AudioDatasetSpeakerOnlineEnroll(BosRobustReadMixin, AudioDataset):
    """Speaker-only target ASR with online mixture and online CAM++ features.

    ``negative_prob`` creates target-absent examples: the enrollment remains
    from the requested speaker, while the acoustic target is replaced by a
    different speaker and all mix partners are also forbidden from being the
    enrolled speaker.  CTC cannot represent an empty target, so negatives are
    supervised with ``no_tgt_spk_id`` (``<no_tgt_spk/>`` = 8407) repeated to the
    length of the transcript that *would* have been produced.

    ``<no_tgt_spk/>`` is a dedicated rejection label, not ``<unk>`` (changed
    2026-08-22).  ``<unk>`` already means "this transcript contains a character
    outside the vocabulary"; overloading it with "the enrolled speaker is absent"
    made the two indistinguishable at eval time and forced one CTC output row to
    model both an acoustic event and the *absence* of one.  The keyword-side
    counterpart is ``<no_kws/>`` (8406), used by ``datasets_kws_span_tsasr.py``;
    keeping the two rejection reasons on separate ids is what lets miss rates be
    attributed to speaker confusion vs keyword confusion.
    """

    def __init__(self, path, index_ds=None, frontend=None, tokenizer=None, **kwargs):
        super().__init__(path, index_ds=index_ds or "IndexDSJsonlSpeakerCTC",
                         frontend=frontend, tokenizer=tokenizer, **kwargs)
        self._init_bos_robust_read(**kwargs)
        self.mix_conf = dict(kwargs.get("mix_conf") or {})
        self.enroll_min_seconds = float(kwargs.get("enroll_min_seconds", 0.4))
        self.enroll_max_seconds = float(kwargs.get("enroll_max_seconds", 5.0))
        self.negative_prob = float(kwargs.get("negative_prob", 0.0))
        self.use_filler_label_for_neg = bool(kwargs.get("use_filler_label_for_neg", True))
        self.no_tgt_spk_id = int(kwargs.get("no_tgt_spk_id", 8407))
        if not 0.0 <= self.negative_prob <= 1.0:
            raise ValueError("negative_prob must be in [0, 1]")
        if self.negative_prob > 0.0 and not self.use_filler_label_for_neg:
            raise ValueError(
                "target-absent speaker negatives require use_filler_label_for_neg=True; "
                "CTC has no empty-target representation in this recipe."
            )
        # Fail loudly on a stale token_list: with tokens_kws8406.json the rejection id
        # 8407 is out of range and would only surface as a CUDA assert inside CTCLoss
        # (or, worse, silently index a valid row if the vocab were larger).
        if self.negative_prob > 0.0 and self.tokenizer is not None:
            n_vocab = len(getattr(self.tokenizer, "token_list", []) or [])
            if n_vocab and not 0 <= self.no_tgt_spk_id < n_vocab:
                raise ValueError(
                    f"no_tgt_spk_id={self.no_tgt_spk_id} out of range for a "
                    f"{n_vocab}-token vocabulary; point tokenizer_conf.token_list at "
                    "conf/tokens_kws8408.json"
                )
            expect = "<no_tgt_spk/>"
            if n_vocab and self.tokenizer.token_list[self.no_tgt_spk_id] != expect:
                raise ValueError(
                    f"token_list[{self.no_tgt_spk_id}] is "
                    f"{self.tokenizer.token_list[self.no_tgt_spk_id]!r}, expected {expect!r}"
                )
        self.speaker_to_indices = {}
        for i, row in enumerate(self.index_ds):
            self.speaker_to_indices.setdefault(row["speaker"], []).append(i)
        if self.negative_prob > 0.0 and len(self.speaker_to_indices) < 2:
            raise ValueError("negative_prob > 0 requires at least two speakers")

    def _load_wave(self, row):
        return self._load_waveform_robust(row["source"])

    def _enroll_candidates(self, index):
        pool = self.speaker_to_indices[self.index_ds[index]["speaker"]]
        sentence_id = self.index_ds[index]["sentence_id"]
        candidates = [
            candidate for candidate in pool
            if candidate != index
            and self.index_ds[candidate]["sentence_id"] != sentence_id
        ]
        if not candidates:
            raise RuntimeError(
                f"No same-speaker different-sentence enrollment for index={index}"
            )
        return candidates

    def _pick_enroll(self, index):
        return random.choice(self._enroll_candidates(index))

    def _pick_enroll_with_min_length(self, index, min_samples):
        candidates = self._enroll_candidates(index)
        random.shuffle(candidates)
        for enroll_index in candidates:
            enroll = self._load_wave(self.index_ds[enroll_index])
            if enroll.numel() >= min_samples:
                return enroll_index, enroll
        raise RuntimeError(
            "No valid enrollment for "
            f"{self.index_ds[index]['source']}: all "
            f"{len(candidates)} same-speaker different-sentence candidates "
            f"are shorter than {min_samples} samples"
        )

    def _pick_negative_source(self, enrolled_speaker):
        """Choose a waveform whose speaker is not the enrollment speaker."""
        for _ in range(32):
            index = random.randrange(len(self.index_ds))
            if self.index_ds[index]["speaker"] != enrolled_speaker:
                return index
        candidates = [
            i for i, row in enumerate(self.index_ds)
            if row["speaker"] != enrolled_speaker
        ]
        if not candidates:
            raise RuntimeError("No different-speaker waveform available for negative sample")
        return random.choice(candidates)

    def _negative_label(self, token_count):
        return [self.no_tgt_spk_id] * min(int(token_count), 1)

    def __getitem__(self, index):
        row = self.index_ds[index]
        min_samples = int(self.enroll_min_seconds * self.fs)
        max_samples = int(self.enroll_max_seconds * self.fs)
        enroll_index, enroll = self._pick_enroll_with_min_length(index, min_samples)
        is_negative = random.random() < self.negative_prob
        speech_row = (
            self.index_ds[self._pick_negative_source(row["speaker"])]
            if is_negative else row
        )
        wav = self._load_wave(speech_row)
        if enroll.numel() > max_samples:
            start = random.randint(0, enroll.numel() - max_samples)
            enroll = enroll[start:start + max_samples]
        target = self.preprocessor_text(row["target"]) if self.preprocessor_text else row["target"]
        ids = self.tokenizer.encode(target)
        label = self._negative_label(len(ids)) if is_negative else ids
        return {
            "_waveform": wav,
            "_enroll": enroll,
            "_speaker": speech_row["speaker"],
            "_forbid_speaker": row["speaker"],
            "_source": speech_row["source"],
            "_target": "<no_target>" if is_negative else target,
            "_enroll_source": self.index_ds[enroll_index]["source"],
            "_enroll_target": self.index_ds[enroll_index]["target"],
            "text": torch.tensor(label, dtype=torch.int64),
            "text_lengths": torch.tensor([len(label)], dtype=torch.int32),
        }

    def _mix(self, waveforms, speakers, forbid_speakers=None):
        if not self.mix_conf or float(self.mix_conf.get("prob", 1.0)) <= 0:
            return waveforms, [[] for _ in waveforms]
        lo, hi = self.mix_conf.get("ratio_range", (0.1, 0.9))
        shift_prob = float(self.mix_conf.get("shift_prob", 0.3))
        shift_seconds = int(self.mix_conf.get("shift_seconds", 3))
        mixed, partners = [], []
        for i, target in enumerate(waveforms):
            forbidden = forbid_speakers[i] if forbid_speakers is not None else speakers[i]
            candidates = [
                j for j, spk in enumerate(speakers)
                if j != i and spk != speakers[i] and spk != forbidden
            ]
            if not candidates or random.random() >= float(self.mix_conf.get("prob", 1.0)):
                mixed.append(target)
                partners.append([])
                continue
            partner = random.choice(candidates)
            interferer = waveforms[partner]
            wavs = [target, interferer]
            if random.random() < shift_prob:
                k = random.randrange(2)
                delay = random.randint(self.fs, self.fs * shift_seconds)
                wavs[k] = torch.cat([wavs[k].new_zeros(delay), wavs[k]])
            rms = [max(w.norm(p=2).item(), 1e-2) for w in wavs]
            length = max(max(w.numel() for w in wavs), self.fs)
            wavs = [
                torch.cat([w, w.new_zeros(length - w.numel())])
                if w.numel() < length else w
                for w in wavs
            ]
            wavs = [w * (max(rms) / r) for w, r in zip(wavs, rms)]
            gains = [random.uniform(lo, hi), random.uniform(lo, hi)]
            mix = gains[0] * wavs[0] + gains[1] * wavs[1]
            if mix.abs().max() > 1:
                total = sum(gains)
                mix = (gains[0] / total) * wavs[0] + (gains[1] / total) * wavs[1]
            mixed.append(mix)
            partners.append([partner])
        return mixed, partners

    @staticmethod
    def _describe_mix(sources, targets, enroll_sources, enroll_targets, partners):
        def one(tag, index):
            return f"    {tag} {sources[index]}\n         {targets[index]}"
        infos = []
        for i, part in enumerate(partners):
            lines = ["整条混音（1 个干扰源）" if part else "未混音", one("tgt ", i)]
            lines += [one(f"itf{k}", j) for k, j in enumerate(part)]
            lines += [f"    enroll {enroll_sources[i]}\n         {enroll_targets[i]}"]
            infos.append("\n".join(lines))
        return infos

    @staticmethod
    def _campp_fbank(wav):
        feat = Kaldi.fbank(wav.unsqueeze(0), num_mel_bins=80)
        return feat - feat.mean(dim=0, keepdim=True)

    def collator(self, samples=None):
        waveforms = [sample.pop("_waveform") for sample in samples]
        enrolls = [sample.pop("_enroll") for sample in samples]
        speakers = [sample.pop("_speaker") for sample in samples]
        forbid_speakers = [sample.pop("_forbid_speaker") for sample in samples]
        sources = [sample.pop("_source") for sample in samples]
        targets = [sample.pop("_target") for sample in samples]
        enroll_sources = [sample.pop("_enroll_source") for sample in samples]
        enroll_targets = [sample.pop("_enroll_target") for sample in samples]
        waveforms, partners = self._mix(waveforms, speakers, forbid_speakers)
        speech, speech_lengths = extract_fbank(
            waveforms, data_type="sound", frontend=self.frontend, is_final=True
        )
        texts = [sample["text"] for sample in samples]
        lengths = [sample["text_lengths"] for sample in samples]
        enroll_feats = [self._campp_fbank(wav) for wav in enrolls]
        enroll_lens = torch.tensor([item.size(0) for item in enroll_feats], dtype=torch.int32)
        ref = torch.zeros(len(enroll_feats), int(enroll_lens.max()), 80, dtype=torch.float32)
        for i, feat in enumerate(enroll_feats):
            ref[i, :feat.size(0)] = feat
        return {
            "speech": speech,
            "speech_lengths": speech_lengths.view(-1, 1).to(torch.int32),
            "text": torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=-1),
            "text_lengths": torch.nn.utils.rnn.pad_sequence(lengths, batch_first=True, padding_value=-1),
            "ref_speech": ref,
            "ref_speech_lengths": enroll_lens.view(-1, 1),
            "mix_info": self._describe_mix(sources, targets, enroll_sources, enroll_targets, partners),
        }


@tables.register("dataset_classes", "AudioDatasetSpeakerOfflineEnroll")
class AudioDatasetSpeakerOfflineEnroll(AudioDatasetSpeakerOnlineEnroll):
    """Same random same-speaker enrollment sampling, backed by offline CAM++ vectors."""

    def __init__(self, *args, emb_store=None, **kwargs):
        super().__init__(*args, **kwargs)
        if not emb_store:
            raise ValueError("AudioDatasetSpeakerOfflineEnroll requires emb_store")
        meta_path = os.path.join(emb_store, "spk_emb.meta.json")
        keys_path = os.path.join(emb_store, "spk_emb.keys")
        emb_path = os.path.join(emb_store, "spk_emb.f32")
        with open(meta_path, encoding="utf-8") as source:
            meta = json.load(source)
        with open(keys_path, encoding="utf-8") as source:
            keys = [line.strip() for line in source if line.strip()]
        if len(keys) != int(meta["num"]):
            raise ValueError("spk_emb.keys and spk_emb.meta.json disagree")
        self.emb_dim = int(meta["dim"])
        self.emb_path = emb_path
        self.emb_shape = (len(keys), self.emb_dim)
        self.key_to_emb = {key: i for i, key in enumerate(keys)}
        missing = [row["key"] for row in self.index_ds if row["key"] not in self.key_to_emb]
        if missing:
            raise ValueError(f"Offline embedding store misses {len(missing)} dataset keys, e.g. {missing[0]}")
        self._emb = None

    def _embedding(self, key):
        if self._emb is None:
            self._emb = np.memmap(self.emb_path, dtype=np.float32, mode="r", shape=self.emb_shape)
        return torch.from_numpy(np.array(self._emb[self.key_to_emb[key]], copy=True))

    def __getitem__(self, index):
        row = self.index_ds[index]
        enroll_index = self._pick_enroll(index)
        is_negative = random.random() < self.negative_prob
        speech_row = (
            self.index_ds[self._pick_negative_source(row["speaker"])]
            if is_negative else row
        )
        target = self.preprocessor_text(row["target"]) if self.preprocessor_text else row["target"]
        ids = self.tokenizer.encode(target)
        label = self._negative_label(len(ids)) if is_negative else ids
        return {
            "_waveform": self._load_wave(speech_row),
            "_speaker": speech_row["speaker"],
            "_forbid_speaker": row["speaker"],
            "_source": speech_row["source"],
            "_target": "<no_target>" if is_negative else target,
            "_enroll_source": self.index_ds[enroll_index]["source"],
            "_enroll_target": self.index_ds[enroll_index]["target"],
            "spk_emb": self._embedding(self.index_ds[enroll_index]["key"]),
            "text": torch.tensor(label, dtype=torch.int64),
            "text_lengths": torch.tensor([len(label)], dtype=torch.int32),
        }

    def collator(self, samples=None):
        waveforms = [sample.pop("_waveform") for sample in samples]
        speakers = [sample.pop("_speaker") for sample in samples]
        forbid_speakers = [sample.pop("_forbid_speaker") for sample in samples]
        sources = [sample.pop("_source") for sample in samples]
        targets = [sample.pop("_target") for sample in samples]
        enroll_sources = [sample.pop("_enroll_source") for sample in samples]
        enroll_targets = [sample.pop("_enroll_target") for sample in samples]
        waveforms, partners = self._mix(waveforms, speakers, forbid_speakers)
        speech, speech_lengths = extract_fbank(
            waveforms, data_type="sound", frontend=self.frontend, is_final=True
        )
        return {
            "speech": speech,
            "speech_lengths": speech_lengths.view(-1, 1).to(torch.int32),
            "text": torch.nn.utils.rnn.pad_sequence([sample["text"] for sample in samples], batch_first=True, padding_value=-1),
            "text_lengths": torch.nn.utils.rnn.pad_sequence([sample["text_lengths"] for sample in samples], batch_first=True, padding_value=-1),
            "spk_emb": torch.stack([sample["spk_emb"] for sample in samples]),
            "mix_info": self._describe_mix(sources, targets, enroll_sources, enroll_targets, partners),
        }
