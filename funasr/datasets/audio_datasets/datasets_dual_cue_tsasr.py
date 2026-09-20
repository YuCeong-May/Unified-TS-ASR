# -*- coding: utf-8 -*-
"""Dataset for `DualCueConformerCTC`: text cue and enrollment-speech cue in one batch.

`AudioDatasetKwsSpan` (V4) produces the text cue — a random contiguous span of the
transcript, wrapped in `psok/peok`, with the same bracket pair injected in place in
the CTC target.  `AudioDatasetSpeakerOnlineEnroll` produces the enrollment cue — a
different utterance of the same speaker.  Neither emits both, so neither can train
a model that has to work with either cue, both, or none.

This dataset extends the V4 text-cue dataset with:

- **an enrollment utterance** of the target speaker (same speaker, *different*
  sentence, so no lexical content leaks), featurised with the *same* frontend as
  the mixture because `ConformerDualCueEncoder` runs both through one shared
  subsampler;
- **an enrollment utterance** of the target speaker (same speaker, *different*
  sentence, so no lexical content leaks), featurised with the *same* frontend as
  the mixture because `ConformerDualCueEncoder` runs both through one shared
  subsampler;
- **enrollment negatives** (`enroll_negative_prob`): the enrollment comes from a
  different speaker, so the target speaker is absent from the mixture.  Without
  negatives the enrollment cue carries no information the audio does not already
  have, and experiment 4-2 showed the model then either ignores the cue or parrots
  it (`exp.md`);
- **per-sample cue dropout** (`text_cue_prob`, `enroll_cue_prob`), drawn
  independently.  A dropped cue is signalled by a length of 0, which the model
  turns into an exact no-op.  Both default to 1.0: the interesting axis is whether
  a cue is *true*, not whether it is supplied.

The two cues' truth values are drawn **independently** — `positive_prob` for the
text cue (in the parent) and `enroll_negative_prob` for the enrollment — and the
two rejection symbols are **prefix flags** rather than replacements for the
transcript, so all four combinations keep whatever supervision they license:

    text real  + speaker present -> transcript with `psok..peok` in place
    text real  + speaker absent  -> `<no_tgt_spk/>` + the same bracketed transcript
    text wrong + speaker present -> `<no_kws/>` + plain transcript
    text wrong + speaker absent  -> `<no_kws/> <no_tgt_spk/>`, no transcript

See `AudioDatasetDualCue._compose_label` for the exact rule and for how a dropped
cue generalises (a cue that was never supplied asserts nothing, so it flags
nothing).
"""
import logging
import random
import unicodedata

import torch

from funasr.datasets.audio_datasets.datasets_kws_span_tsasr import AudioDatasetKwsSpan
from funasr.register import tables
from funasr.utils.load_utils import extract_fbank


def _sentence_id(text):
    """Canonical transcript id, used to forbid same-sentence enrollment.

    AISHELL-2 records the same prompted sentence for one speaker on iOS, Android
    and microphone.  Those are distinct files but not independent enrollment
    utterances: using one to enrol the other leaks the transcript.  Ignore
    whitespace, punctuation and case so the copies collapse to one id.
    """
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return "".join(
        ch for ch in text
        if not ch.isspace() and not unicodedata.category(ch).startswith("P")
    )


@tables.register("dataset_classes", "AudioDatasetDualCue")
class AudioDatasetDualCue(AudioDatasetKwsSpan):
    """V4 text-cue dataset plus an enrollment cue and per-sample cue dropout.

    Args:
        path / index_ds / frontend / tokenizer / mix_conf / ... : see
            `AudioDatasetKwsSpan`.  `index_ds` defaults to `IndexDSJsonlKwsSpanV4`
            here (not the V1 phone index) because the enrollment pool needs the
            `speaker` field that index keeps.
        text_cue_prob: probability that a sample is *given* a text cue at all.
            Defaults to 1.0; whether that cue is true is `positive_prob` (parent).
        enroll_cue_prob: probability that a sample is *given* an enrollment cue at
            all.  Defaults to 1.0 and is drawn independently of `text_cue_prob`.
        enroll_negative_prob: among samples that *do* carry an enrollment, the
            probability that it comes from a different speaker, i.e. the target
            speaker is absent from the mixture.  `1 - enroll_negative_prob` is the
            "enrollment positive" rate.  Independent of the text cue's truth, so
            all four truth combinations occur; see `_compose_label`.
        enroll_min_seconds / enroll_max_seconds: enrollment duration window; longer
            enrollments are cropped at a random offset, shorter candidates are
            skipped.
        no_tgt_spk_id: `<no_tgt_spk/>` = 8407 in `conf/tokens_kws8408.json`.
            Deliberately not `<no_kws/>` (8406, the text-cue rejection) and not
            `<unk>`: keeping the two rejection reasons apart is what lets misses be
            attributed to speaker confusion vs keyword confusion.  The two can
            appear in the same target (`<no_kws/> <no_tgt_spk/>`) when both cues
            reject; that is the only case with no transcript.
        enroll_candidate_tries: how many same-speaker candidates to try before
            giving up on an enrollment for this sample (it then has no enrollment
            cue rather than raising, because open-domain corpora contain speakers
            with a single usable utterance).
    """

    def __init__(self, path, index_ds: str = None, text_cue_prob: float = 1.0,
                 enroll_cue_prob: float = 1.0, enroll_negative_prob: float = 0.1,
                 enroll_min_seconds: float = 0.4, enroll_max_seconds: float = 5.0,
                 no_tgt_spk_id: int = 8407, enroll_candidate_tries: int = 8, **kwargs):
        super().__init__(path, index_ds=index_ds or "IndexDSJsonlKwsSpanV4", **kwargs)
        for name, value in (
            ("text_cue_prob", text_cue_prob),
            ("enroll_cue_prob", enroll_cue_prob),
            ("enroll_negative_prob", enroll_negative_prob),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        self.text_cue_prob = float(text_cue_prob)
        self.enroll_cue_prob = float(enroll_cue_prob)
        self.enroll_negative_prob = float(enroll_negative_prob)
        self.enroll_min_seconds = float(enroll_min_seconds)
        self.enroll_max_seconds = float(enroll_max_seconds)
        self.no_tgt_spk_id = int(no_tgt_spk_id)
        self.enroll_candidate_tries = int(enroll_candidate_tries)
        # `_compose_label` always rebuilds the CTC target, so the parent's
        # "replace the transcript with one filler token" path would be dead code
        # whose only effect is to confuse whoever reads the config.  Turn it off
        # explicitly instead of leaving it half-applied.
        if getattr(self, "use_filler_label_for_neg", False):
            logging.info(
                "[AudioDatasetDualCue] ignoring use_filler_label_for_neg=True: "
                "rejection symbols are prefix flags here, see _compose_label"
            )
            self.use_filler_label_for_neg = False
        # Set per batch in `collator`, read by `_pick_partner` during mixing.
        self._forbid_enroll_speakers = None

        # Fail now rather than inside CTCLoss as a CUDA assert hundreds of steps
        # later: 8407 is out of range for the older tokens_kws8406.json vocabulary.
        if self.enroll_cue_prob > 0 and self.enroll_negative_prob > 0:
            token_list = getattr(self.tokenizer, "token_list", None) or []
            if token_list and not 0 <= self.no_tgt_spk_id < len(token_list):
                raise ValueError(
                    f"no_tgt_spk_id={self.no_tgt_spk_id} is out of range for a "
                    f"{len(token_list)}-token vocabulary; point "
                    "tokenizer_conf.token_list at conf/tokens_kws8408.json"
                )
            if token_list and token_list[self.no_tgt_spk_id] != "<no_tgt_spk/>":
                raise ValueError(
                    f"token_list[{self.no_tgt_spk_id}] is "
                    f"{token_list[self.no_tgt_spk_id]!r}, expected '<no_tgt_spk/>'"
                )

        self._sentence_ids = []
        self._speaker_pool = {}
        for i in range(len(self.index_ds)):
            row = self.index_ds[i]
            if "speaker" not in row:
                raise ValueError(
                    "AudioDatasetDualCue needs a 'speaker' field on every index row "
                    "to build the enrollment pool; use IndexDSJsonlKwsSpanV4."
                )
            self._sentence_ids.append(_sentence_id(row["target"]))
            self._speaker_pool.setdefault(row["speaker"], []).append(i)
        n_usable = sum(1 for pool in self._speaker_pool.values() if len(pool) >= 2)
        logging.info(
            "[AudioDatasetDualCue] %d speakers, %d of them with >=2 utterances; "
            "text_cue_prob=%.2f enroll_cue_prob=%.2f enroll_negative_prob=%.2f "
            "(enrolled speaker present %.0f%% of the time)",
            len(self._speaker_pool), n_usable, self.text_cue_prob,
            self.enroll_cue_prob, self.enroll_negative_prob,
            100.0 * (1.0 - self.enroll_negative_prob),
        )
        if self.enroll_cue_prob > 0 and n_usable == 0:
            raise ValueError(
                "enroll_cue_prob > 0 but no speaker has two utterances; the jsonl "
                "speaker field is probably a per-utterance id"
            )
        if self.enroll_negative_prob > 0 and len(self._speaker_pool) < 2:
            raise ValueError("enroll_negative_prob > 0 requires at least two speakers")


    # ------------------------------------------------------------- enrollment
    def _positive_candidates(self, index):
        """Same speaker, different sentence, different row."""
        row = self.index_ds[index]
        sentence_id = self._sentence_ids[index]
        return [
            candidate for candidate in self._speaker_pool.get(row["speaker"], ())
            if candidate != index and self._sentence_ids[candidate] != sentence_id
        ]

    def _negative_candidates(self, index):
        """Any row whose speaker differs from the target's speaker."""
        forbid = self.index_ds[index]["speaker"]
        for _ in range(32):
            other = random.randrange(len(self.index_ds))
            if self.index_ds[other]["speaker"] != forbid:
                return [other]
        return [
            i for i in range(len(self.index_ds))
            if self.index_ds[i]["speaker"] != forbid
        ]

    def _load_enrollment(self, index, negative):
        """Pick and load one enrollment waveform, cropped to the length window.

        Returns:
            `(waveform, source, speaker)` or `(None, None, None)` when this sample
            cannot be enrolled (no same-speaker partner, or every candidate is too
            short).
        """
        candidates = (
            self._negative_candidates(index) if negative
            else self._positive_candidates(index)
        )
        if not candidates:
            return None, None, None
        random.shuffle(candidates)
        min_samples = int(self.enroll_min_seconds * self.fs)
        max_samples = int(self.enroll_max_seconds * self.fs)
        for candidate in candidates[: self.enroll_candidate_tries]:
            row = self.index_ds[candidate]
            # Always the untrimmed waveform: the enrollment is only used as a
            # voiceprint, so the wake word being present in it is harmless, while
            # trimming would throw away the part with the cleanest speech.
            waveform = self._load_waveform_robust(row["source"])
            if waveform.numel() < min_samples:
                continue
            if waveform.numel() > max_samples:
                start = random.randint(0, waveform.numel() - max_samples)
                waveform = waveform[start: start + max_samples]
            return waveform, row["source"], row["speaker"]
        return None, None, None

    def _enroll_batch(self, enrolls):
        """Featurise the enrollment column with the mixture's own frontend.

        Absent enrollments keep a placeholder row and get length 0, which the
        model reads as "this sample has no enrollment cue".  When the whole batch
        is enrollment-free the tensor is dropped entirely so the model runs its
        genuine no-enrollment path.

        Args:
            enrolls: list of length B holding waveforms or None.

        Returns:
            `(ref_speech (B, T_ref, 80) or None, ref_speech_lengths (B, 1) or None)`.
        """
        present = [wav is not None for wav in enrolls]
        if not any(present):
            return None, None
        # A one-frame placeholder is enough: its length is overwritten with 0.
        filler = torch.zeros(int(0.1 * self.fs), dtype=torch.float32)
        waveforms = [wav if wav is not None else filler for wav in enrolls]
        ref, ref_lengths = extract_fbank(
            waveforms, data_type="sound", frontend=self.frontend, is_final=True
        )
        ref_lengths = ref_lengths.reshape(-1).to(torch.int32)
        ref_lengths[~torch.tensor(present)] = 0
        return ref, ref_lengths.view(-1, 1)


    # ---------------------------------------------------------------- getitem
    def _compose_label(self, sample, want_text, text_positive, want_enroll, enroll_negative):
        """Build the CTC target for one sample.

        The two rejection symbols are **prefix flags**, not replacements for the
        transcript: each one asserts that the corresponding cue found nothing, and
        the body still carries whatever the *other* cue licenses.  With both cues
        always supplied (`text_cue_prob = enroll_cue_prob = 1.0`) this is exactly
        the four-way table:

            text real  + speaker present -> transcript with `psok..peok` in place
            text real  + speaker absent  -> `<no_tgt_spk/>` + the same bracketed transcript
            text wrong + speaker present -> `<no_kws/>` + plain transcript
            text wrong + speaker absent  -> `<no_kws/> <no_tgt_spk/>`, no transcript

        So 97% of samples keep full transcription supervision at 0.7/0.9, against
        68% under the older "one filler token replaces the transcript" scheme.
        Only the both-cues-reject case has nothing to transcribe: no cue points at
        a speaker, so demanding a transcript would teach the model to guess.

        The generalisation to a dropped cue (`*_cue_prob < 1`) follows the same
        rule: a cue that was never supplied makes no assertion, so it contributes
        no flag, and a sample with no cue at all is a plain-ASR sample.

        Flag order is `<no_kws/>` then `<no_tgt_spk/>`, fixed so the CTC target is
        deterministic given the draws.

        Args:
            sample: the parent's dict; `text` is its already-bracketed label and
                `_ids` the clean transcript.
            want_text / want_enroll: whether that cue is supplied at all.
            text_positive: the cue really occurs in this audio (parent's draw).
            enroll_negative: the enrolled speaker is absent from the mixture.

        Returns:
            list[int]: the CTC target ids.
        """
        flags = []
        if want_text and not text_positive:
            flags.append(self.no_kws_id)
        if want_enroll and enroll_negative:
            flags.append(self.no_tgt_spk_id)

        if want_text and text_positive:
            # The parent already inserted `psok .. peok` at the keyword's position.
            body = [int(x) for x in sample["text"].tolist()]
        elif (want_enroll and not enroll_negative) or not (want_text or want_enroll):
            # Either the enrollment identifies the speaker, or there is no cue at
            # all (plain ASR).  No keyword position to mark, so no brackets.
            body = list(sample["_ids"])
        else:
            body = []
        return flags + body

    def __getitem__(self, index):
        """The V4 text-cue sample, with the two cues drawn independently.

        The text cue's truth (`positive_prob`, drawn by the parent) and the
        enrollment's truth (`enroll_negative_prob`) are **independent**: all four
        combinations occur, and each is labelled by `_compose_label`.

        Returns:
            The parent's dict, plus `_enroll` (waveform or None), `_enroll_source`
            and `_enroll_speaker` (popped in `collator`).  A switched-off cue is
            expressed as a length of 0, never as a missing key, so the batch layout
            is identical for every condition.
        """
        sample = super().__getitem__(index)
        # The parent drew this from `positive_prob` and built its label with it.
        text_positive = int(sample["kws_target"]) == 1

        want_text = random.random() < self.text_cue_prob
        want_enroll = random.random() < self.enroll_cue_prob
        enroll_negative = want_enroll and random.random() < self.enroll_negative_prob
        enroll, enroll_source, enroll_speaker = (None, None, None)
        if want_enroll:
            enroll, enroll_source, enroll_speaker = self._load_enrollment(index, enroll_negative)
        if enroll is None:
            # Nothing was enrolled, so nothing licenses the "target speaker is
            # absent" label; fall back to an ordinary no-enrollment sample.
            want_enroll = False
            enroll_negative = False

        label = self._compose_label(
            sample, want_text, text_positive, want_enroll, enroll_negative
        )
        sample["text"] = torch.tensor(label, dtype=torch.int64)
        sample["text_lengths"] = torch.tensor([len(label)], dtype=torch.int32)
        # `_target` stays the **transcript of this audio**.  It is not a description
        # of the label: `_describe_mix` indexes it by batch position to caption the
        # interferer lines too, so overwriting it here would make one sample's label
        # show up as another sample's provenance.  The `ref` line of the sampled
        # train-time log already renders the exact CTC target from its ids.

        if not want_text:
            # One padding id keeps the cue tensor rectangular; length 0 is what the
            # model reads.  `_cue_raw = []` also stops the parent's negative-cue
            # re-check from firing (`_contains` is False for an empty needle).
            sample["cue_token"] = torch.tensor([0], dtype=torch.int64)
            sample["cue_token_lengths"] = torch.tensor([0], dtype=torch.int32)
            sample["kws_target"] = torch.tensor([-1], dtype=torch.int64)
            sample["kw_pos"] = torch.tensor([-1], dtype=torch.int64)
            sample["_cue_raw"] = []

        sample["_enroll"] = enroll
        sample["_enroll_source"] = enroll_source or ""
        sample["_enroll_speaker"] = enroll_speaker
        sample["_enroll_negative"] = enroll_negative
        return sample


    # ---------------------------------------------------------------- collate
    def _pick_partner(self, i, n, speakers=None, exclude=()):
        """Also forbid the enrolled speaker when this sample is an enrollment negative.

        The parent only prevents mixing a sample with its own speaker.  For a
        negative enrollment the label asserts that the *enrolled* speaker is absent
        from the mixture, so that speaker must be excluded from partner selection
        as well; otherwise the interferer could be exactly the enrolled voice and
        the `<no_tgt_spk/>` supervision would be inverted.
        """
        forbid = None
        if self._forbid_enroll_speakers is not None:
            forbid = self._forbid_enroll_speakers[i]
        if forbid is None or speakers is None:
            return super()._pick_partner(i, n, speakers=speakers, exclude=exclude)
        pool = [
            k for k in range(n)
            if k != i and k not in exclude
            and speakers[k] != speakers[i] and speakers[k] != forbid
        ]
        return random.choice(pool) if pool else None

    @staticmethod
    def _append_enroll_info(mix_info, sources, negatives):
        """Add the enrollment provenance to the parent's `mix_info` debug string."""
        if not mix_info:
            return mix_info
        lines = []
        for i, info in enumerate(mix_info):
            if not sources[i]:
                lines.append(f"{info}\n    enroll -")
            else:
                tag = "enroll(neg)" if negatives[i] else "enroll"
                lines.append(f"{info}\n    {tag} {sources[i]}")
        return lines

    def collator(self, samples: list = None):
        """Parent collation plus the enrollment column.

        Returns:
            The parent's dict.  `ref_speech` (B, T_ref, 80) and `ref_speech_lengths`
            (B, 1) are added only when at least one sample in the batch carries an
            enrollment, so an enrollment-free batch drives the model's genuine
            no-enrollment path instead of an all-zero tensor.
        """
        enrolls = [sample.pop("_enroll") for sample in samples]
        enroll_sources = [sample.pop("_enroll_source") for sample in samples]
        enroll_speakers = [sample.pop("_enroll_speaker") for sample in samples]
        negatives = [sample.pop("_enroll_negative") for sample in samples]

        # Consumed by `_pick_partner` inside the parent's mixing step.
        self._forbid_enroll_speakers = [
            speaker if negative else None
            for speaker, negative in zip(enroll_speakers, negatives)
        ]
        try:
            outputs = super().collator(samples)
        finally:
            self._forbid_enroll_speakers = None

        ref, ref_lengths = self._enroll_batch(enrolls)
        if ref is not None:
            outputs["ref_speech"] = ref
            outputs["ref_speech_lengths"] = ref_lengths
        outputs["mix_info"] = self._append_enroll_info(
            outputs.get("mix_info"), enroll_sources, negatives
        )
        return outputs

