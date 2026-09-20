"""Standalone speaker-embedding conditioned Conformer-CTC.

This module intentionally has no keyword/text-cue encoder or cross-attention.
It is the speaker-embedding-only arm of the TS-ASR experiment.
"""

import logging

import torch
import torch.nn as nn

from funasr.models.tsasr.dae_extra import compute_dim_reduction
from funasr.models.tsasr.dae_kws_asr_model import KwsAsrConformerDae
from funasr.models.tsasr.dae_modules import (
    DepthWiseConv, FNNBlock, MultiHeadAtt, PositionalEncoding, make_mask,
)
from funasr.register import tables


def apply_lfr_batch(feats, lengths, lfr_m, lfr_n):
    """Paraformer's low frame rate stacking, over a padded batch, length aware.

    Splices `lfr_m` consecutive frames into one and hops by `lfr_n`, so the output
    is `ceil(T / lfr_n)` frames of `lfr_m * D` dims.  This is the same operation as
    `funasr.frontends.wav_frontend.apply_lfr`, which Paraformer applies to the whole
    input (`lfr_m: 7, lfr_n: 6`); here it is applied to the enrollment stream only.

    That reference implementation takes a single unpadded `(T, D)` utterance: it
    prepends `(lfr_m - 1) // 2` copies of the first frame and appends copies of the
    last one, then reads the result with `as_strided`.  Running it on a padded batch
    would splice padding into the last valid output frame and repeat a padding frame
    as the right padding, so it is reproduced here as a gather whose source indices
    are clamped to `[0, length_i - 1]`.  Clamping is exactly equivalent to those two
    edge paddings, and it is per sample.

    Args:
        feats: (B, T, D) padded features.
        lengths: (B,) valid frames per sample.
        lfr_m: frames spliced into one output frame.
        lfr_n: hop, i.e. the downsampling factor.

    Returns:
        `(out (B, T_lfr, lfr_m * D), out_lengths (B,))`.  Positions beyond
        `out_lengths` are zero rather than a repeat of the last valid frame.
    """
    if lfr_m < 1 or lfr_n < 1:
        raise ValueError(f"lfr_m/lfr_n must be >= 1, got {lfr_m}/{lfr_n}")
    batch, _, dim = feats.shape
    lengths = lengths.to(device=feats.device).view(-1).to(torch.long)
    out_lengths = torch.div(lengths + lfr_n - 1, lfr_n, rounding_mode="floor")
    out_frames = int(out_lengths.max().clamp(min=1))

    left_pad = (lfr_m - 1) // 2
    steps = torch.arange(out_frames, device=feats.device).unsqueeze(1)
    slots = torch.arange(lfr_m, device=feats.device).unsqueeze(0)
    index = (steps * lfr_n + slots - left_pad).clamp(min=0)          # (T_lfr, lfr_m)
    last = (lengths - 1).clamp(min=0).view(-1, 1, 1)
    index = index.unsqueeze(0).expand(batch, -1, -1).clamp(max=last)  # (B, T_lfr, lfr_m)

    gathered = feats.gather(
        1, index.reshape(batch, -1, 1).expand(-1, -1, dim)
    ).reshape(batch, out_frames, lfr_m * dim)
    keep = ~make_mask(out_lengths, max_len=out_frames)
    return gathered * keep.unsqueeze(-1).to(dtype=gathered.dtype), out_lengths.to(torch.int32)



class SpeakerFiLM(nn.Module):
    def __init__(self, hidden_dim, speaker_dim, bottleneck=0):
        super().__init__()
        if bottleneck:
            self.gen = nn.Sequential(
                nn.Linear(speaker_dim, bottleneck), nn.ReLU(),
                nn.Linear(bottleneck, hidden_dim * 2),
            )
            last = self.gen[-1]
        else:
            self.gen = nn.Linear(speaker_dim, hidden_dim * 2)
            last = self.gen
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x, spk_emb):
        gamma, beta = self.gen(spk_emb).chunk(2, dim=-1)
        return (1.0 + gamma.unsqueeze(1)) * x + beta.unsqueeze(1)


class SpeakerConformerLayer(nn.Module):
    """Conformer block with a speaker-only FiLM adapter after self-attention."""

    def __init__(self, size, heads, ffn_dim, conv_channels, conv_kernel_size, conv_norm,
                 spk_emb_dim, spk_film_hidden=0):
        super().__init__()
        self.macaron_layer = FNNBlock(list(ffn_dim))
        self.macaron_norm = nn.LayerNorm(size, eps=1e-5)
        self.self_att = MultiHeadAtt(n_head=heads, n_feats=size)
        self.input_norm = nn.LayerNorm(size, eps=1e-5)
        self.spk_film = SpeakerFiLM(size, spk_emb_dim, spk_film_hidden)
        self.conv_norm = nn.LayerNorm(size, eps=1e-5)
        self.conv_layer = DepthWiseConv(conv_channels, conv_kernel_size, norm=conv_norm)
        self.fnn_norm = nn.LayerNorm(size, eps=1e-5)
        self.feed_forward = FNNBlock(list(ffn_dim))

    def forward(self, x, mask, spk_emb):
        residual = x
        x = residual + 0.5 * self.macaron_layer(self.macaron_norm(x))

        residual = x
        x_norm = self.input_norm(x)
        context, _ = self.self_att(x_norm, x_norm, x_norm, mask)
        x = residual + context

        # Speaker-only condition: no text cue or cross-attention exists in this
        # architecture.  Mask again because FiLM beta would otherwise fill pads.
        x = self.spk_film(x, spk_emb)
        x = x * mask.transpose(1, 2).to(dtype=x.dtype)

        residual = x
        x = residual + self.conv_layer(self.conv_norm(x))
        residual = x
        return residual + self.feed_forward(self.fnn_norm(x))


class SpeakerCrossAttConformerLayer(nn.Module):
    """Conformer block with frame-level enrollment cross-attention.

    The mixture stream is the query and the complete, time-resolved enrollment
    stream is key/value.

    Two knobs control how closely this matches the text-cue arm
    (``dae_kws_encoders.ConformerKceEncoderDae``), which is the other single-cue
    baseline:

    - ``cross_att_heads``: heads for the enrollment attention only.  Defaults to
      ``heads`` (the self-attention count, 4) to preserve the original model;
      set to 1 to match the text-cue arm.
    - ``use_enroll_gate``: with the gate, ``enroll_gate`` starts at zero so a
      newly created model is bit-identical to a plain Conformer and every layer
      learns how much enrollment it wants — that is what makes warm-starting
      from a plain-CTC checkpoint meaningful.  Without it the block adds the
      cross-attention context on a bare residual, exactly like the text-cue arm.
      Note the gate adds no expressive power: a per-channel scale of the
      attention output can be absorbed into ``enroll_att.linear_out``'s rows, so
      this switch changes the initialisation and the optimisation path, not the
      function family.
    """

    def __init__(self, size, heads, ffn_dim, conv_channels, conv_kernel_size, conv_norm,
                 cross_att_heads=None, use_enroll_gate=True):
        super().__init__()
        self.macaron_layer = FNNBlock(list(ffn_dim))
        self.macaron_norm = nn.LayerNorm(size, eps=1e-5)
        self.self_att = MultiHeadAtt(n_head=heads, n_feats=size)
        self.input_norm = nn.LayerNorm(size, eps=1e-5)
        cross_att_heads = heads if cross_att_heads is None else int(cross_att_heads)
        self.enroll_att = MultiHeadAtt(n_head=cross_att_heads, n_feats=size)
        self.enroll_q_norm = nn.LayerNorm(size, eps=1e-5)
        self.enroll_kv_norm = nn.LayerNorm(size, eps=1e-5)
        if use_enroll_gate:
            self.enroll_gate = nn.Parameter(torch.zeros(size))
        else:
            # Keeps `self.enroll_gate` readable as None and out of the state dict.
            self.register_parameter("enroll_gate", None)
        self.conv_norm = nn.LayerNorm(size, eps=1e-5)
        self.conv_layer = DepthWiseConv(conv_channels, conv_kernel_size, norm=conv_norm)
        self.fnn_norm = nn.LayerNorm(size, eps=1e-5)
        self.feed_forward = FNNBlock(list(ffn_dim))

    def forward(self, x, mask, enroll, enroll_mask, return_att=False):
        residual = x
        x = residual + 0.5 * self.macaron_layer(self.macaron_norm(x))

        residual = x
        x_norm = self.input_norm(x)
        context, _ = self.self_att(x_norm, x_norm, x_norm, mask)
        x = residual + context

        # Every mixture layer queries all valid enrollment frames.  This is
        # intentionally a sequence-to-sequence interaction, not an embedding
        # injection.  The key mask prevents padded enrollment fbank frames from
        # participating in the softmax.
        residual = x
        context, att = self.enroll_att(
            self.enroll_q_norm(x), self.enroll_kv_norm(enroll),
            self.enroll_kv_norm(enroll), enroll_mask,
        )
        x = residual + (
            context if self.enroll_gate is None
            else torch.tanh(self.enroll_gate).view(1, 1, -1) * context
        )
        x = x * mask.transpose(1, 2).to(dtype=x.dtype)

        residual = x
        x = residual + self.conv_layer(self.conv_norm(x))
        residual = x
        x = residual + self.feed_forward(self.fnn_norm(x))
        return x, att if return_att else None


@tables.register("encoder_classes", "ConformerSpeakerCrossAttEncoder")
class ConformerSpeakerCrossAttEncoder(nn.Module):
    """Embedding-free TS-ASR encoder with enrollment interaction in every block.

    Mixture and enrollment share the fbank subsampler/projection, but enrollment
    remains a frame sequence rather than being passed through CAM++ or pooled to
    a fixed-dimensional speaker vector.  The same encoded enrollment memory is
    supplied to all Conformer layers, so the comparison with the embedding-FiLM
    baseline changes only the conditioning mechanism.

    Optional enrollment-only LFR (`enroll_lfr_m`/`enroll_lfr_n`, both 1 = off,
    which keeps this class bit-identical to its pre-LFR behaviour) shortens the
    cross-attention key sequence.  Cross-attention costs `O(T_mix * T_enroll)` in
    every one of the 9 blocks, and a 5 s enrollment is ~500 fbank frames, so this
    is the cheapest lever on enrollment cost.  The mixture is never LFR'd: its
    frame rate is what the CTC output resolution depends on.

    `enroll_lfr_where` picks the injection point, which is a real trade-off:

    - `fbank`: LFR on the raw 80-dim enrollment fbank, i.e. Paraformer's own
      position, followed by `enroll_lfr_proj` (`lfr_m*80 -> D`) instead of the
      shared `au_conv`.  Skips the Conv2d on the enrollment entirely, but the two
      streams no longer share the first mapping.  Note the shared front end
      already subsamples 4x, so `lfr_n` must exceed 4 to make the sequence
      shorter than the LFR-off path (`lfr_n: 3` here gives 167 frames vs 124).
    - `encoded`: LFR after the shared subsampler, on the D-dim enrollment frames,
      followed by `enroll_lfr_proj` (`lfr_m*D -> D`).  Weight sharing is fully
      preserved and the total enrollment downsampling is `4 * lfr_n`
      (`lfr_n: 3` -> ~42 frames), but the Conv2d still runs on the enrollment.

    `cross_att_heads` and `use_enroll_gate` exist to make this arm structurally
    comparable with the text-cue arm; see `SpeakerCrossAttConformerLayer`.  Both
    default to the original behaviour (4 heads, gate on).  Changing either one
    breaks checkpoint compatibility in a way that is worth knowing about:
    dropping the gate leaves `enroll_gate` as an unexpected key, while changing
    the head count changes **no parameter shape at all** (every projection is
    `Linear(D, D)`; only the reshape differs), so an old checkpoint would load
    silently and compute a different attention.
    """

    _LFR_WHERE = ("fbank", "encoded")

    def __init__(
        self, input_size=80, au_hidden_dim=256, num_mel_bins=80, num_blocks=9,
        input_trans_dim=None, self_att_heads=4, ffn_dim=None, conv_channels=256,
        conv_kernel_size=15, conv_norm="batch_norm", mask_pad_before_conv=True,
        enroll_lfr_m=1, enroll_lfr_n=1, enroll_lfr_where="encoded",
        cross_att_heads=None, use_enroll_gate=True,
        **kwargs,
    ):
        super().__init__()
        if input_size != num_mel_bins:
            raise ValueError("ConformerSpeakerCrossAttEncoder requires 80-dim fbank input")
        self._output_size = int(au_hidden_dim)
        self.num_mel_bins = int(num_mel_bins)
        self.mask_pad_before_conv = bool(mask_pad_before_conv)
        input_trans_dim = input_trans_dim or [au_hidden_dim, au_hidden_dim]
        ffn_dim = ffn_dim or [au_hidden_dim, 4 * au_hidden_dim, au_hidden_dim]
        # Shared front end: identical weights turn mixture and enrollment into
        # comparable acoustic tokens without discarding enrollment time detail.
        self.au_conv = nn.Sequential(
            nn.Conv2d(1, au_hidden_dim, 3, 2), nn.ReLU(),
            nn.Conv2d(au_hidden_dim, au_hidden_dim, 3, 2), nn.ReLU(),
        )
        freq_dim = ((num_mel_bins - 1) // 2 - 1) // 2
        self.au_conv_trans = nn.Linear(au_hidden_dim * freq_dim, au_hidden_dim)
        self.input_projection = FNNBlock(list(input_trans_dim))
        self.positional_encoding = PositionalEncoding(au_hidden_dim)

        self.enroll_lfr_m = int(enroll_lfr_m)
        self.enroll_lfr_n = int(enroll_lfr_n)
        self.enroll_lfr_where = str(enroll_lfr_where)
        self.use_enroll_lfr = self.enroll_lfr_m > 1 or self.enroll_lfr_n > 1
        self.enroll_lfr_proj = None
        if self.use_enroll_lfr:
            if self.enroll_lfr_where not in self._LFR_WHERE:
                raise ValueError(
                    f"enroll_lfr_where must be one of {self._LFR_WHERE}, "
                    f"got {enroll_lfr_where!r}"
                )
            spliced_dim = self.enroll_lfr_m * (
                self.num_mel_bins if self.enroll_lfr_where == "fbank" else au_hidden_dim
            )
            self.enroll_lfr_proj = nn.Linear(spliced_dim, au_hidden_dim)
            logging.info(
                "[ConformerSpeakerCrossAttEncoder] enrollment LFR m=%d n=%d at %r "
                "(total enrollment downsampling %dx)",
                self.enroll_lfr_m, self.enroll_lfr_n, self.enroll_lfr_where,
                self.enroll_lfr_n if self.enroll_lfr_where == "fbank" else 4 * self.enroll_lfr_n,
            )

        self.cross_att_heads = self_att_heads if cross_att_heads is None else int(cross_att_heads)
        self.use_enroll_gate = bool(use_enroll_gate)
        self.layers = nn.ModuleList([
            SpeakerCrossAttConformerLayer(
                au_hidden_dim, self_att_heads, ffn_dim, conv_channels,
                conv_kernel_size, conv_norm,
                cross_att_heads=self.cross_att_heads,
                use_enroll_gate=self.use_enroll_gate,
            ) for _ in range(num_blocks)
        ])

    def output_size(self):
        return self._output_size

    @staticmethod
    def subsampled_lengths(lengths):
        lengths = compute_dim_reduction(lengths, 3, 2, 0, 1)
        return compute_dim_reduction(lengths, 3, 2, 0, 1)

    def _subsample(self, speech):
        """`au_conv` + `au_conv_trans`: (B, T, mel) -> (B, T/4, D)."""
        x = self.au_conv(speech.unsqueeze(1))
        batch, channels, frames, freq = x.shape
        return self.au_conv_trans(
            x.transpose(1, 2).contiguous().view(batch, frames, channels * freq)
        )

    def _project(self, x):
        """Shared `input_projection` + positional encoding."""
        return self.positional_encoding(self.input_projection(x))

    @staticmethod
    def _mask_from(lengths, frames):
        lengths = torch.clamp(lengths, max=frames)
        return lengths, (~make_mask(lengths, max_len=frames)).unsqueeze(1)

    def _frontend(self, speech, speech_lengths):
        """Mixture path, also the enrollment path when LFR is off."""
        x = self._project(self._subsample(speech))
        lengths, mask = self._mask_from(
            self.subsampled_lengths(speech_lengths.view(-1)), x.size(1)
        )
        return x, lengths, mask

    def _enroll_frontend(self, ref_speech, ref_speech_lengths):
        """Enrollment path, with the optional LFR at the configured position."""
        if not self.use_enroll_lfr:
            return self._frontend(ref_speech, ref_speech_lengths)

        lengths = ref_speech_lengths.view(-1)
        if self.enroll_lfr_where == "fbank":
            # Paraformer's position: no Conv2d on the enrollment at all.
            spliced, lengths = apply_lfr_batch(
                ref_speech, lengths, self.enroll_lfr_m, self.enroll_lfr_n
            )
            hidden = self.enroll_lfr_proj(spliced)
        else:
            hidden = self._subsample(ref_speech)
            lengths, _ = self._mask_from(
                self.subsampled_lengths(lengths), hidden.size(1)
            )
            spliced, lengths = apply_lfr_batch(
                hidden, lengths, self.enroll_lfr_m, self.enroll_lfr_n
            )
            hidden = self.enroll_lfr_proj(spliced)
        hidden = self._project(hidden)
        lengths, mask = self._mask_from(lengths, hidden.size(1))
        return hidden, lengths, mask

    def forward(self, speech, speech_lengths, ref_speech=None, ref_speech_lengths=None,
                return_spk_att=False, **kwargs):
        if ref_speech is None or ref_speech_lengths is None:
            raise ValueError(
                "ConformerSpeakerCrossAttEncoder requires ref_speech and ref_speech_lengths; "
                "embedding-free conditioning cannot use precomputed spk_emb."
            )
        x, lengths, mask = self._frontend(speech, speech_lengths)
        enroll, _, enroll_mask = self._enroll_frontend(ref_speech, ref_speech_lengths)
        pad_mask = ~mask.transpose(1, 2)
        att_layers = []
        for layer in self.layers:
            if self.mask_pad_before_conv:
                x = x.masked_fill(pad_mask, 0.0)
            x, att = layer(x, mask, enroll, enroll_mask, return_att=return_spk_att)
            if return_spk_att:
                att_layers.append(att)
        extra = {"spk_cross_att": att_layers} if return_spk_att else {}
        return x, lengths, extra


@tables.register("encoder_classes", "ConformerSpeakerFiLMEncoder")
class ConformerSpeakerFiLMEncoder(nn.Module):
    """9-layer DAE-style Conformer with speaker FiLM and no text-cue path."""

    def __init__(
        self, input_size=80, au_hidden_dim=256, num_mel_bins=80, num_blocks=9,
        input_trans_dim=None, self_att_heads=4, ffn_dim=None, conv_channels=256,
        conv_kernel_size=15, conv_norm="batch_norm", mask_pad_before_conv=True,
        spk_emb_dim=192, spk_film_hidden=0, **kwargs,
    ):
        super().__init__()
        if input_size != num_mel_bins:
            raise ValueError("ConformerSpeakerFiLMEncoder requires 80-dim fbank input")
        self._output_size = int(au_hidden_dim)
        self.mask_pad_before_conv = bool(mask_pad_before_conv)
        self.spk_emb_dim = int(spk_emb_dim)
        input_trans_dim = input_trans_dim or [au_hidden_dim, au_hidden_dim]
        ffn_dim = ffn_dim or [au_hidden_dim, 4 * au_hidden_dim, au_hidden_dim]
        self.au_conv = nn.Sequential(
            nn.Conv2d(1, au_hidden_dim, 3, 2), nn.ReLU(),
            nn.Conv2d(au_hidden_dim, au_hidden_dim, 3, 2), nn.ReLU(),
        )
        freq_dim = ((num_mel_bins - 1) // 2 - 1) // 2
        self.au_conv_trans = nn.Linear(au_hidden_dim * freq_dim, au_hidden_dim)
        self.input_projection = FNNBlock(list(input_trans_dim))
        self.positional_encoding = PositionalEncoding(au_hidden_dim)
        self.layers = nn.ModuleList([
            SpeakerConformerLayer(
                au_hidden_dim, self_att_heads, ffn_dim, conv_channels,
                conv_kernel_size, conv_norm, self.spk_emb_dim, spk_film_hidden,
            ) for _ in range(num_blocks)
        ])

    def output_size(self):
        return self._output_size

    @staticmethod
    def subsampled_lengths(lengths):
        lengths = compute_dim_reduction(lengths, 3, 2, 0, 1)
        return compute_dim_reduction(lengths, 3, 2, 0, 1)

    def forward(self, speech, speech_lengths, spk_emb=None, **kwargs):
        if spk_emb is None:
            raise ValueError("ConformerSpeakerFiLMEncoder requires spk_emb")
        if spk_emb.size(-1) != self.spk_emb_dim:
            raise ValueError(f"spk_emb dim {spk_emb.size(-1)} != {self.spk_emb_dim}")
        lengths = self.subsampled_lengths(speech_lengths.view(-1))
        x = self.au_conv(speech.unsqueeze(1))
        b, c, t, d = x.shape
        x = self.au_conv_trans(x.transpose(1, 2).contiguous().view(b, t, c * d))
        x = self.positional_encoding(self.input_projection(x))
        lengths = torch.clamp(lengths, max=t)
        mask = (~make_mask(lengths, max_len=t)).unsqueeze(1)
        pad_mask = ~mask.transpose(1, 2)
        spk_emb = spk_emb.to(dtype=x.dtype)
        for layer in self.layers:
            if self.mask_pad_before_conv:
                x = x.masked_fill(pad_mask, 0.0)
            x = layer(x, mask, spk_emb)
        return x, lengths, {}


@tables.register("model_classes", "SpeakerConformerCTCOnlineEnroll")
class SpeakerConformerCTCOnlineEnroll(KwsAsrConformerDae):
    """CTC TS-ASR using only an online frozen CAM++ enrollment condition."""

    def __init__(self, spk_encoder="CAMPPlus", spk_encoder_conf=None,
                 spk_emb_dim=192, spk_emb_norm="l2", skip_spk_encoder=False, **kwargs):
        encoder_conf = dict(kwargs.get("encoder_conf") or {})
        encoder_conf["spk_emb_dim"] = int(spk_emb_dim)
        kwargs["encoder_conf"] = encoder_conf
        kwargs["cue_encoder"] = None
        super().__init__(**kwargs)
        self.spk_encoder = None
        if not skip_spk_encoder:
            spk_cls = tables.spk_encoder_classes.get(spk_encoder)
            if spk_cls is None:
                raise ValueError(f"Unknown speaker encoder: {spk_encoder}")
            conf = dict(spk_encoder_conf or {})
            conf["freeze"] = True
            self.spk_encoder = spk_cls(**conf)
            if int(self.spk_encoder.output_size()) != int(spk_emb_dim):
                raise ValueError("CAM++ output dimension does not match spk_emb_dim")
        self.spk_emb_norm = str(spk_emb_norm).lower()
        self._ref_speech = None
        self._ref_speech_lengths = None
        self._precomputed_spk_emb = None

    def train(self, mode=True):
        super().train(mode)
        if self.spk_encoder is not None:
            self.spk_encoder.train(False)
        return self

    def forward(self, speech, speech_lengths, text, text_lengths, ref_speech=None,
                ref_speech_lengths=None, spk_emb=None, **kwargs):
        self._ref_speech, self._ref_speech_lengths = ref_speech, ref_speech_lengths
        self._precomputed_spk_emb = spk_emb
        try:
            return super().forward(speech, speech_lengths, text, text_lengths, **kwargs)
        finally:
            self._ref_speech = self._ref_speech_lengths = None
            self._precomputed_spk_emb = None

    def inference(self, *args, ref_speech=None, ref_speech_lengths=None, spk_emb=None, **kwargs):
        """Decode with online enrollment fbank or offline CAM++ embeddings.

        ``ref_speech`` is (B, T_ref, 80), using the same Kaldi fbank + utterance
        mean normalisation as the training dataset.  Alternatively ``spk_emb``
        is a (B, 192) vector batch looked up from the offline enrollment store.
        """
        self._ref_speech, self._ref_speech_lengths = ref_speech, ref_speech_lengths
        self._precomputed_spk_emb = spk_emb
        try:
            return super().inference(*args, **kwargs)
        finally:
            self._ref_speech = self._ref_speech_lengths = None
            self._precomputed_spk_emb = None

    def encode(self, speech, speech_lengths, **kwargs):
        spk_emb = self._precomputed_spk_emb
        if spk_emb is None and self._ref_speech is None:
            raise ValueError("SpeakerConformerCTCOnlineEnroll requires ref_speech or spk_emb")
        with torch.cuda.amp.autocast(enabled=False):
            if self.specaug is not None and self.training:
                speech, speech_lengths = self.specaug(speech, speech_lengths)
            if self.normalize is not None:
                speech, speech_lengths = self.normalize(speech, speech_lengths)
            if spk_emb is None:
                if self.spk_encoder is None:
                    raise ValueError(
                        "This model was built for offline spk_emb inference; "
                        "provide spk_emb instead of ref_speech."
                    )
                ref = self._ref_speech.to(device=speech.device, dtype=torch.float32)
                ref_lens = self._ref_speech_lengths.to(device=speech.device).view(-1)
                spk_emb = self.spk_encoder(ref, ref_lens)
            else:
                spk_emb = spk_emb.to(device=speech.device, dtype=torch.float32)
        if self.spk_emb_norm == "l2":
            spk_emb = spk_emb / spk_emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return self.encoder(speech, speech_lengths, spk_emb=spk_emb)


@tables.register("model_classes", "SpeakerConformerCTCEmbeddingFree")
class SpeakerConformerCTCEmbeddingFree(KwsAsrConformerDae):
    """CTC TS-ASR with full enrollment-frame interaction at every Conformer layer."""

    def __init__(self, **kwargs):
        # The inherited DAE model would otherwise create a text-cue branch.  This
        # arm is conditioned exclusively by ref_speech in its acoustic encoder.
        kwargs["cue_encoder"] = None
        super().__init__(**kwargs)
        self._ref_speech = None
        self._ref_speech_lengths = None

    def forward(self, speech, speech_lengths, text, text_lengths, ref_speech=None,
                ref_speech_lengths=None, **kwargs):
        self._ref_speech, self._ref_speech_lengths = ref_speech, ref_speech_lengths
        try:
            return super().forward(speech, speech_lengths, text, text_lengths, **kwargs)
        finally:
            self._ref_speech = self._ref_speech_lengths = None

    def inference(self, *args, ref_speech=None, ref_speech_lengths=None, **kwargs):
        self._ref_speech, self._ref_speech_lengths = ref_speech, ref_speech_lengths
        try:
            return super().inference(*args, **kwargs)
        finally:
            self._ref_speech = self._ref_speech_lengths = None

    def encode(self, speech, speech_lengths, **kwargs):
        if self._ref_speech is None or self._ref_speech_lengths is None:
            raise ValueError(
                "SpeakerConformerCTCEmbeddingFree requires online ref_speech and "
                "ref_speech_lengths; offline speaker embeddings are unsupported."
            )
        if self.specaug is not None and self.training:
            speech, speech_lengths = self.specaug(speech, speech_lengths)
        if self.normalize is not None:
            speech, speech_lengths = self.normalize(speech, speech_lengths)
        ref = self._ref_speech.to(device=speech.device, dtype=speech.dtype)
        ref_lengths = self._ref_speech_lengths.to(device=speech.device).view(-1)
        return self.encoder(
            speech, speech_lengths, ref_speech=ref, ref_speech_lengths=ref_lengths,
            return_spk_att=bool(kwargs.get("return_spk_att", False)),
        )
