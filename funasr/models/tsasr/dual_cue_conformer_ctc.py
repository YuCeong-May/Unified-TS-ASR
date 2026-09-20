# -*- coding: utf-8 -*-
"""Conformer + CTC backbone conditioned on a text cue and/or an enrollment-speech cue.

This is the unified arm of the TS-ASR experiments.  The two existing arms each
carry exactly one cue and hard-fail without it:

- `dae_kws_encoders.ConformerKceEncoderDae`  text cue only (keyword char ids).
- `speaker_conformer_ctc.ConformerSpeakerCrossAttEncoder`  enrollment fbank only.

`ConformerDualCueEncoder` keeps the same 9x256 DAE Conformer backbone and the
same per-layer cross-attention injection point, but every block owns **two**
independent cross-attention adapters.  Both are optional, per sample and per
batch, so one model covers all four conditions:

    no cue          plain CTC ASR
    text cue only   keyword-conditioned target-speaker ASR
    enroll only     embedding-free enrollment-conditioned TS-ASR
    text + enroll   both cues at once

Three mechanisms make the mixed-cue batch work:

1. **Zero-initialised per-channel gates** (`tanh(gate)`, `gate=0` at init), when
   `use_cue_gate` is on.  A freshly built model is then bit-identical to the plain
   Conformer baseline, and each layer learns how much of each cue it wants.  This
   is what lets the dual-cue model be initialised from a plain-CTC or single-cue
   checkpoint.  With `use_cue_gate: false` the adapters add their context on a
   bare residual, matching the text-cue baseline; the function family is the same
   either way (see `CueCrossAtt`), but warm-starting no longer preserves the
   checkpoint's behaviour at step 0.
2. **Per-sample presence flags**.  A sample that carries no text cue gets the
   text context multiplied by 0, so its forward *and* its gradient are exactly
   those of a model without the text branch.  Absence is therefore not a
   distribution shift: it is the same computation.
3. **Length clamping to >= 1 valid key**.  An all-padding key row would send
   `softmax` a row of `-inf`, whose `NaN` survives into the backward pass even
   though `masked_fill` hides it in the forward pass.  Rows with an empty cue are
   clamped to one (arbitrary) valid key and then zeroed by their presence flag.

`funasr.models.tsasr.dae_modules` attention takes **True = valid** masks; the
`make_mask` / `combine_mask` helpers produce True = padding, hence the `~`.
"""
import logging

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from funasr.models.tsasr.dae_extra import compute_dim_reduction
from funasr.models.tsasr.dae_kws_asr_model import KwsAsrConformerDae
from funasr.models.tsasr.dae_modules import (
    DepthWiseConv,
    FNNBlock,
    MultiHeadAtt,
    MultiHeadCrossAtt,
    PositionalEncoding,
    combine_mask,
    make_mask,
)
from funasr.register import tables

class CueCrossAtt(nn.Module):
    """Cross-attention adapter with an optional zero-init gate and a presence mask.

    Args:
        size: model dimension D.
        heads: cross-attention heads (DAE-TSE uses 1).
        norm: pre-norm type inside `MultiHeadCrossAtt`.
        use_gate: add a per-channel `tanh(gate)` scale, zero-initialised.  With
            the gate a fresh model is bit-identical to a plain Conformer, which is
            what makes warm-starting meaningful; without it the context lands on a
            bare residual, exactly like the text-cue baseline
            `dae_kws_encoders.ConformerKceEncoderDae`.  The gate adds no expressive
            power (a per-channel scale of the attention output is absorbable into
            `att.linear_out`'s rows) — it changes the initialisation and the
            optimisation path only.
    """

    def __init__(self, size, heads=1, norm="LayerNorm", use_gate=True):
        super().__init__()
        self.att = MultiHeadCrossAtt(n_head=heads, n_feats=size, norm=norm)
        if use_gate:
            # tanh(0) == 0: the adapter contributes nothing until it is trained.
            self.gate = nn.Parameter(torch.zeros(size))
        else:
            # Keeps `self.gate` readable as None and out of the state dict.
            self.register_parameter("gate", None)

    def forward(self, x, memory, cross_mask, present=None):
        """Add the (optionally gated) cue context to the mixture stream.

        Args:
            x: (B, T', D) mixture hidden states (also the query).
            memory: (B, L, D) cue key/value.
            cross_mask: (B, 1, T', L) bool, True = valid.
            present: (B,) bool/float, 0 for samples that carry no cue.

        Returns:
            (x + context, attention score (B, heads, T', L)).
        """
        context, score = self.att(x, memory, memory, cross_mask)
        if self.gate is not None:
            context = context * torch.tanh(self.gate).view(1, 1, -1)
        if present is not None:
            context = context * present.view(-1, 1, 1).to(dtype=context.dtype)
        return x + context, score


class DualCueConformerLayer(nn.Module):
    """DAE-style pre-norm Conformer block with the cue adapters.

    `cue_fusion` decides how the two cues reach the mixture stream:

    - `parallel` (default): two independent adapters applied in sequence,
      macaron FFN -> self-att -> text cross-att -> enroll cross-att -> conv -> FFN.
      Each cue owns its softmax and its gate.  The text cue comes first only so
      that its attention map stays comparable with the single-cue V4 model, where
      it is the sole adapter at this position.
    - `concat`: one adapter over the two memories concatenated along time, so a
      single softmax spans both cue types.  Simpler (one gate, one attention), but
      the two cues compete for the same attention mass, and the ~7-token text cue
      is normalised against a ~124-frame enrollment.
    """

    _FUSION = ("parallel", "concat")

    def __init__(self, size, heads, ffn_dim, conv_channels, conv_kernel_size, conv_norm,
                 use_text_cue=True, use_enroll_cue=True, cross_att_heads=1,
                 cross_att_norm="LayerNorm", cue_fusion="parallel", use_cue_gate=True):
        super().__init__()
        if cue_fusion not in self._FUSION:
            raise ValueError(f"cue_fusion must be one of {self._FUSION}, got {cue_fusion!r}")
        self.cue_fusion = cue_fusion
        self.macaron_layer = FNNBlock(list(ffn_dim))
        self.macaron_norm = nn.LayerNorm(size, eps=1e-5)
        self.self_att = MultiHeadAtt(n_head=heads, n_feats=size)
        self.input_norm = nn.LayerNorm(size, eps=1e-5)
        self.text_cue_att = self.enroll_cue_att = self.joint_cue_att = None
        adapter = lambda: CueCrossAtt(size, cross_att_heads, cross_att_norm, use_cue_gate)
        if cue_fusion == "concat":
            self.joint_cue_att = adapter()
        else:
            if use_text_cue:
                self.text_cue_att = adapter()
            if use_enroll_cue:
                self.enroll_cue_att = adapter()
        self.conv_norm = nn.LayerNorm(size, eps=1e-5)
        self.conv_layer = DepthWiseConv(conv_channels, conv_kernel_size, norm=conv_norm)
        self.fnn_norm = nn.LayerNorm(size, eps=1e-5)
        self.feed_forward = FNNBlock(list(ffn_dim))

    def forward(self, x, mask, text=None, enroll=None, joint=None):
        """One block.

        Args:
            x: (B, T', D) hidden states.
            mask: (B, 1, T') bool, True = valid frame.
            text: `(memory, cross_mask, present)` for the text cue, or None.
            enroll: `(memory, cross_mask, present)` for the enrollment cue, or None.
            joint: the same triple over the concatenated memory, `concat` mode only.

        Returns:
            `(x, text_score, enroll_score)`; in `concat` mode the joint score is
            returned as `text_score` (its first columns *are* the text keys) and
            `enroll_score` is None.
        """
        residual = x
        x = residual + 0.5 * self.macaron_layer(self.macaron_norm(x))

        residual = x
        x_norm = self.input_norm(x)
        context, _ = self.self_att(x_norm, x_norm, x_norm, mask)
        x = residual + context

        text_score = enroll_score = None
        if self.joint_cue_att is not None:
            if joint is not None:
                x, text_score = self.joint_cue_att(x, *joint)
        else:
            if self.text_cue_att is not None and text is not None:
                x, text_score = self.text_cue_att(x, *text)
            if self.enroll_cue_att is not None and enroll is not None:
                x, enroll_score = self.enroll_cue_att(x, *enroll)
        # The adapters write to padded frames as well (their gate bias is a
        # per-channel constant), so re-zero the padding before the conv.
        x = x * mask.transpose(1, 2).to(dtype=x.dtype)

        residual = x
        x = residual + self.conv_layer(self.conv_norm(x))
        residual = x
        x = residual + self.feed_forward(self.fnn_norm(x))
        return x, text_score, enroll_score


@tables.register("encoder_classes", "ConformerDualCueEncoder")
class ConformerDualCueEncoder(nn.Module):
    """9x256 DAE Conformer whose every block can attend to a text cue, an
    enrollment-speech cue, both, or neither.

    The mixture and the enrollment share one fbank front end (`au_conv` ->
    `au_conv_trans` -> `speech_input_projection` -> `speech_pe_module`), exactly
    like the embedding-free speaker arm: enrollment stays a **frame sequence**
    instead of being pooled into a speaker vector, so the two cues are injected
    by the same mechanism and only differ in what they carry.  Front-end and
    `speech_transformer` module names match `ConformerKceEncoderDae` so a V4
    text-cue checkpoint loads its acoustic weights into this encoder unchanged.

    Args:
        input_size: must equal `num_mel_bins`; `build_model` overrides it with
            `frontend.output_size()`, so the frontend must be LFR-free 80-dim fbank.
        au_hidden_dim: model dimension D.
        num_blocks: Conformer blocks (DAE-TSE 9).
        self_att_heads / cross_att_heads: heads for self- and cross-attention.
        use_text_cue / use_enroll_cue: build the corresponding adapter in every
            block.  Turning one off makes this encoder equivalent to the matching
            single-cue arm and drops its parameters from the checkpoint.
            Ignored when `cue_fusion: concat`, which has a single adapter.
        cue_fusion: `parallel` = one gated adapter per cue, applied in sequence
            (text then enroll).  `concat` = the two memories are concatenated along
            time and a single adapter attends over both, so one softmax spans the
            whole cue set.  See `DualCueConformerLayer`.
        use_cue_gate: per-channel zero-init `tanh(gate)` on every adapter.  Off
            makes the injection a bare residual add, matching the text-cue
            baseline; see `CueCrossAtt` for why this costs no expressive power.
        mask_pad_before_conv: zero padded frames before each depthwise conv.
    """

    def __init__(self, input_size=80, au_hidden_dim=256, num_mel_bins=80, num_blocks=9,
                 input_trans_dim=None, self_att_heads=4, cross_att_heads=1,
                 cross_att_norm="LayerNorm", ffn_dim=None, conv_channels=256,
                 conv_kernel_size=15, conv_norm="batch_norm", mask_pad_before_conv=True,
                 use_text_cue=True, use_enroll_cue=True, cue_fusion="parallel",
                 use_cue_gate=True, **kwargs):
        super().__init__()
        if int(input_size) != int(num_mel_bins):
            raise ValueError(
                f"ConformerDualCueEncoder: input_size({input_size}) != "
                f"num_mel_bins({num_mel_bins}); the frontend must be LFR-free 80-dim fbank"
            )
        if cue_fusion not in DualCueConformerLayer._FUSION:
            raise ValueError(
                f"cue_fusion must be one of {DualCueConformerLayer._FUSION}, "
                f"got {cue_fusion!r}"
            )
        self._output_size = int(au_hidden_dim)
        self.num_mel_bins = int(num_mel_bins)
        self.mask_pad_before_conv = bool(mask_pad_before_conv)
        self.cue_fusion = str(cue_fusion)
        self.use_cue_gate = bool(use_cue_gate)
        # A concat adapter reads both memories through one attention, so the two
        # branches cannot be switched off independently.
        self.use_text_cue = True if self.cue_fusion == "concat" else bool(use_text_cue)
        self.use_enroll_cue = True if self.cue_fusion == "concat" else bool(use_enroll_cue)
        input_trans_dim = input_trans_dim or [au_hidden_dim, au_hidden_dim]
        ffn_dim = ffn_dim or [au_hidden_dim, 4 * au_hidden_dim, au_hidden_dim]

        self.au_conv = nn.Sequential(
            nn.Conv2d(1, au_hidden_dim, 3, 2), nn.ReLU(),
            nn.Conv2d(au_hidden_dim, au_hidden_dim, 3, 2), nn.ReLU(),
        )
        self.freq_dim = ((self.num_mel_bins - 1) // 2 - 1) // 2
        self.au_conv_trans = nn.Linear(au_hidden_dim * self.freq_dim, au_hidden_dim)
        self.speech_input_projection = FNNBlock(list(input_trans_dim))
        self.speech_pe_module = PositionalEncoding(au_hidden_dim)
        self.speech_transformer = nn.ModuleList([
            DualCueConformerLayer(
                au_hidden_dim, self_att_heads, ffn_dim, conv_channels, conv_kernel_size,
                conv_norm, use_text_cue=self.use_text_cue, use_enroll_cue=self.use_enroll_cue,
                cross_att_heads=cross_att_heads, cross_att_norm=cross_att_norm,
                cue_fusion=self.cue_fusion, use_cue_gate=self.use_cue_gate,
            ) for _ in range(num_blocks)
        ])

    def output_size(self):
        return self._output_size

    @staticmethod
    def subsampled_lengths(lengths):
        """Frame count after the two kernel-3 / stride-2 / pad-0 convolutions."""
        lengths = compute_dim_reduction(lengths, 3, 2, 0, 1)
        return compute_dim_reduction(lengths, 3, 2, 0, 1)

    def _frontend(self, speech, speech_lengths):
        """Shared fbank front end used for both the mixture and the enrollment.

        Returns:
            (x (B, T', D), lengths (B,), mask (B, 1, T') with True = valid).
        """
        lengths = self.subsampled_lengths(speech_lengths.view(-1))
        x = self.au_conv(speech.unsqueeze(1))
        batch, channels, frames, freq = x.shape
        x = self.au_conv_trans(
            x.transpose(1, 2).contiguous().view(batch, frames, channels * freq)
        )
        x = self.speech_pe_module(self.speech_input_projection(x))
        lengths = torch.clamp(lengths, max=frames)
        mask = (~make_mask(lengths, max_len=frames)).unsqueeze(1)
        return x, lengths, mask


    # Two kernel-3 / stride-2 / pad-0 convolutions need at least this many frames
    # before the second one receives a window it can slide over.
    _MIN_FRONTEND_FRAMES = 7

    @staticmethod
    def _safe_valid(valid):
        """Force at least one valid key per row, returning `(valid, present)`.

        A row with no valid key would make the attention softmax see only `-inf`.
        `masked_fill` hides the resulting `NaN` in the forward pass, but softmax's
        backward still multiplies by the `NaN` weights, so the gradient of the
        whole batch becomes `NaN`.  Give such rows one dummy key instead and let
        the presence flag remove their contribution afterwards.
        """
        present = valid.any(dim=-1)
        if bool(present.all()):
            return valid, present
        valid = valid.clone()
        valid[~present, 0] = True
        return valid, present

    def _check_cue_dim(self, memory, name):
        """Reject a cue memory that is not already at the model dimension."""
        if memory.size(-1) != self._output_size:
            # There is no projection between a cue encoder and the adapters: the
            # cross-attention's q/k/v are all `Linear(D, D)`, so the cue memory has
            # to arrive at the model dimension already.  Without this check the
            # mismatch only shows up as a matmul shape error deep inside attention.
            raise ValueError(
                f"{name} cue memory is {memory.size(-1)}-dim but the encoder runs at "
                f"{self._output_size}; set cue_encoder_conf.dim == "
                f"encoder_conf.au_hidden_dim"
            )

    def _cue_branch(self, memory, valid, sph_valid, present, name):
        """Package a cue stream into the `(memory, cross_mask, present)` triple."""
        self._check_cue_dim(memory, name)
        valid, derived = self._safe_valid(valid)
        present = derived if present is None else (present.to(torch.bool) & derived)
        cross_mask = ~combine_mask(sph_valid, valid, 1)
        return memory, cross_mask, present

    def _joint_branch(self, pieces, sph_valid):
        """Concatenate the cue memories along time into a single attended stream.

        `concat` mode.  A cue that a sample does not carry is folded into the key
        mask instead of being dropped, so within one batch a text-only sample sees
        only its text keys while its neighbour sees both.  When a sample carries
        nothing at all, `_safe_valid` lends it one dummy key and the returned
        presence flag zeroes the whole context again.

        Text is concatenated first so the leading columns of the joint attention
        score are still the text keys, which is what the inherited `inference`
        slices with `att_last[i, :n_frame, :kw_len]`.

        Args:
            pieces: ordered `(name, (memory, valid, present) or None)` pairs.
            sph_valid: (B, T') bool query mask, True = valid frame.

        Returns:
            `((memory, cross_mask, present), text_key_count)`, or `(None, 0)` when
            the batch carries no cue at all.
        """
        memories, valids, text_keys = [], [], 0
        for name, piece in pieces:
            if piece is None:
                continue
            memory, valid, present = piece
            self._check_cue_dim(memory, name)
            if present is not None:
                valid = valid & present.to(torch.bool).view(-1, 1)
            memories.append(memory)
            valids.append(valid)
            if name == "text":
                text_keys = memory.size(1)
        if not memories:
            return None, 0
        joint = self._cue_branch(
            torch.cat(memories, dim=1), torch.cat(valids, dim=1), sph_valid, None, "joint"
        )
        return joint, text_keys

    def forward(self, speech, speech_lengths, cue_feat=None, cue_valid=None,
                ref_speech=None, ref_speech_lengths=None, text_cue_present=None,
                enroll_cue_present=None, return_att=False, **kwargs):
        """Encode the mixture with zero, one, or both cues.

        Args:
            speech: (B, T, 80) mixture fbank.
            speech_lengths: (B,) valid frames.
            cue_feat: (B, L, D) text-cue encoder output, or None for the whole batch.
            cue_valid: (B, L) bool, True = valid text-cue position.
            ref_speech: (B, T_ref, 80) enrollment fbank, or None for the whole batch.
            ref_speech_lengths: (B,) valid enrollment frames; 0 marks a sample
                without an enrollment cue.
            text_cue_present / enroll_cue_present: (B,) bool, per-sample override.
            return_att: also return the per-layer cross-attention scores.

        Returns:
            (encoder_out (B, T', D), encoder_out_lens (B,), extra dict).
        """
        x, olens, sph_mask = self._frontend(speech, speech_lengths)
        sph_valid = sph_mask.squeeze(1)

        text_pieces = None
        if self.use_text_cue and cue_feat is not None and cue_feat.size(1) > 0:
            if cue_valid is None:
                cue_valid = torch.ones(
                    cue_feat.shape[:2], dtype=torch.bool, device=cue_feat.device
                )
            text_pieces = (cue_feat.to(dtype=x.dtype), cue_valid, text_cue_present)

        enroll_pieces = None
        if self.use_enroll_cue and ref_speech is not None:
            if ref_speech.size(1) < self._MIN_FRONTEND_FRAMES:
                # A whole batch of empty enrollments: the shared subsampler cannot
                # run at all, so this is the no-enrollment condition by definition.
                logging.warning(
                    "[ConformerDualCueEncoder] ref_speech has %d frames (< %d); "
                    "skipping the enrollment cue for this batch.",
                    ref_speech.size(1), self._MIN_FRONTEND_FRAMES,
                )
            else:
                if ref_speech_lengths is None:
                    ref_speech_lengths = torch.full(
                        (ref_speech.size(0),), ref_speech.size(1),
                        dtype=torch.int32, device=ref_speech.device,
                    )
                memory, _, enroll_mask = self._frontend(
                    ref_speech.to(dtype=x.dtype), ref_speech_lengths
                )
                enroll_pieces = (memory, enroll_mask.squeeze(1), enroll_cue_present)

        text = enroll = joint = None
        text_keys = 0
        if self.cue_fusion == "concat":
            joint, text_keys = self._joint_branch(
                (("text", text_pieces), ("enroll", enroll_pieces)), sph_valid
            )
        else:
            if text_pieces is not None:
                memory, valid, present = text_pieces
                text = self._cue_branch(memory, valid, sph_valid, present, "text")
            if enroll_pieces is not None:
                memory, valid, present = enroll_pieces
                enroll = self._cue_branch(memory, valid, sph_valid, present, "enroll")

        pad_zero_mask = ~sph_mask.transpose(1, 2)
        text_scores, enroll_scores = [], []
        for layer in self.speech_transformer:
            if self.mask_pad_before_conv:
                x = x.masked_fill(pad_zero_mask, 0.0)
            x, text_score, enroll_score = layer(
                x, sph_mask, text=text, enroll=enroll, joint=joint
            )
            if return_att:
                text_scores.append(text_score)
                enroll_scores.append(enroll_score)
        extra = {}
        if return_att:
            # "cross_att" keeps the V4 key name so the inherited inference path
            # can plot the text-cue attention without a special case.  In `concat`
            # mode it holds the joint score, whose first `cue_split` columns are
            # the text keys and the rest the enrollment frames.
            extra = {"cross_att": text_scores, "enroll_att": enroll_scores}
            if self.cue_fusion == "concat":
                extra["cue_split"] = text_keys
        return x, olens, extra


@tables.register("model_classes", "DualCueConformerCTC")
class DualCueConformerCTC(KwsAsrConformerDae):
    """Conformer + CTC target-speaker ASR taking a text cue and/or an enrollment cue.

    Inherits the CTC loss, the token-level CER report, the sampled train-time
    `cue/ref/hyp/mix` log and the greedy `inference` (including `psok/peok` span
    extraction and `<no_kws/>` / `<no_tgt_spk/>` stripping) from
    `KwsAsrConformerDae`.  What is added here is the routing of the second cue and
    the per-sample presence bookkeeping.

    Batch fields consumed:
        speech / speech_lengths     mixture fbank.
        text / text_lengths         CTC target.
        cue_token / cue_token_lengths   text cue char ids, 0-padded; a length of 0
            means "this sample has no text cue".
        ref_speech / ref_speech_lengths enrollment fbank; a length of 0 means
            "this sample has no enrollment cue".
    Either group may be absent from the batch entirely, in which case the model
    runs as the corresponding single-cue or plain CTC model.

    Args:
        require_any_cue: raise if a batch arrives with neither cue.  Off by
            default so the plain-ASR condition can be evaluated with the same
            checkpoint.
    """

    def __init__(self, require_any_cue: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.require_any_cue = bool(require_any_cue)
        self._ref_speech = None
        self._ref_speech_lengths = None
        self._return_att = False
        self._warned_no_enroll_branch = False

    # ------------------------------------------------------------------ cues
    def _encode_text_cue(self, cue_token, cue_token_lengths):
        """Encode the text cue and report which samples actually carry one.

        Returns:
            (cue_feat (B, L, D), cue_valid (B, L), present (B,)) or three `None`s.
        """
        if self.cue_encoder is None or cue_token is None:
            return None, None, None
        if cue_token_lengths is None:
            lengths = torch.full(
                (cue_token.size(0),), cue_token.size(1),
                dtype=torch.int32, device=cue_token.device,
            )
        else:
            lengths = cue_token_lengths.to(cue_token.device).view(-1)
        present = lengths > 0
        # The cue encoder runs self-attention over the cue sequence, so an
        # all-padding row would produce NaN gradients there as well.  Give empty
        # rows one position and drop their contribution via `present`.
        safe_lengths = lengths.clamp(min=1)
        cue_feat, cue_valid = self.cue_encoder(cue_token.to(torch.long), safe_lengths)
        return cue_feat, cue_valid, present

    def _take_ref_speech(self, device, dtype):
        """Return the stashed enrollment batch moved onto the acoustic device."""
        if self._ref_speech is None:
            return None, None
        ref = self._ref_speech.to(device=device, dtype=dtype)
        if self._ref_speech_lengths is None:
            lengths = torch.full(
                (ref.size(0),), ref.size(1), dtype=torch.int32, device=device
            )
        else:
            lengths = self._ref_speech_lengths.to(device=device).view(-1)
        return ref, lengths


    def encode(self, speech, speech_lengths, cue_token=None, cue_token_lengths=None, **kwargs):
        """specaug -> normalize -> text-cue encoder -> dual-cue Conformer.

        Returns:
            (encoder_out, encoder_out_lens, extra).
        """
        with autocast(False):
            if self.specaug is not None and self.training:
                speech, speech_lengths = self.specaug(speech, speech_lengths)
            if self.normalize is not None:
                speech, speech_lengths = self.normalize(speech, speech_lengths)

        cue_feat, cue_valid, text_present = self._encode_text_cue(cue_token, cue_token_lengths)
        ref_speech, ref_lengths = self._take_ref_speech(speech.device, speech.dtype)
        if ref_speech is not None and not getattr(self.encoder, "use_enroll_cue", False):
            if not self._warned_no_enroll_branch:
                logging.warning(
                    "[%s] ref_speech supplied but encoder_conf.use_enroll_cue is false; "
                    "the enrollment cue is being ignored.", type(self).__name__,
                )
                self._warned_no_enroll_branch = True
        if self.require_any_cue and cue_feat is None and ref_speech is None:
            raise ValueError(
                f"{type(self).__name__} was built with require_any_cue=True but the batch "
                "carries neither cue_token nor ref_speech."
            )
        return self.encoder(
            speech,
            speech_lengths,
            cue_feat=cue_feat,
            cue_valid=cue_valid,
            ref_speech=ref_speech,
            ref_speech_lengths=ref_lengths,
            text_cue_present=text_present,
            enroll_cue_present=None,
            return_att=bool(kwargs.get("return_att", False)) or self._return_att,
        )

    def forward(self, speech, speech_lengths, text, text_lengths, cue_token=None,
                cue_token_lengths=None, ref_speech=None, ref_speech_lengths=None, **kwargs):
        """CTC forward; the enrollment batch is stashed for `encode`.

        The parent `forward` owns the loss, the CER stat and the sample log and
        calls `self.encode` with the text-cue arguments only, so the enrollment
        pair travels on the instance.  Same pattern as
        `speaker_conformer_ctc.SpeakerConformerCTCEmbeddingFree`.
        """
        self._ref_speech, self._ref_speech_lengths = ref_speech, ref_speech_lengths
        try:
            return super().forward(
                speech, speech_lengths, text, text_lengths,
                cue_token=cue_token, cue_token_lengths=cue_token_lengths, **kwargs,
            )
        finally:
            self._ref_speech = self._ref_speech_lengths = None

    def inference(self, *args, ref_speech=None, ref_speech_lengths=None, **kwargs):
        """Greedy CTC decoding with any combination of the two cues.

        `cue_text` / `cue_token` select the text cue (see the parent docstring);
        `ref_speech` is (B, T_ref, 80) enrollment fbank computed the same way as
        in training.  Omitting both decodes as a plain ASR model.
        """
        self._ref_speech, self._ref_speech_lengths = ref_speech, ref_speech_lengths
        self._return_att = bool(kwargs.get("return_att", False))
        try:
            return super().inference(*args, **kwargs)
        finally:
            self._ref_speech = self._ref_speech_lengths = None
            self._return_att = False

