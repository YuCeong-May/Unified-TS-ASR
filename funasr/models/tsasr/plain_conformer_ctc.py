"""Plain DAE-style Conformer encoder for the no-conditioning CTC baseline.

This deliberately shares the acoustic front end and every Conformer hyper-
parameter with the text-CUE / enrollment experiments, while omitting only the
condition injection (cross-attention or FiLM).
"""
import logging
import random

import torch
import torch.nn as nn
import torch.distributed as dist

from funasr.models.ctc.model import Transformer as CTCModel
from funasr.models.tsasr.dae_extra import compute_dim_reduction
from funasr.models.tsasr.dae_modules import DepthWiseConv, FNNBlock, MultiHeadAtt, PositionalEncoding, make_mask
from funasr.register import tables
from funasr.train_utils.device_funcs import force_gatherable


class PlainConformerLayer(nn.Module):
    """One pre-norm DAE Conformer block without a conditioning adapter."""

    def __init__(self, size, heads, ffn_dim, conv_channels, conv_kernel_size, conv_norm):
        super().__init__()
        self.macaron_layer = FNNBlock(list(ffn_dim))
        self.macaron_norm = nn.LayerNorm(size, eps=1e-5)
        self.self_att = MultiHeadAtt(n_head=heads, n_feats=size)
        self.input_norm = nn.LayerNorm(size, eps=1e-5)
        self.conv_norm = nn.LayerNorm(size, eps=1e-5)
        self.conv_layer = DepthWiseConv(conv_channels, conv_kernel_size, norm=conv_norm)
        self.fnn_norm = nn.LayerNorm(size, eps=1e-5)
        self.feed_forward = FNNBlock(list(ffn_dim))

    def forward(self, x, mask):
        residual = x
        x = residual + 0.5 * self.macaron_layer(self.macaron_norm(x))
        residual = x
        context, _ = self.self_att(self.input_norm(x), self.input_norm(x), self.input_norm(x), mask)
        x = residual + context
        residual = x
        x = residual + self.conv_layer(self.conv_norm(x))
        residual = x
        return residual + self.feed_forward(self.fnn_norm(x))


@tables.register("encoder_classes", "ConformerPlainEncoder")
class ConformerPlainEncoder(nn.Module):
    """Same 9x256 DAE acoustic backbone, with no CUE/enrollment path."""

    def __init__(self, input_size=80, au_hidden_dim=256, num_mel_bins=80, num_blocks=9,
                 input_trans_dim=None, self_att_heads=4, ffn_dim=None, conv_channels=256,
                 conv_kernel_size=15, conv_norm="batch_norm", mask_pad_before_conv=True, **kwargs):
        super().__init__()
        if int(input_size) != int(num_mel_bins):
            raise ValueError("ConformerPlainEncoder requires 80-dim fbank input")
        self._output_size = int(au_hidden_dim)
        self.mask_pad_before_conv = bool(mask_pad_before_conv)
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
            PlainConformerLayer(au_hidden_dim, self_att_heads, ffn_dim, conv_channels,
                                conv_kernel_size, conv_norm)
            for _ in range(num_blocks)
        ])

    def output_size(self):
        return self._output_size

    @staticmethod
    def subsampled_lengths(lengths):
        lengths = compute_dim_reduction(lengths, 3, 2, 0, 1)
        return compute_dim_reduction(lengths, 3, 2, 0, 1)

    def forward(self, speech, speech_lengths, **kwargs):
        lengths = self.subsampled_lengths(speech_lengths.view(-1))
        x = self.au_conv(speech.unsqueeze(1))
        batch, channels, frames, freq = x.shape
        x = self.au_conv_trans(x.transpose(1, 2).contiguous().view(batch, frames, channels * freq))
        x = self.positional_encoding(self.input_projection(x))
        lengths = torch.clamp(lengths, max=frames)
        mask = (~make_mask(lengths, max_len=frames)).unsqueeze(1)
        pad_mask = ~mask.transpose(1, 2)
        for layer in self.layers:
            if self.mask_pad_before_conv:
                x = x.masked_fill(pad_mask, 0.0)
            x = layer(x, mask)
        return x, lengths


@tables.register("model_classes", "PlainConformerCTC")
class PlainConformerCTC(CTCModel):
    """Plain CTC baseline with DAE-style sample/mix diagnostic logging.

    The encoder has no CUE or enrollment input.  The shared target-speaker
    dataset nevertheless returns ``mix_info`` so that a hypothesis can be
    checked against the actual target and interference utterances.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Older generic CTC implementations do not retain this AutoModel
        # argument as an attribute.  The plain diagnostic formatter needs it.
        self.token_list = kwargs.get("token_list", None)
        # Keep exactly the same trigger as dae_kws_asr_model.py: sample output
        # is printed alongside the trainer's ordinary log line.
        self.sample_log_interval = int(
            (kwargs.get("train_conf") or {}).get("log_interval", 0) or 0
        )
        self._plain_fwd_count = 0

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ):
        """CTC forward with an in-file sample-log hook.

        This is intentionally implemented here rather than relying on a hook
        added to the generic ``CTC`` parent: copying this one file to a
        training machine is enough for the plain baseline to emit its logs.
        """
        if text_lengths.ndim > 1:
            text_lengths = text_lengths[:, 0]
        if speech_lengths.ndim > 1:
            speech_lengths = speech_lengths[:, 0]

        batch_size = speech.shape[0]
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        loss_ctc, _ = self._calc_ctc_loss(
            encoder_out, encoder_out_lens, text, text_lengths
        )
        self._maybe_log_ctc_hyp(
            encoder_out,
            encoder_out_lens,
            text,
            text_lengths,
            mix_info=kwargs.get("mix_info"),
        )
        stats = {"loss": torch.clone(loss_ctc.detach())}
        if self.length_normalized_loss:
            batch_size = int((text_lengths + 1).sum())
        return force_gatherable((loss_ctc, stats, batch_size), loss_ctc.device)

    @torch.no_grad()
    def _ids2text(self, ids):
        """Render ids exactly like the DAE-KWS diagnostic log."""
        if not self.token_list:
            return " ".join(str(int(x)) for x in ids)
        return "".join(
            self.token_list[int(x)]
            if 0 <= int(x) < len(self.token_list)
            else f"<{int(x)}>"
            for x in ids
        )

    @staticmethod
    def _fmt_mix_info(mix_info, i):
        try:
            return "" if mix_info is None else f"\n  mix: {mix_info[i]}"
        except Exception:
            return ""

    @torch.no_grad()
    def _maybe_log_ctc_hyp(
        self, encoder_out, encoder_out_lens, text, text_lengths, mix_info=None
    ):
        """Print ``sample/ref/hyp/mix`` in the DAE-KWS sample-log format."""
        self._plain_fwd_count += 1
        if (
            not self.training
            or self.sample_log_interval <= 0
            or self._plain_fwd_count % self.sample_log_interval != 0
            or not self.token_list
        ):
            return
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return

        i = random.randrange(encoder_out.size(0))
        n = int(encoder_out_lens[i])
        hyp = torch.unique_consecutive(self.ctc.argmax(encoder_out[i : i + 1, :n])[0])
        hyp = [int(x) for x in hyp.tolist() if x != self.blank_id]
        ref = [
            int(x)
            for x in text[i, : int(text_lengths[i])].tolist()
            if int(x) != self.ignore_id
        ]
        logging.info(
            f"sample[{i}/{encoder_out.size(0)}] T={n} ref_len={len(ref)}\n"
            f"  ref: {self._ids2text(ref)}\n"
            f"  hyp: {self._ids2text(hyp)}"
            f"{self._fmt_mix_info(mix_info, i)}"
        )
