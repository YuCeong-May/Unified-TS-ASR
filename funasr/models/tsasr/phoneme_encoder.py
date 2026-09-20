# -*- coding: utf-8 -*-
"""Phoneme (keyword) encoder for text-cued target-speaker ASR.

Encode a phoneme-id sequence into contextual features that are later injected
into the mixture-speech stream via cross-attention.
"""
# =============================================================================
# Phoneme (keyword) encoder for text-cued target-speaker ASR.
#
# The architecture (token embedding -> input projection -> positional encoding
# -> stack of self-attention Transformer layers) faithfully follows the keyword
# branch of DAE-TSE / KCE `AEDKWSASRPhone` (kce/src/kce/model/AEDKWSASRPhone.py,
# `phn_emb` + `keyword_input_projection` + `keyword_pe_module` +
# `keyword_transformer`). Building blocks are reused from `dae_modules.py`.
#   repo   : https://github.com/GnafiY/DAE-TSE
#   branch : cursor/cloud-agent-1784704881858-cfrvr
# All credit to the original authors.
# =============================================================================
import torch
import torch.nn as nn

from funasr.register import tables
from funasr.models.tsasr.dae_modules import (
    WordEmbedding,
    PositionalEncoding,
    MultiHeadAtt,
    FNNBlock,
    TransformerLayer,
    make_mask,
)


@tables.register("phoneme_encoder_classes", "PhonemeTransformerEncoder")
class PhonemeTransformerEncoder(nn.Module):
    """Encode a phoneme-id sequence (keyword / wake-word) into contextual features.

    Args:
        num_phn_token: phoneme vocabulary size (index 0 reserved for padding).
        dim:           model dimension D (must match the fusion cross-attention dim).
        num_blocks:    number of self-attention Transformer layers.
        attention_heads: number of heads in each self-attention layer.
        ffn_hidden:    hidden size of the feed-forward blocks (default 4*dim).
        padding_idx:   embedding padding index (default 0).
    """

    def __init__(
        self,
        num_phn_token: int,
        dim: int = 256,
        num_blocks: int = 3,
        attention_heads: int = 4,
        ffn_hidden: int = None,
        padding_idx: int = 0,
        **kwargs,
    ):
        super().__init__()
        self._dim = dim
        ffn_hidden = ffn_hidden if ffn_hidden is not None else dim * 4

        self.phn_emb = WordEmbedding(num_phn_token, dim=dim, padding_idx=padding_idx)
        # input projection: D -> D (mirrors keyword_input_projection FNNBlock)
        self.input_projection = FNNBlock([dim, ffn_hidden, dim])
        self.pe_module = PositionalEncoding(dim)
        self.transformer = nn.ModuleList(
            [
                TransformerLayer(
                    size=dim,
                    self_att=MultiHeadAtt(n_head=attention_heads, n_feats=dim),
                    feed_forward=FNNBlock([dim, ffn_hidden, dim]),
                )
                for _ in range(num_blocks)
            ]
        )

    def output_size(self) -> int:
        """返回音素编码器的输出特征维度 D。

        Returns:
            int: 模型维度 D（与融合 cross-attention 的 dim 一致）。
        """
        return self._dim

    def forward(self, phone_ids: torch.Tensor, phone_lengths: torch.Tensor):
        """
        Args:
            phone_ids:     (B, L) long tensor of phoneme ids (0 = pad).
            phone_lengths: (B,) valid lengths.
        Returns:
            kw_feat:  (B, L, D) contextual phoneme features.
            kw_valid: (B, L) bool mask, True at valid positions.
        """
        phone_ids = phone_ids.to(torch.long)
        # valid mask: True where valid (~padding)
        kw_valid = ~make_mask(phone_lengths.view(-1), max_len=phone_ids.size(1))
        attn_mask = kw_valid.unsqueeze(1)  # (B, 1, L) -> used as attention mask

        emb = self.phn_emb(phone_ids)              # (B, L, D)
        emb = self.input_projection(emb)           # (B, L, D)
        emb = self.pe_module(emb)                  # (B, L, D)

        hidden = emb
        for layer in self.transformer:
            hidden, _ = layer(hidden, attn_mask)
        return hidden, kw_valid
