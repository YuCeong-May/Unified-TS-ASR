# -*- coding: utf-8 -*-
"""Phoneme <-> mixture-speech interaction (cross-attention) for text-cued
target-speaker ASR."""
# =============================================================================
# Phoneme <-> mixture-speech interaction (cross-attention) for text-cued
# target-speaker ASR.
#
# Faithfully follows the speech branch of DAE-TSE / KCE `AEDKWSASRPhone`
# (kce/src/kce/model/AEDKWSASRPhone.py, `speech_transformer` run with
# `cross_hidden_states=(keyword_feature, keyword_feature, cross_mask)`):
# the mixture-speech features act as query and attend to the phoneme features
# (key/value) through a stack of Conformer-style Transformer layers
# (self-att + cross-att + macaron FFN + depth-wise conv). Building blocks are
# reused from `dae_modules.py`.
#   repo   : https://github.com/GnafiY/DAE-TSE
#   branch : cursor/cloud-agent-1784704881858-cfrvr
# All credit to the original authors.
#
# Unlike the speaker-embedding fusions in `fusion.py` (which take a single
# (B, D) vector), this module consumes a phoneme *sequence* (B, L, D) plus its
# mask, so it deliberately uses its own forward signature.
# =============================================================================
import torch
import torch.nn as nn

from funasr.register import tables
from funasr.models.tsasr.dae_modules import (
    PositionalEncoding,
    MultiHeadAtt,
    MultiHeadCrossAtt,
    FNNBlock,
    DepthWiseConv,
    TransformerLayer,
    make_mask,
    combine_mask,
)


@tables.register("fusion_classes", "PhonemeCrossAttn")
class PhonemeCrossAttnFusion(nn.Module):
    """Inject phoneme (keyword) features into mixture-speech features.

    Args:
        feats_dim:   acoustic feature dim from the frontend (Paraformer LFR = 560).
        d_model:     internal model dim D (must match the phoneme encoder dim).
        num_blocks:  number of Conformer-style interaction layers M.
        attention_heads: heads for self- and cross-attention.
        ffn_hidden:  hidden size of the feed-forward / macaron blocks (default 4*D).
        conv_kernel: depth-wise conv kernel size (odd).
        cross_att_norm: norm used inside MultiHeadCrossAtt q/k/v (default LayerNorm).

    Output dim equals feats_dim so the downstream encoder input_size is unchanged.
    """

    def __init__(
        self,
        feats_dim: int = 560,
        d_model: int = 256,
        num_blocks: int = 6,
        attention_heads: int = 4,
        ffn_hidden: int = None,
        conv_kernel: int = 15,
        cross_att_norm: str = "LayerNorm",
        **kwargs,
    ):
        super().__init__()
        self._feats_dim = feats_dim
        ffn_hidden = ffn_hidden if ffn_hidden is not None else d_model * 4

        self.proj_in = nn.Linear(feats_dim, d_model)
        self.pe_module = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    size=d_model,
                    self_att=MultiHeadAtt(n_head=attention_heads, n_feats=d_model),
                    cross_att=MultiHeadCrossAtt(
                        n_head=attention_heads, n_feats=d_model, norm=cross_att_norm
                    ),
                    feed_forward=FNNBlock([d_model, ffn_hidden, d_model]),
                    macaron_layer=FNNBlock([d_model, ffn_hidden, d_model]),
                    conv_layer=DepthWiseConv(channels=d_model, kernel_size=conv_kernel),
                )
                for _ in range(num_blocks)
            ]
        )
        self.proj_out = nn.Linear(d_model, feats_dim)
        # 零初始化出口 + 残差：微调起点与预训练 Paraformer 逐位等价（DAE 原实现直接
        # 用 proj_out 输出替换声学特征，那是从头训练的场景；这里主干是预训练好的，
        # 随机初始化的 6 层交互会把 CMVN 后的 fbank 换成噪声，实测 step0 acc
        # 从 0.908 掉到 0.092）。音素线索的贡献从 0 开始被学出来。
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def output_size(self) -> int:
        """返回融合模块的输出特征维度（等于 feats_dim）。

        Returns:
            int: 输出维度，保持与输入声学特征维度一致。
        """
        return self._feats_dim

    def forward(
        self,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        kw_feat: torch.Tensor,
        kw_valid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            feats:         (B, T, feats_dim) mixture-speech features.
            feats_lengths: (B,) valid frame counts.
            kw_feat:       (B, L, D) phoneme features from the phoneme encoder.
            kw_valid:      (B, L) bool mask, True at valid phoneme positions.
        Returns:
            (B, T, feats_dim) phoneme-conditioned features (dim preserved).
            残差形式：`feats + proj_out(...)`，proj_out 零初始化 -> 训练起点等于
            不注入时的 feats，随训练逐步注入音素线索。
        """
        # speech valid mask (True = valid), mirrors DAE `sph_mask = ~make_mask(...)`
        sph_valid = ~make_mask(feats_lengths.view(-1), max_len=feats.size(1))  # (B, T)
        sph_attn_mask = sph_valid.unsqueeze(1)  # (B, 1, T) for self-attention
        # cross mask: True = valid pair, mirrors DAE `~combine_mask(sph, kw, 1)`
        cross_mask = ~combine_mask(sph_valid, kw_valid, 1)  # (B, 1, T, L)

        hidden = self.proj_in(feats)          # (B, T, D)
        hidden = self.pe_module(hidden)       # (B, T, D)

        cross_input = (kw_feat, kw_feat, cross_mask)
        for layer in self.layers:
            hidden, _ = layer(hidden, sph_attn_mask, cross_input=cross_input)

        out = feats + self.proj_out(hidden)   # (B, T, feats_dim)
        return out
