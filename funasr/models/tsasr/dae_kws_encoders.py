# -*- coding: utf-8 -*-
"""实验4（V4）：DAE-TSE / KCE `AEDKWSASRPhone` 的忠实复刻——声学与关键词两个分支。

与实验3（`kce_encoder.SANMEncoderKCE`）的本质区别：V3 是把 cross-attention adapter
嫁接进**预训练的 50 层 512 维 Paraformer SANMEncoder**；这里是 DAE-TSE 自己的小模型，
9 层 256 维 Conformer（**每一层**都做 cross-attention）+ 4 层关键词 Transformer，
**从零训练**，不加载任何预训练权重。

两个注册类：
- `ConformerKceEncoderDae`  -> `encoder_classes`，声学分支（Conv2d 前端 + 9 层）。
- `KeywordTransformerEncoderDae` -> `phoneme_encoder_classes`，关键词分支（4 层）。

为什么不复用已有的 `PhonemeTransformerEncoder`：它硬编码
`FNNBlock([dim, ffn_hidden, dim])` 作 input projection（256->2048->256），而 DAE-TSE 的
`kw_input_trans_config.dim: 256` 是 `FNNBlock(256)` = 256->256->256，且 `ffn_hidden`
同时驱动 block 内 FFN，改配置改不出来。

两处刻意保留的「反常」：
- 9 层之后**没有** after_norm。FunASR 所有 encoder 在 `normalize_before` 时都会收尾一个
  LayerNorm，而 `TransformerLayer` 是 pre-norm，所以这里的输出是未归一化的。忠实照抄。
- `kw_au_link` 在原实现里被构造但从未调用（两边都 256 维，本就是 Identity），这里同样
  构造为 `nn.Identity()` 且不调用，只为保持 state_dict 结构可比。

唯一的功能性偏离（开关 `mask_pad_before_conv`，默认 True）：`DepthWiseConv.forward`
不收 mask，kernel 15 会把 ±7 帧 padding 泄进有效帧、`BatchNorm1d` 也把 padding 计入
统计。DAE-TSE 用固定 `batch_size: 48` 条，padding 比例稳定；本项目是 token 预算动态
batching，padding 比例波动大得多，故每层 conv 前把 padding 位显式置零。
"""
# =============================================================================
# Adapted from the DAE-TSE / KCE project:
#   repo   : https://github.com/GnafiY/DAE-TSE
#   branch : cursor/cloud-agent-1784704881858-cfrvr
#   source : kce/src/kce/model/AEDKWSASRPhone.py
# All credit to the original authors.
# =============================================================================
import torch
import torch.nn as nn

from funasr.register import tables
from funasr.models.tsasr.dae_extra import compute_dim_reduction
from funasr.models.tsasr.dae_modules import (
    DepthWiseConv,
    FNNBlock,
    MultiHeadAtt,
    MultiHeadCrossAtt,
    PositionalEncoding,
    TransformerLayer,
    WordEmbedding,
    combine_mask,
    make_mask,
)

@tables.register("phoneme_encoder_classes", "KeywordTransformerEncoderDae")
class KeywordTransformerEncoderDae(nn.Module):
    """关键词分支：token embedding -> input projection -> PE -> N 层自注意力。

    忠实复刻 `AEDKWSASRPhone` 的 `phn_emb` + `keyword_input_projection` +
    `keyword_pe_module` + `keyword_transformer`。V4 把词表从英文音素（74）换成
    ASR 的中文 char 词表（8404）——这是本实验相对 DAE-TSE 的**唯一实质改动**。

    Args:
        num_token: 词表大小（id 0 是 `<blank>`，真实文本不会出现，正好当 padding_idx）。
        dim: 模型维度 D，须与声学分支的 `au_hidden_dim` 一致（否则 cross-att 维度不匹配）。
        num_blocks: 自注意力层数（DAE-TSE `num_kw_block: 4`）。
        input_trans_dim: input projection 的 FNNBlock dim（DAE-TSE 是 int 256）。
        self_att_heads: 自注意力头数（DAE-TSE `n_head: 4`）。
        ffn_dim: block 内 FFN 的 FNNBlock dim（DAE-TSE `[256, 2048, 256]`）。
        padding_idx: embedding 的 padding 下标。
    """

    def __init__(
        self,
        num_token: int,
        dim: int = 256,
        num_blocks: int = 4,
        input_trans_dim=256,
        self_att_heads: int = 4,
        ffn_dim=None,
        padding_idx: int = 0,
        **kwargs,
    ):
        super().__init__()
        self._dim = dim
        ffn_dim = ffn_dim if ffn_dim is not None else [dim, 4 * dim, dim]

        self.phn_emb = WordEmbedding(num_token, dim=dim, padding_idx=padding_idx)
        self.keyword_input_projection = FNNBlock(input_trans_dim)
        self.keyword_pe_module = PositionalEncoding(dim)
        self.keyword_transformer = nn.ModuleList(
            [
                TransformerLayer(
                    size=dim,
                    self_att=MultiHeadAtt(n_head=self_att_heads, n_feats=dim),
                    feed_forward=FNNBlock(list(ffn_dim)),
                )
                for _ in range(num_blocks)
            ]
        )

    def output_size(self) -> int:
        """返回关键词编码器的输出维度 D。

        Returns:
            int: 模型维度 D。
        """
        return self._dim

    def forward(self, token_ids: torch.Tensor, token_lengths: torch.Tensor):
        """把关键词 token 序列编码成上下文特征。

        Args:
            token_ids: (B, L) long，char id，0 为 padding。
            token_lengths: (B,) 有效长度。

        Returns:
            (kw_feat, kw_valid)：kw_feat 形状 (B, L, D)；kw_valid 形状 (B, L) bool，
            True 表示有效位置。
        """
        token_ids = token_ids.to(torch.long)
        kw_valid = ~make_mask(token_lengths.view(-1), max_len=token_ids.size(1))
        attn_mask = kw_valid.unsqueeze(1)  # (B, 1, L)，True=有效（注意力内部再 eq(0)）

        hidden = self.phn_emb(token_ids)
        hidden = self.keyword_input_projection(hidden)
        hidden = self.keyword_pe_module(hidden)
        for layer in self.keyword_transformer:
            hidden, _ = layer(hidden, attn_mask)
        return hidden, kw_valid

@tables.register("encoder_classes", "ConformerKceEncoderDae")
class ConformerKceEncoderDae(nn.Module):
    """声学分支：Conv2d x2 前端 + N 层 Conformer，**每层**都 cross-attend 关键词。

    Args:
        input_size: 输入特征维度，必须等于 `num_mel_bins`（`build_model` 会用
            `frontend.output_size()` 覆盖顶层 `input_size`；本实验 frontend 无 LFR，
            所以是 80）。
        au_hidden_dim: 模型维度 D（DAE-TSE `d_model: 256`）。
        num_mel_bins: fbank 维度，决定 `au_conv_trans` 的输入宽度。
        num_blocks: Conformer 层数（DAE-TSE `num_audio_block: 9`）。
        input_trans_dim: `speech_input_projection` 的 FNNBlock dim（DAE-TSE `[256, 256]`）。
        self_att_heads: 自注意力头数（DAE-TSE 4）。
        cross_att_heads: 交叉注意力头数（DAE-TSE 1）。
        cross_att_norm: 交叉注意力 q/k/v 前置归一化类型（DAE-TSE `LayerNorm`）。
        ffn_dim: FFN 与 macaron FFN 的 FNNBlock dim（DAE-TSE `[256, 2048, 256]`）。
        conv_channels / conv_kernel_size / conv_norm: `DepthWiseConv` 配置
            （DAE-TSE 256 / 15 / batch_norm）。
        mask_pad_before_conv: 每层 conv 前把 padding 帧置零（相对原实现的唯一偏离）。
    """

    def __init__(
        self,
        input_size: int = 80,
        au_hidden_dim: int = 256,
        num_mel_bins: int = 80,
        num_blocks: int = 9,
        input_trans_dim=None,
        self_att_heads: int = 4,
        cross_att_heads: int = 1,
        cross_att_norm: str = "LayerNorm",
        ffn_dim=None,
        conv_channels: int = 256,
        conv_kernel_size: int = 15,
        conv_norm: str = "batch_norm",
        mask_pad_before_conv: bool = True,
        **kwargs,
    ):
        super().__init__()
        if input_size != num_mel_bins:
            raise ValueError(
                f"ConformerKceEncoderDae: input_size({input_size}) != "
                f"num_mel_bins({num_mel_bins})。frontend 必须是无 LFR 的 80 维 fbank"
                f"（lfr_m=1, lfr_n=1），否则 au_conv_trans 的输入宽度算错。"
            )
        self._output_size = au_hidden_dim
        self.num_mel_bins = num_mel_bins
        self.mask_pad_before_conv = mask_pad_before_conv
        input_trans_dim = input_trans_dim if input_trans_dim is not None else [au_hidden_dim] * 2
        ffn_dim = ffn_dim if ffn_dim is not None else [au_hidden_dim, 4 * au_hidden_dim, au_hidden_dim]

        # 两次 stride-2 / kernel-3 / **无 padding** 的 Conv2d -> 时间 4 倍下采样
        self.au_conv = nn.Sequential(
            nn.Conv2d(1, au_hidden_dim, 3, 2),
            nn.ReLU(),
            nn.Conv2d(au_hidden_dim, au_hidden_dim, 3, 2),
            nn.ReLU(),
        )
        # 频率维同样被两次下采样：80 -> ((80-1)//2 - 1)//2 = 19
        self.freq_dim = ((self.num_mel_bins - 1) // 2 - 1) // 2
        self.au_conv_trans = nn.Linear(au_hidden_dim * self.freq_dim, au_hidden_dim)
        self.speech_input_projection = FNNBlock(list(input_trans_dim))
        self.speech_pe_module = PositionalEncoding(au_hidden_dim)
        self.speech_transformer = nn.ModuleList(
            [
                TransformerLayer(
                    size=au_hidden_dim,
                    self_att=MultiHeadAtt(n_head=self_att_heads, n_feats=au_hidden_dim),
                    cross_att=MultiHeadCrossAtt(
                        n_head=cross_att_heads, n_feats=au_hidden_dim, norm=cross_att_norm
                    ),
                    feed_forward=FNNBlock(list(ffn_dim)),
                    macaron_layer=FNNBlock(list(ffn_dim)),
                    conv_layer=DepthWiseConv(
                        channels=conv_channels,
                        kernel_size=conv_kernel_size,
                        norm=conv_norm,
                    ),
                )
                for _ in range(num_blocks)
            ]
        )
        # 原实现构造了但从未调用（两边同维，本就是 Identity）。忠实保留、不调用。
        self.kw_au_link = nn.Identity()
        # 原实现在 9 层之后没有 after_norm，这里同样不加（见模块 docstring）。

    def output_size(self) -> int:
        """返回声学编码器输出维度 D（= `au_hidden_dim`）。

        Returns:
            int: 模型维度 D。
        """
        return self._output_size

    def subsampled_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        """把输入帧长换算成 `au_conv` 之后的帧长（两次 kernel3/stride2/pad0）。

        Args:
            lengths: (B,) 输入帧长。

        Returns:
            (B,) 下采样后的帧长。
        """
        lengths = compute_dim_reduction(lengths, 3, 2, 0, 1)
        lengths = compute_dim_reduction(lengths, 3, 2, 0, 1)
        return lengths

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        cue_feat: torch.Tensor = None,
        cue_valid: torch.Tensor = None,
        return_hidden: bool = False,
        **kwargs,
    ):
        """逐行转录 `AEDKWSASRPhone.forward` 的声学分支。

        mask 约定：`make_mask` True=padding，`combine_mask` True=无效，而
        `MultiHeadAtt` / `MultiHeadCrossAtt` 内部会 `mask.eq(0)`，所以传进注意力的
        mask 必须 **True=有效**。

        `cue_feat` 为 None 时（无线索前向，仅评测对照用）退化成「长度 1 的全零线索」
        且 mask 全有效——注意这与训练分布不一致，因为本模型每层都注入线索。

        Args:
            speech: (B, T, num_mel_bins) fbank。
            speech_lengths: (B,) 有效帧长。
            cue_feat: (B, L, D) 关键词分支输出。
            cue_valid: (B, L) bool，True 表示有效。
            return_hidden: 额外把**每层**的输出收进 `extra["hidden_states"]`（长度
                `num_blocks` 的 list，各元素 (B, T', D)）。给说话人分类辅助头
                （`spk_ce_head.SpeakerCEHead`）取中间层用。不额外占激活显存：
                autograd 图本来就把这些张量留着，list 只是多存一份引用。

        Returns:
            (encoder_out, encoder_out_lens, extra)：encoder_out 形状 (B, T', D)；
            extra 是 `{"cross_att": [每层 (B, n_head, T', L) 的注意力权重]}`，
            `return_hidden` 时另含 `"hidden_states"`。
        """
        speech_lengths = speech_lengths.view(-1)
        olens = self.subsampled_lengths(speech_lengths)

        x = self.au_conv(speech.unsqueeze(1))  # (B, 1, T, F) -> (B, D, T', F')
        b, c, t, d = x.size()
        x = self.au_conv_trans(x.transpose(1, 2).contiguous().view(b, t, c * d))
        x = self.speech_input_projection(x)
        x = self.speech_pe_module(x)

        # olens 由长度公式算出，t 由 padded T 算出，两者一致；不一致说明前端配置不匹配
        olens = torch.clamp(olens, max=t)
        sph_mask = (~make_mask(olens, max_len=t)).unsqueeze(1)  # (B, 1, T') True=有效

        if cue_feat is None:
            cue_feat = x.new_zeros(b, 1, x.size(-1))
            cue_valid = torch.ones(b, 1, dtype=torch.bool, device=x.device)
        kw_mask = cue_valid.unsqueeze(1)  # (B, 1, L) True=有效
        cross_mask = ~combine_mask(sph_mask.squeeze(1), kw_mask.squeeze(1), 1)

        pad_zero_mask = ~sph_mask.transpose(1, 2)  # (B, T', 1) True=padding
        att_scores, hidden_states = [], []
        for layer in self.speech_transformer:
            if self.mask_pad_before_conv:
                x = x.masked_fill(pad_zero_mask, 0.0)
            x, cross_score = layer(x, sph_mask, cross_input=(cue_feat, cue_feat, cross_mask))
            att_scores.append(cross_score)
            if return_hidden:
                hidden_states.append(x)

        extra = {"cross_att": att_scores}
        if return_hidden:
            extra["hidden_states"] = hidden_states
        return x, olens, extra
