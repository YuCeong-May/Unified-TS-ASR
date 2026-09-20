# -*- coding: utf-8 -*-
"""DAE-TSE / KCE 的通用网络构建块（词嵌入、位置编码、注意力、卷积、Transformer 层等）。

这些模块被 tsasr 的音素编码器与融合模块复用。
"""
# =============================================================================
# Adapted from the DAE-TSE / KCE project:
#   repo   : https://github.com/GnafiY/DAE-TSE
#   branch : cursor/cloud-agent-1784704881858-cfrvr
#   source : kce/src/kce/model/NetModules.py
# The building blocks below (WordEmbedding / PositionalEncoding / MultiHeadAtt /
# MultiHeadCrossAtt / FNNBlock / DepthWiseConv / TransformerLayer / make_mask /
# combine_mask) are reproduced almost verbatim from that file, which itself is
# derived from WeNet (https://github.com/wenet-e2e/wenet). All credit to the
# original authors. Only unused code paths and imports were trimmed so this file
# is self-contained inside FunASR; the module logic is unchanged.
# =============================================================================
import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

norm_dict = {
    "LayerNorm": nn.LayerNorm,
    "BatchNorm1d": nn.BatchNorm1d,
    "BatchNorm2d": nn.BatchNorm2d,
    "BatchNorm3d": nn.BatchNorm3d,
    "InstanceNorm1d": nn.InstanceNorm1d,
    "InstanceNorm2d": nn.InstanceNorm2d,
    "InstanceNorm3d": nn.InstanceNorm3d,
}

act_dict = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "Sigmoid": nn.Sigmoid,
    "Tanh": nn.Tanh,
}


class WordEmbedding(nn.Module):
    """词嵌入层：把 token id 映射为稠密向量。"""

    def __init__(self, num_tokens, dim, padding_idx=0):
        super(WordEmbedding, self).__init__()
        self.emb = nn.Embedding(num_tokens, dim, padding_idx=padding_idx)

    def forward(self, input):
        """查表得到输入 token 的嵌入向量。

        Args:
            input: (..., L) token id 张量。

        Returns:
            (..., L, dim) 嵌入向量。
        """
        return self.emb(input)


class PositionalEncoding(nn.Module):
    """正弦位置编码：为输入特征叠加位置信息。"""

    def __init__(self, model_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = model_dim
        self.xcale = math.sqrt(self.d_model)
        self.max_len = max_len

        self.pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        self.pe[0:, 0::2] = torch.sin(position * div_term)
        self.pe[0:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def position_encoding(self, size, device):
        """生成前 size 个位置的位置编码。

        Args:
            size: 序列长度。
            device: 目标设备。

        Returns:
            (size, model_dim) 位置编码张量。
        """
        index = torch.arange(0, size).to(device)
        pos_emb = F.embedding(index, self.pe[0])
        return pos_emb

    def forward(self, input):
        """对输入叠加缩放与位置编码。

        Args:
            input: (B, T, model_dim) 特征。

        Returns:
            (B, T, model_dim) 加位置编码后的特征。
        """
        self.pe = self.pe.to(input.device)
        pos_emb = self.position_encoding(input.size(1), input.device)
        input = input * self.xcale + pos_emb
        return input


class MultiHeadAtt(nn.Module):
    """多头自注意力模块。"""

    def __init__(self, n_head, n_feats):
        super(MultiHeadAtt, self).__init__()
        assert n_feats % n_head == 0
        self.d_k = n_feats // n_head
        self.n_head = n_head
        self.q = nn.Linear(n_feats, n_feats)
        self.k = nn.Linear(n_feats, n_feats)
        self.v = nn.Linear(n_feats, n_feats)
        self.linear_out = nn.Linear(n_feats, n_feats)

    def forward_qkv(self, q, k, v):
        """线性投影并把 q/k/v 切分为多头。

        Args:
            q: (B, T, n_feats) query。
            k: (B, T, n_feats) key。
            v: (B, T, n_feats) value。

        Returns:
            (q, k, v)，各形状 (B, n_head, T, d_k)。
        """
        batch = q.size(0)
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)
        q = q.view(batch, -1, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(batch, -1, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(batch, -1, self.n_head, self.d_k).transpose(1, 2)
        return q, k, v

    def forward_selfatt(self, q, k, v, mask, softmax=True):
        """多头缩放点积自注意力。

        Args:
            q: (B, n_head, T, d_k) query。
            k: (B, n_head, T, d_k) key。
            v: (B, n_head, T, d_k) value。
            mask: 注意力 mask，None 表示不做 mask。
            softmax: 为 False 时返回原始分数而非归一化权重。

        Returns:
            (context, score)：context 形状 (B, T, n_feats)；score 为权重或原始分数。
        """
        batch, nhead, tq, d = q.size()
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
        if mask is not None:
            if len(mask.size()) < len(q.size()):
                mask = mask.unsqueeze(1).eq(0)
            else:
                mask = mask.eq(0)
        if mask is not None:
            score = score.masked_fill(mask, -float("inf"))
            att_weight = torch.softmax(score, dim=-1).masked_fill(mask, 0.0)
        else:
            att_weight = torch.softmax(score, dim=-1)
        context = torch.matmul(att_weight, v)
        context = context.transpose(1, 2).contiguous().view(batch, tq, -1)
        return context, att_weight if softmax else score

    def forward(self, q, k, v, mask, softmax=True):
        """完整多头注意力：qkv 投影 -> 自注意力 -> 输出投影。

        Args:
            q: (B, T, n_feats) query。
            k: (B, T, n_feats) key。
            v: (B, T, n_feats) value。
            mask: 注意力 mask，None 表示不做 mask。
            softmax: 为 False 时返回原始分数而非归一化权重。

        Returns:
            (context, score)。
        """
        q, k, v = self.forward_qkv(q, k, v)
        context, score = self.forward_selfatt(q, k, v, mask)
        return self.linear_out(context), score


class MultiHeadCrossAtt(nn.Module):
    """多头交叉注意力模块（q/k/v 先 norm 再投影）。"""

    def __init__(self, n_head, n_feats, norm=None):
        super(MultiHeadCrossAtt, self).__init__()
        self.q = nn.Sequential(norm_dict[norm](n_feats), nn.Linear(n_feats, n_feats))
        self.k = nn.Sequential(norm_dict[norm](n_feats), nn.Linear(n_feats, n_feats))
        self.v = nn.Sequential(norm_dict[norm](n_feats), nn.Linear(n_feats, n_feats))
        self.linear_out = nn.Linear(n_feats, n_feats)
        self.n_head = n_head
        self.n_feats = n_feats

    def forward_selfatt(self, q, k, v, mask, aux_score=None, softmax=True, print_mask=False):
        """交叉注意力打分（支持 aux_score 缩放）。

        Args:
            q: (B, n_head, Tq, d_k) query。
            k: (B, n_head, Tk, d_k) key。
            v: (B, n_head, Tv, d_k) value。
            mask: 注意力 mask，None 表示不做 mask。
            aux_score: 额外的逐位置缩放分数，None 时不生效。
            softmax: 为 False 时返回原始分数而非归一化权重。
            print_mask: 预留调试参数。

        Returns:
            (context, score)。
        """
        batch, nhead, tq, d = q.size()
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
        if mask is not None:
            if len(mask.size()) < len(q.size()):
                mask = mask.unsqueeze(1).eq(0)
            else:
                mask = mask.eq(0)
        if aux_score is not None:
            score = score * aux_score[:, None, None, :]
        if mask is not None:
            score = score.masked_fill(mask, -float("inf"))
            att_weight = torch.softmax(score, dim=-1).masked_fill(mask, 0.0)
        else:
            att_weight = torch.softmax(score, dim=-1)
        context = torch.matmul(att_weight, v)
        context = context.transpose(1, 2).contiguous().view(batch, tq, -1)
        return context, att_weight if softmax else score

    def forward(self, q, k, v, mask=None, aux_score=None, softmax=True, print_mask=False):
        """完整交叉注意力：norm+投影 -> 注意力 -> 输出投影。

        Args:
            q: (B, Tq, n_feats) query。
            k: (B, Tk, n_feats) key。
            v: (B, Tv, n_feats) value。
            mask: 注意力 mask，None 表示不做 mask。
            aux_score: 额外的逐位置缩放分数，None 时不生效。
            softmax: 为 False 时返回原始分数而非归一化权重。
            print_mask: 预留调试参数。

        Returns:
            (context, score)。
        """
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)
        batch = k.size(0)
        tq = q.size(1)
        tk = k.size(1)
        tv = v.size(1)
        q = q.view(batch, tq, self.n_head, -1).transpose(1, 2)
        k = k.view(batch, tk, self.n_head, -1).transpose(1, 2)
        v = v.view(batch, tv, self.n_head, -1).transpose(1, 2)
        context, score = self.forward_selfatt(
            q, k, v, mask, aux_score=aux_score, softmax=softmax, print_mask=print_mask
        )
        return self.linear_out(context), score


class DepthWiseConv(nn.Module):
    """深度可分离卷积层（pointwise -> GLU -> depthwise -> norm/激活 -> pointwise）。"""

    def __init__(
        self,
        channels,
        kernel_size=15,
        activation=None,
        norm="batch_norm",
        causal=False,
        bias=True,
    ):
        super().__init__()
        if activation is None:
            activation = nn.ReLU()
        self.pointwise_conv1 = nn.Conv1d(
            channels, 2 * channels, kernel_size=1, stride=1, padding=0, bias=bias
        )
        if causal:  # keep this as false
            padding = 0
            self.lorder = kernel_size - 1
        else:
            # kernel_size should be an odd number for none causal convolution
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0
        self.depthwise_conv = nn.Conv1d(
            channels, channels, kernel_size, stride=1, padding=padding,
            groups=channels, bias=bias,
        )

        assert norm in ["batch_norm", "layer_norm"]
        if norm == "batch_norm":
            self.use_layer_norm = False
            self.norm = nn.BatchNorm1d(channels)
        else:
            self.use_layer_norm = True
            self.norm = nn.LayerNorm(channels)

        self.pointwise_conv2 = nn.Conv1d(
            channels, channels, kernel_size=1, stride=1, padding=0, bias=bias
        )
        self.activation = activation

    def forward(self, input):
        """前向：GLU -> 深度卷积 -> 归一化激活 -> 输出投影。

        Args:
            input: (B, T, channels) 特征。

        Returns:
            (B, T, channels) 卷积输出。
        """
        # exchange the temporal dimension and the feature dimension
        input = input.transpose(1, 2)  # (#batch, channels, time)
        # GLU mechanism
        input = self.pointwise_conv1(input)  # (batch, 2*channel, dim)
        input = nn.functional.glu(input, dim=1)  # (batch, channel, dim)
        # 1D Depthwise Conv
        input = self.depthwise_conv(input)
        if self.use_layer_norm:
            input = input.transpose(1, 2)
        input = self.activation(self.norm(input))
        if self.use_layer_norm:
            input = input.transpose(1, 2)
        input = self.pointwise_conv2(input)
        return input.transpose(1, 2)


class FNNBlock(nn.Module):
    """前馈网络块（Linear -> ReLU -> Linear）。"""

    def __init__(self, dim, num_block=1, bias=True, norm=None, act="ReLU"):
        super(FNNBlock, self).__init__()
        if isinstance(dim, list):
            assert len(dim) <= 3
            if len(dim) == 2:
                idim, odim = dim
                hdim = None
            else:
                idim, hdim, odim = dim
        else:
            assert isinstance(dim, int)
            idim = hdim = odim = dim

        if norm is not None:
            assert norm in norm_dict
        if act is not None:
            assert act in act_dict

        self.w1 = nn.Linear(idim, hdim if hdim else idim)
        self.act = nn.ReLU()
        self.w2 = nn.Linear(hdim if hdim else idim, odim)

    def forward(self, input):
        """前向：w2(act(w1(input)))。

        Args:
            input: (..., idim) 输入。

        Returns:
            (..., odim) 输出。
        """
        return self.w2(self.act(self.w1(input)))


class TransformerLayer(nn.Module):
    """Conformer 风格 Transformer 层（自注意力/交叉注意力/卷积/前馈）。

    可选 macaron FFN、depthwise conv、cross-attention 与 decoder-attention。
    """

    def __init__(
        self,
        self_att,
        feed_forward,
        macaron_layer=None,
        conv_layer=None,
        cross_att=None,
        decoder_att=None,
        size=256,
    ):
        super(TransformerLayer, self).__init__()
        self.self_att = self_att
        self.feed_forward = feed_forward
        self.macaron_layer = macaron_layer
        self.cross_att = cross_att
        self.decoder_att = decoder_att
        self.input_norm = nn.LayerNorm(size, eps=1e-5)
        self.fnn_norm = nn.LayerNorm(size, eps=1e-5)
        self.size = size
        self.conv_layer = conv_layer
        if self.conv_layer is not None:
            self.conv_norm = nn.LayerNorm(size, eps=1e-5)
        if self.macaron_layer is not None:
            self.macaron_norm = nn.LayerNorm(size, eps=1e-5)
            self.macaron_factor = 0.5

    def forward(self, input, mask, cross_input=None, aux_score=None, args=None, print_mask=False):
        """前向：macaron FFN -> 自注意力 -> 交叉/解码器注意力 -> 卷积 -> 前馈。

        Args:
            input: (B, T, size) 输入特征。
            mask: 自注意力 mask。
            cross_input: (k, v, cross_mask) 或 (memory, memory, memory_mask)。
            aux_score: 额外的逐位置缩放分数，None 时不生效。
            args: 传给 feed_forward 的额外关键字参数。
            print_mask: 预留调试参数。

        Returns:
            (output, score)：output 形状 (B, T, size)；score 为最后使用的注意力分数。
        """
        if self.macaron_layer is not None:
            residual = input
            input = self.macaron_norm(input)
            input = residual + self.macaron_factor * self.macaron_layer(input)

        residual = input
        input = self.input_norm(input)

        context, att_score = self.self_att(input, input, input, mask)
        input = residual + context
        residual = input

        if self.cross_att is not None:
            k, v, cross_mask = cross_input
            cross_context, cross_score = self.cross_att(input, k, v, cross_mask, aux_score)
            input = cross_context + residual
            residual = input

        if self.decoder_att is not None:
            memory, memory, memory_mask = cross_input
            decoder_context, decoder_score = self.decoder_att(
                input, memory, memory, memory_mask, print_mask=print_mask
            )
            input = decoder_context + residual
            residual = input

        if self.conv_layer is not None:
            conv_out = self.conv_norm(input)
            conv_out = self.conv_layer(conv_out)
            input = conv_out + residual
            residual = input

        input = self.fnn_norm(input)
        if args is not None:
            fnn_output, act = self.feed_forward(input, **args)
        else:
            fnn_output = self.feed_forward(input)

        input = fnn_output + residual

        if args is not None:
            return input, act
        elif self.cross_att is not None:
            return input, cross_score
        else:
            return input, att_score


def make_mask(length, max_len=None):
    """按长度构造 padding mask（True 表示 padding 位置）。

    Args:
        length: (B,) 每段有效长度。
        max_len: 最大长度，缺省取 length 的最大值。

    Returns:
        (B, max_len) bool mask，True 表示无效（padding）位置。
    """
    assert isinstance(length, torch.Tensor)
    batch_size = length.size(0)
    max_len = max_len if max_len is not None else length.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=length.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)  # 2d
    seq_length_expand = length.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask  # 2d mask batch x time


def combine_mask(mask1, mask2, tidx):
    """把两个 mask 沿指定维对齐并合并成交叉注意力 mask。

    Args:
        mask1: (..., L1) bool mask。
        mask2: (..., L2) bool mask。
        tidx: 对齐合并所沿的维度。

    Returns:
        (1, 1, L1, L2) bool mask，True 表示无效（padding）位置。
    """
    t1 = mask1.size(tidx)
    t2 = mask2.size(tidx)
    mask1 = mask1.unsqueeze(1).repeat(1, t2, 1).transpose(-2, -1)
    mask2 = mask2.unsqueeze(1).repeat(1, t1, 1)
    cmask = mask1 & mask2
    cmask = ~cmask
    return cmask.unsqueeze(1)

