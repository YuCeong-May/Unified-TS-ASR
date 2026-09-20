"""可插拔的「说话人 embedding × 语音特征」融合模块。

通用接口 SpeakerFusion：
    forward(feats, feats_lengths, spk_emb) -> fused_feats
        feats     : (B, T, feats_dim)   语音特征（如 560 维 LFR-fbank）
        feats_lengths : (B,) 有效帧数（部分融合方式需要做 mask）
        spk_emb   : (B, spk_dim)         说话人 embedding（整句一条）
        返回      : (B, T, output_size)  融合后特征

约定：所有融合方式默认 output_size == feats_dim，
从而下游编码器 input_size 保持不变、现有结构零改动。

新增融合方式：继承 SpeakerFusion 并用 @tables.register("fusion_classes", "<Name>") 注册。
在 config 里通过 fusion / fusion_conf 切换。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from funasr.register import tables


class SpeakerFusion(nn.Module):
    """融合模块基类。"""

    def __init__(self, feats_dim: int, spk_dim: int, **kwargs):
        super().__init__()
        self.feats_dim = int(feats_dim)
        self.spk_dim = int(spk_dim)
        self._output_size = int(feats_dim)

    def output_size(self) -> int:
        """返回融合后特征的输出维度。

        Returns:
            int: 融合后特征维度，默认与输入 feats_dim 一致。
        """
        return self._output_size

    def forward(self, feats, feats_lengths, spk_emb):
        """融合说话人 embedding 与语音特征（子类实现）。

        Args:
            feats: (B, T, feats_dim) 语音特征。
            feats_lengths: (B,) 每段有效帧数。
            spk_emb: (B, spk_dim) 说话人 embedding（整句一条）。

        Returns:
            torch.Tensor: (B, T, output_size) 融合后特征。

        Raises:
            NotImplementedError: 基类未实现，需由子类覆盖。
        """
        raise NotImplementedError

    @staticmethod
    def _broadcast(spk_emb: torch.Tensor, t: int) -> torch.Tensor:
        """(B, D) -> (B, T, D)"""
        return spk_emb.unsqueeze(1).expand(-1, t, -1)


@tables.register("fusion_classes", "ADD")
class AddFusion(SpeakerFusion):
    """相加融合：将 spk_emb 投影到 feats_dim 后逐帧相加。

        fused = feats + proj(spk_emb)
    """

    def __init__(self, feats_dim: int, spk_dim: int, **kwargs):
        super().__init__(feats_dim, spk_dim)
        self.proj = nn.Linear(spk_dim, feats_dim)

    def forward(self, feats, feats_lengths, spk_emb):
        """相加融合：fused = feats + proj(spk_emb)。

        Args:
            feats: (B, T, feats_dim) 语音特征。
            feats_lengths: (B,) 每段有效帧数。
            spk_emb: (B, spk_dim) 说话人 embedding。

        Returns:
            (B, T, feats_dim) 融合后特征。
        """
        bias = self.proj(spk_emb)  # (B, feats_dim)
        return feats + bias.unsqueeze(1)


@tables.register("fusion_classes", "CAT")
class ConcatFusion(SpeakerFusion):
    """拼接融合：逐帧拼接 spk_emb，再线性映射回 feats_dim（保持下游维度不变）。

        fused = Linear([feats ; broadcast(spk_emb)])
    """

    def __init__(self, feats_dim: int, spk_dim: int, **kwargs):
        super().__init__(feats_dim, spk_dim)
        self.proj = nn.Linear(feats_dim + spk_dim, feats_dim)

    def forward(self, feats, feats_lengths, spk_emb):
        """拼接融合：fused = Linear([feats ; broadcast(spk_emb)])。

        Args:
            feats: (B, T, feats_dim) 语音特征。
            feats_lengths: (B,) 每段有效帧数。
            spk_emb: (B, spk_dim) 说话人 embedding。

        Returns:
            (B, T, feats_dim) 融合后特征。
        """
        t = feats.shape[1]
        spk = self._broadcast(spk_emb, t)  # (B, T, spk_dim)
        return self.proj(torch.cat([feats, spk], dim=-1))


@tables.register("fusion_classes", "FiLM")
class FiLMFusion(SpeakerFusion):
    """FiLM 融合：由 spk_emb 生成逐通道的缩放 gamma 与偏置 beta。

        fused = (1 + gamma) * feats + beta
    (用 1 + gamma 让初始接近恒等映射，训练更稳。)

    zero_init=True（默认）时把生成器最后一层权重/偏置置零，于是初始 gamma=beta=0、
    fused 与 feats **逐比特相同**。微调预训练 Paraformer 时这一点是必需的：随机初始化的
    注入层会直接破坏预训练编码器的输入分布（实测 acc 0.908 -> 0.092），置零后 step 0
    等于预训练基线，条件信号从 0 开始学（adapter / ControlNet 的做法）。
    代价：gen 最后一层为零时 spk_encoder 分支在第 1 步拿不到梯度，属预期现象。
    """

    def __init__(self, feats_dim: int, spk_dim: int, hidden: int = 0,
                 zero_init: bool = True, **kwargs):
        super().__init__(feats_dim, spk_dim)
        if hidden and hidden > 0:
            self.gen = nn.Sequential(
                nn.Linear(spk_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 2 * feats_dim),
            )
            last = self.gen[-1]
        else:
            self.gen = nn.Linear(spk_dim, 2 * feats_dim)
            last = self.gen
        if zero_init:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, feats, feats_lengths, spk_emb):
        """FiLM 融合：fused = (1 + gamma) * feats + beta。

        Args:
            feats: (B, T, feats_dim) 语音特征。
            feats_lengths: (B,) 每段有效帧数。
            spk_emb: (B, spk_dim) 说话人 embedding。

        Returns:
            (B, T, feats_dim) 融合后特征。
        """
        gamma, beta = self.gen(spk_emb).chunk(2, dim=-1)  # each (B, feats_dim)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return (1.0 + gamma) * feats + beta


@tables.register("fusion_classes", "CLN")
class CLNFusion(SpeakerFusion):
    """Conditional Layer Norm：先对 feats 做无仿射 LayerNorm，
    再用 spk_emb 生成的 gamma/beta 做条件仿射。

        fused = (1 + gamma) * LayerNorm(feats) + beta
    """

    def __init__(self, feats_dim: int, spk_dim: int, eps: float = 1e-5, **kwargs):
        super().__init__(feats_dim, spk_dim)
        self.norm = nn.LayerNorm(feats_dim, eps=eps, elementwise_affine=False)
        self.to_gamma = nn.Linear(spk_dim, feats_dim)
        self.to_beta = nn.Linear(spk_dim, feats_dim)

    def forward(self, feats, feats_lengths, spk_emb):
        """条件 LayerNorm 融合：fused = (1 + gamma) * LayerNorm(feats) + beta。

        Args:
            feats: (B, T, feats_dim) 语音特征。
            feats_lengths: (B,) 每段有效帧数。
            spk_emb: (B, spk_dim) 说话人 embedding。

        Returns:
            (B, T, feats_dim) 融合后特征。
        """
        normed = self.norm(feats)
        gamma = self.to_gamma(spk_emb).unsqueeze(1)
        beta = self.to_beta(spk_emb).unsqueeze(1)
        return (1.0 + gamma) * normed + beta


@tables.register("fusion_classes", "CrossAttn")
class CrossAttnFusion(SpeakerFusion):
    """交叉注意力融合：以语音帧为 query，说话人 embedding 为 key/value。

    可将单条 spk_emb 展开成 num_tokens 个可学习偏置的 token 作为 K/V，
    输出经残差加回 feats（保持 feats_dim 不变）。
    """

    def __init__(
        self,
        feats_dim: int,
        spk_dim: int,
        attention_heads: int = 4,
        num_tokens: int = 1,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(feats_dim, spk_dim)
        self.num_tokens = max(1, int(num_tokens))
        self.kv_proj = nn.Linear(spk_dim, feats_dim * self.num_tokens)
        # 每个 token 一个可学习偏置，使多 token 表达不同说话人子空间
        self.token_bias = nn.Parameter(torch.zeros(self.num_tokens, feats_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=feats_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(feats_dim)

    def forward(self, feats, feats_lengths, spk_emb):
        """交叉注意力融合：语音帧为 query，说话人 embedding 为 key/value。

        Args:
            feats: (B, T, feats_dim) 语音特征。
            feats_lengths: (B,) 每段有效帧数。
            spk_emb: (B, spk_dim) 说话人 embedding。

        Returns:
            (B, T, feats_dim) 融合后特征（残差加回 feats）。
        """
        b, t, _ = feats.shape
        kv = self.kv_proj(spk_emb).view(b, self.num_tokens, self.feats_dim)
        kv = kv + self.token_bias.unsqueeze(0)  # (B, num_tokens, feats_dim)
        attn_out, _ = self.attn(query=feats, key=kv, value=kv, need_weights=False)
        return self.norm(feats + attn_out)
