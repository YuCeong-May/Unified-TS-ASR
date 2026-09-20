# -*- coding: utf-8 -*-
"""实验3（V3）：KCE 线索在 Paraformer encoder **内部**做 cross-attention。

与实验1（`phoneme_fusion.PhonemeCrossAttnFusion`）的结构差异：

    V1: fbank -> [6 层独立 Conformer, 每层 cross-att] -> 50 层 SANMEncoder -> decoder/CIF
    V3: fbank -> SANMEncoder 1..8 层 -> [xattn] -> 9..16 层 -> [xattn] -> ... -> CTC

V1 把线索一次性揉进声学特征后就交给主干，主干本身不知道线索存在；V3 让线索在主干
内部反复注入，主干自己参与「按线索定位目标说话人」的计算。DAE-TSE 用的是它自己的
speech_transformer，这里换成预训练好的 Paraformer SANMEncoder。

cross-attention 模块复用 `dae_modules.MultiHeadCrossAtt`（源自 DAE-TSE NetModules）。
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from funasr.register import tables
from funasr.models.sanm.encoder import SANMEncoder
from funasr.models.transformer.utils.nets_utils import make_pad_mask
from funasr.models.tsasr.dae_modules import MultiHeadCrossAtt, combine_mask


class KceCrossAttnAdapter(nn.Module):
    """单个层间线索注入块：encoder hidden 作 query，线索序列作 key/value。

    Args:
        enc_dim: encoder hidden 维度 D（SANMEncoder output_size，本项目 512）。
        attention_heads: cross-attention 头数。DAE-TSE 原配置用 1 头
            （`kce/config/OverlapASR/train.yaml` 的 `corss_att_config.n_head: 1`），
            这里默认 4 以与实验1 保持一致，可在 config 里改回 1 做消融。
        cross_att_norm: `MultiHeadCrossAtt` 内部 q/k/v 投影前的归一化。
        zero_init: True 时把出口投影置零 -> step 0 的输出与未注入时逐位相同，
            等价于预训练 Paraformer。实验1 实测随机初始化的注入层会把 acc
            从 0.908 直接打到 0.092，这里沿用同样的处理。
    """

    def __init__(
        self,
        enc_dim: int = 512,
        attention_heads: int = 4,
        cross_att_norm: str = "LayerNorm",
        zero_init: bool = True,
    ):
        super().__init__()
        self.cross_att = MultiHeadCrossAtt(
            n_head=attention_heads, n_feats=enc_dim, norm=cross_att_norm
        )
        self.out_proj = nn.Linear(enc_dim, enc_dim)
        if zero_init:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self, hidden: torch.Tensor, cue: torch.Tensor, cross_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden:     (B, T, D) encoder 某一层的输出。
            cue:        (B, L, D) 线索特征（已投影到 D 维）。
            cross_mask: (B, 1, T, L) bool，**True 表示有效**（`MultiHeadCrossAtt`
                        内部会 `mask.eq(0)` 反转后 masked_fill）。

        Returns:
            (B, T, D) 注入后的 hidden，残差形式 `hidden + out_proj(cross_att(...))`。
        """
        context, _ = self.cross_att(hidden, cue, cue, cross_mask)
        return hidden + self.out_proj(context)


@tables.register("encoder_classes", "SANMEncoderKCE")
class SANMEncoderKCE(SANMEncoder):
    """SANMEncoder + 层间线索 cross-attention（V3）。

    除下列参数外，其余全部透传给 `SANMEncoder`，因此可以直接复用预训练
    Paraformer 的 encoder 权重（新增的 `kce_*` 子模块在 ckpt 里不存在，
    由 `init_param` 的 strict=False 加载路径留作随机/零初始化）。

    Args:
        kce_inject_layers: 注入位置，**全局 1-based 层号**。层 1 = `encoders0`，
            层 n(>=2) = `encoders[n-2]`。默认 (8,16,24,32,40,48) 即每 8 层一处。
        kce_cue_dim: 线索编码器的输出维度。与 `output_size` 不等时自动加一层
            线性投影（对应 DAE-TSE 的 `kw_au_link`）。
        kce_attention_heads / kce_cross_att_norm / kce_zero_init: 见
            `KceCrossAttnAdapter`。
    """

    def __init__(
        self,
        *args,
        kce_inject_layers: List[int] = (8, 16, 24, 32, 40, 48),
        kce_cue_dim: int = 256,
        kce_attention_heads: int = 4,
        kce_cross_att_norm: str = "LayerNorm",
        kce_zero_init: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if len(self.interctc_layer_idx) > 0:
            raise NotImplementedError("SANMEncoderKCE 不支持 interctc_layer_idx")

        num_blocks = 1 + len(self.encoders)
        layers = sorted({int(x) for x in kce_inject_layers})
        for n in layers:
            if not 1 <= n <= num_blocks:
                raise ValueError(
                    f"kce_inject_layers 含非法层号 {n}，合法范围 [1, {num_blocks}]"
                )
        self.kce_inject_layers = layers

        enc_dim = self.output_size()
        self.kce_cue_link = (
            nn.Linear(kce_cue_dim, enc_dim) if kce_cue_dim != enc_dim else nn.Identity()
        )
        # 用 ModuleDict（键=层号字符串）而非 ModuleList，让 state_dict 键自带层号，
        # 改动注入位置后旧 ckpt 里对得上的 adapter 仍能按层号复用。
        self.kce_adapters = nn.ModuleDict(
            {
                str(n): KceCrossAttnAdapter(
                    enc_dim=enc_dim,
                    attention_heads=kce_attention_heads,
                    cross_att_norm=kce_cross_att_norm,
                    zero_init=kce_zero_init,
                )
                for n in layers
            }
        )

    def _inject(
        self,
        layer_no: int,
        xs_pad: torch.Tensor,
        cue: Optional[torch.Tensor],
        cross_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """若 layer_no 是注入点且线索存在，则做一次 cross-attention 注入。"""
        key = str(layer_no)
        if cue is None or key not in self.kce_adapters:
            return xs_pad
        return self.kce_adapters[key](xs_pad, cue, cross_mask)

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
        ctc=None,
        cue_feat: torch.Tensor = None,
        cue_valid: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            xs_pad:   (B, T, input_size) 声学特征（CMVN 之后）。
            ilens:    (B,) 有效帧数。
            cue_feat: (B, L, kce_cue_dim) 线索特征；None 时退化为普通 SANMEncoder。
            cue_valid: (B, L) bool，True 表示线索有效位置。

        Returns:
            (encoder_out, olens, None)，与 `SANMEncoder.forward` 一致。
        """
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
        xs_pad = xs_pad * self.output_size() ** 0.5
        if self.embed is not None:
            xs_pad = self.embed(xs_pad)

        cue, cross_mask = None, None
        if cue_feat is not None:
            if cue_valid is None:
                cue_valid = torch.ones(
                    cue_feat.shape[:2], dtype=torch.bool, device=cue_feat.device
                )
            cue = self.kce_cue_link(cue_feat)
            # combine_mask 返回 True=无效，取反得到 True=有效（MultiHeadCrossAtt 约定）
            cross_mask = ~combine_mask(masks.squeeze(1), cue_valid, 1)

        encoder_outs = self.encoders0(xs_pad, masks)
        xs_pad, masks = encoder_outs[0], encoder_outs[1]
        xs_pad = self._inject(1, xs_pad, cue, cross_mask)

        for layer_idx, encoder_layer in enumerate(self.encoders):
            encoder_outs = encoder_layer(xs_pad, masks)
            xs_pad, masks = encoder_outs[0], encoder_outs[1]
            xs_pad = self._inject(layer_idx + 2, xs_pad, cue, cross_mask)

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        olens = masks.squeeze(1).sum(1)
        return xs_pad, olens, None
