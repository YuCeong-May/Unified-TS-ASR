# -*- coding: utf-8 -*-
"""说话人分类辅助头：DAE-TSE / KCE `NetModules.CE` 的移植。

DAE-TSE 的 `AEDKWSASRPhone` 除了音素 CTC 之外还挂了一个说话人验证 CE 头
（`kce/src/kce/model/NetModules.py` 的 `class CE`，`sv_ce_crit`，权重
`loss_weight['sv_ce']`）。本项目复刻 V4 时把它**整个去掉**了，理由写在
`dae_kws_asr_model.py` 的模块 docstring 里：「本项目全树无说话人标签」。
aishell1 + aishell2 的 jsonl 带 `speaker` 字段，这个前提不再成立，于是把它加回来。

这个头在做什么、为什么值得加：
    encoder 的隐层状态**已经是线索条件化**的（每层 cross-attend text-cue /
    enroll-cue）。在它上面问「你锁定的是谁的声音」，就是直接监督「有没有锁对人」。
    纯 CTC 只惩罚转写错误，而混合语音里模型完全可以靠语言模型先验蒙对大半内容却
    根本没做说话人选择；这个 CE 把「选人」这一步显式地拉出来打分。

与 DAE-TSE 的三处差异（都是修 bug / 去掉死配置，机制不变）：
    1. DAE-TSE 的 `forward` 里 `label[is_target == 0] = CE_IGNORE_INDEX` 是**原地**
       改调用方的张量。这里先 clone 再改。
    2. DAE-TSE 的 L2 正则 `(self.weight.norm(2) - 1) ** 2` 无条件执行，但
       `self.weight` 只在 `pooling_type == 'learnable_weights'` 下存在，`mean` 池化
       配上 `use_reg_loss=True` 会 AttributeError。这里 `weight_reg` 默认 0，且只在
       真的有可学权重时才生效。
    3. `use_softmax=False`（层权重不归一化）那条路去掉了：不归一化时层权重与后面的
       `Linear` 可以互相吸收，等于多一个无约束的整体缩放，没有意义。

`reduction` 用 `mean`（DAE-TSE 用 `sum`），这样 loss 的量纲不随 batch 里被忽略的
样本数漂移，权重 `spk_ce_weight` 才是一个可跨实验比较的数。
"""
import logging

import torch
import torch.nn as nn

# `nn.CrossEntropyLoss` 的默认 ignore_index，沿用 DAE-TSE 的 CE_IGNORE_INDEX。
CE_IGNORE_INDEX = -100


class SpeakerCEHead(nn.Module):
    """逐层加权池化 + 时间维掩码平均 + 线性分类 + CE。

    Args:
        size: encoder 的模型维度 D。
        num_speakers: 说话人类别数，须与 `spk2id` 映射的大小一致。
        layer_indices: 参与池化的 `hidden_states` 下标，支持负数。默认 `[-2]`，
            即**倒数第二层**——DAE-TSE 不开 SV 时用的就是 `sph_emb_list[-2]`
            （`wesep/modules/common/text.py:164`）。最后一层已经被 CTC 拉向
            「输出字」的方向，说话人信息在那里最弱。
        layer_pooling: 多层怎么合并。`mean` = 等权平均；`learnable_weights` =
            一组可学权重过 softmax 后加权（DAE-TSE 的默认）。只选一层时本项无效。
        hidden_dim: None（默认，与 DAE-TSE 一致）表示单层 `Linear(D, S)`；给一个
            整数则插一层 `Linear(D, h) + ReLU`。
        dropout: 池化向量上的 dropout。
        label_smoothing: 传给 `nn.CrossEntropyLoss`。
        weight_reg: 层权重的 L2 拉向 1 的正则系数，仅 `learnable_weights` 下生效。
            默认 0 = 关闭。
    """

    _POOLING = ("mean", "learnable_weights")

    def __init__(
        self,
        size: int,
        num_speakers: int,
        layer_indices=None,
        layer_pooling: str = "mean",
        hidden_dim: int = None,
        dropout: float = 0.0,
        label_smoothing: float = 0.0,
        weight_reg: float = 0.0,
    ):
        super().__init__()
        if int(num_speakers) < 2:
            raise ValueError(
                f"SpeakerCEHead: num_speakers={num_speakers}；请用 "
                "script/data/make_spk2id.py 从训练 jsonl 生成 spk2id.json，"
                "并把它的大小填进 num_speakers"
            )
        if layer_pooling not in self._POOLING:
            raise ValueError(
                f"layer_pooling must be one of {self._POOLING}, got {layer_pooling!r}"
            )
        layer_indices = [-2] if layer_indices is None else list(layer_indices)
        if not layer_indices:
            raise ValueError("layer_indices must not be empty")

        self.num_speakers = int(num_speakers)
        self.layer_indices = [int(i) for i in layer_indices]
        self.layer_pooling = str(layer_pooling)
        self.weight_reg = float(weight_reg)

        if self.layer_pooling == "learnable_weights" and len(self.layer_indices) > 1:
            # 初值均匀 = 与 `mean` 起点相同，训练再自己挪。
            self.layer_weight = nn.Parameter(
                torch.full((len(self.layer_indices),), 1.0 / len(self.layer_indices))
            )
        else:
            self.register_parameter("layer_weight", None)

        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        if hidden_dim:
            self.classifier = nn.Sequential(
                nn.Linear(size, int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), self.num_speakers),
            )
        else:
            self.classifier = nn.Linear(size, self.num_speakers)
        self.ce = nn.CrossEntropyLoss(
            ignore_index=CE_IGNORE_INDEX,
            label_smoothing=float(label_smoothing),
            reduction="mean",
        )
        self._warned_all_ignored = False

    # ------------------------------------------------------------------ 池化
    def _select_layers(self, hidden_states):
        """按 `layer_indices` 取层并合并成一个 (B, T, D)。"""
        n = len(hidden_states)
        if n == 0:
            raise ValueError(
                "SpeakerCEHead 收到空的 hidden_states；encoder 需要以 "
                "return_hidden=True 前向"
            )
        for i in self.layer_indices:
            if not -n <= i < n:
                raise ValueError(
                    f"layer_indices 里的 {i} 越界：encoder 只返回了 {n} 层 hidden_states"
                )
        picked = [hidden_states[i] for i in self.layer_indices]
        if len(picked) == 1:
            return picked[0]
        stacked = torch.stack(picked, dim=0)  # (L, B, T, D)
        if self.layer_weight is None:
            return stacked.mean(dim=0)
        weight = torch.softmax(self.layer_weight, dim=0)
        return (stacked * weight.view(-1, 1, 1, 1).to(stacked.dtype)).sum(dim=0)

    @staticmethod
    def _masked_mean(feats, lengths):
        """时间维掩码平均，只对每条自己的有效帧求均值。

        Args:
            feats: (B, T, D)。
            lengths: (B,) 有效帧数。

        Returns:
            (B, D)。
        """
        frames = feats.size(1)
        index = torch.arange(frames, device=feats.device).unsqueeze(0)
        keep = (index < lengths.view(-1, 1).to(index.device)).unsqueeze(-1)
        keep = keep.to(feats.dtype)
        # 长度为 0 的行（理论上不该出现）分母兜到 1，避免 0/0。
        return (feats * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)

    def extract_embedding(self, hidden_states, lengths, detach=False):
        """Extract the same pooled speaker representation used by the CE head."""
        feats = self._select_layers(hidden_states)
        embedding = self._masked_mean(feats, lengths.reshape(-1))
        return embedding.detach() if detach else embedding

    # ------------------------------------------------------------------ 前向
    def forward(self, hidden_states, lengths, labels, valid=None):
        """算说话人 CE 与 top-1 正确率。

        Args:
            hidden_states: list[(B, T, D)]，encoder 每层的输出。
            lengths: (B,) 或 (B, 1) encoder 输出有效帧数。
            labels: (B,) 或 (B, 1) 目标说话人 id；**负值表示未知**（例如验证集的
                说话人不在训练映射里），会被忽略。
            valid: (B,) 或 (B, 1) 可选布尔/0-1 掩码，False 表示这条样本没有任何
                线索指向混合语音里的某个人，因此「目标说话人是谁」这个问题无解，
                必须忽略（对应 DAE-TSE 的 `is_target == 0`）。

        Returns:
            `(loss, acc)`。整个 batch 全被忽略时返回一个**连着计算图的零**，
            这样 DDP 仍然收到本头所有参数的（零）梯度，不会报 unused parameter。
        """
        pooled = self.extract_embedding(hidden_states, lengths)
        logits = self.classifier(self.dropout(pooled))

        target = labels.reshape(-1).to(device=logits.device, dtype=torch.long).clone()
        if valid is not None:
            drop = ~valid.reshape(-1).to(device=logits.device, dtype=torch.bool)
            target = target.masked_fill(drop, CE_IGNORE_INDEX)
        # 未知说话人（-1）与被忽略的样本统一成 ignore_index。
        target = target.masked_fill(target < 0, CE_IGNORE_INDEX)
        over = target >= self.num_speakers
        if bool(over.any()):
            raise ValueError(
                f"spk_id 最大为 {int(target.max())} 但 num_speakers={self.num_speakers}；"
                "model_conf.spk_ce_conf.num_speakers 与 dataset_conf.spk2id 不是同一份映射"
            )

        keep = target != CE_IGNORE_INDEX
        n_keep = int(keep.sum())
        if n_keep == 0:
            if not self._warned_all_ignored:
                logging.warning(
                    "[SpeakerCEHead] 整个 batch 的说话人标签都被忽略；"
                    "检查 dataset_conf.spk2id 是否覆盖了这份数据的说话人"
                )
                self._warned_all_ignored = True
            zero = logits.sum() * 0.0
            return zero, torch.zeros((), device=logits.device)

        loss = self.ce(logits, target)
        if self.weight_reg > 0 and self.layer_weight is not None:
            loss = loss + self.weight_reg * (self.layer_weight.norm(2) - 1.0) ** 2
        with torch.no_grad():
            acc = (logits.argmax(dim=-1)[keep] == target[keep]).to(loss.dtype).mean()
        return loss, acc
