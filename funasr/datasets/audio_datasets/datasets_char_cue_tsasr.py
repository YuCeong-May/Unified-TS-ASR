# -*- coding: utf-8 -*-
"""实验3（V3）：唤醒词 **中文 char** 线索的 TS-ASR 数据集。

与实验1（`datasets_phone_tsasr.AudioDatasetPhoneTS`）的唯一区别是线索的表示：
V1 读 jsonl 里预先 g2p 好的 `keyword_phone_ids`（音素词表），V3 直接用 ASR 自己的
`tokenizer` 现编 `keyword_text` 的 char id（与 CTC 输出同一个词表），因此**不需要
重新准备数据**，V1 的 jsonl 可以原样复用。

索引类沿用 `IndexDSJsonlPhoneTS`——它已经把父类丢掉的 `keyword_text` 补了回来。

collator 的坑与 V1 相同：父类对 int 张量统一用 `int_pad_value`(-1) pad，会与
`WordEmbedding(padding_idx=0)` 冲突，故 `cue_token` 单独用 0 重新 pad；padded 位置
由 `cue_token_lengths` 在 cross-attention 的 mask 里屏蔽。
"""

import torch

from funasr.register import tables
from funasr.datasets.audio_datasets.bos_robust_read import BosRobustReadMixin
from funasr.datasets.audio_datasets.datasets import AudioDataset


@tables.register("dataset_classes", "AudioDatasetCharCueTS")
class AudioDatasetCharCueTS(BosRobustReadMixin, AudioDataset):
    """带唤醒词 char 线索的 ASR 数据集（无 reference 音频）。

    BOS 读取走 `BosRobustReadMixin`（重试退避 + worker 内并行），开关见
    `bos_robust_read.py`：`bos_workers` / `bos_read_retry` / `bos_retry_backoff`。
    """

    def __init__(self, path, index_ds: str = None, frontend=None, tokenizer=None, **kwargs):
        if index_ds is None:
            index_ds = "IndexDSJsonlPhoneTS"
        super().__init__(
            path, index_ds=index_ds, frontend=frontend, tokenizer=tokenizer, **kwargs
        )
        self._init_bos_robust_read(**kwargs)
        if self.tokenizer is None:
            raise ValueError("AudioDatasetCharCueTS 需要 tokenizer 来编码 char 线索")
        # 唤醒词种类极少（本项目 20 种 / 151 万条），缓存避免每条样本重复分词
        self._cue_cache = {}

    def _encode_cue(self, text: str):
        """把唤醒词文本编码成 char id 列表（按文本缓存）。"""
        ids = self._cue_cache.get(text)
        if ids is None:
            ids = self.tokenizer.encode(text)
            self._cue_cache[text] = ids
        return ids

    def __getitem__(self, index):
        sample = super().__getitem__(index)  # {speech, speech_lengths, text, text_lengths}

        kw_text = self.index_ds[index].get("keyword_text", "")
        if kw_text:
            ids = self._encode_cue(kw_text)
            sample["cue_token"] = torch.tensor(ids, dtype=torch.int64)
            sample["cue_token_lengths"] = torch.tensor([len(ids)], dtype=torch.int32)
        # 无 keyword_text -> 不加字段 -> 走无线索前向
        return sample

    def collator(self, samples: list = None):
        """父类 collate + `cue_token` 用 padding_value=0 重新 pad。

        一个 batch 内可能混有带/不带线索的样本；父类按 key 堆叠，key 不齐会导致
        batch 维错位。因此只要 batch 内出现任一 `cue_token`，就给缺失的样本补一个
        长度 1、值 0 的占位（`cue_token_lengths=0`），它在 cross-attention 里被
        完全屏蔽，等价于无线索前向。
        """
        has_cue = any("cue_token" in s for s in samples)
        cue_list = None
        if has_cue:
            for s in samples:
                if "cue_token" not in s:
                    s["cue_token"] = torch.zeros(1, dtype=torch.int64)
                    s["cue_token_lengths"] = torch.tensor([0], dtype=torch.int32)
            cue_list = [s.pop("cue_token") for s in samples]

        outputs = super().collator(samples)

        if cue_list is not None:
            outputs["cue_token"] = torch.nn.utils.rnn.pad_sequence(
                cue_list, batch_first=True, padding_value=0
            )
        return outputs
