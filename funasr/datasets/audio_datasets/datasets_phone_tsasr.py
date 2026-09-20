"""Text-cued Target-Speaker ASR 数据集（实验1）。

两种训练样本混用（同一 jsonl 内可混存）：
    1. 传统 ASR：无唤醒词线索字段 -> 正常 encoder-decoder（与现网一致）。
    2. 带唤醒词：提供唤醒词文本及其预转好的音素 id -> 走音素注入路径。

jsonl 每行在标准 ASR 字段（key/source/target/source_len/target_len）基础上，
可选新增以下字段（缺失即视为传统 ASR 样本）：
    {
      "keyword_text":       "小吉同学",           # 唤醒词原文（透传，不参与训练计算）
      "keyword_phones":     ["x","iao3",...],       # 音素 token 列表（透传，便于调试）
      "keyword_phone_ids":  [12, 5, 33, ...],       # 音素 int id 列表（训练用，0 预留 pad）
      "keyword_phone_len":  8                        # 音素序列长度（可选，缺则按 ids 长度）
    }

设计要点：
    - 训练阶段不做 g2p：dataset 直接读取预先转换好的 keyword_phone_ids。
    - 无 reference 音频：唤醒词与指令是同一条语音，不读 target_wav / reference。
    - 无 keyword_phone_ids 的样本自动退化为传统 ASR（不加 ref_phoneme 字段）。
    - collator 必须重写：父类 int_pad_value=-1 会把 ref_phoneme pad 成 -1，
      与 WordEmbedding(padding_idx=0) 冲突（负索引非法）。这里对 ref_phoneme
      用 padding_value=0 重新 pad，padded 位置由 kw_valid（真实长度）在
      cross-attention 中屏蔽，不影响输出。

不改动原有 AudioDataset / IndexDSJsonl，通过子类实现，未启用时零影响。
"""

import json
import logging

import torch

from funasr.register import tables
from funasr.datasets.audio_datasets.bos_robust_read import BosRobustReadMixin
from funasr.datasets.audio_datasets.datasets import AudioDataset
from funasr.datasets.audio_datasets.index_ds import IndexDSJsonlRankFull


@tables.register("index_ds_classes", "IndexDSJsonlPhoneTS")
class IndexDSJsonlPhoneTS(IndexDSJsonlRankFull):
    """在 IndexDSJsonl 基础上额外解析唤醒词音素字段（缺失即传统 ASR 样本）。"""

    def __init__(self, path: str, **kwargs):
        super().__init__(path, **kwargs)
        # 父类已解析并过滤生成 self.contents，但丢弃了唤醒词字段。
        # 这里重新扫描一遍 jsonl，按 source 建映射并补回音素字段。
        kw_map = {}
        # 唤醒词种类通常远少于样本数（本项目 20 种 / 151 万条），按 keyword_text
        # 归一到同一份对象，避免为每条样本各存一份音素 list 而吃掉大量内存。
        kw_cache = {}
        file_list = self._resolve_file_list(path, **kwargs)
        for file_json in file_list:
            with open(file_json.strip(), encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if "source" not in data:
                        continue
                    src = data["source"].replace("/cpfs01", "/cpfs_speech/data")
                    phone_ids = data.get("keyword_phone_ids", None)
                    if phone_ids is None:
                        continue  # 传统 ASR 样本，无需记录
                    kw_text = data.get("keyword_text", "")
                    entry = kw_cache.get(kw_text)
                    if entry is None:
                        entry = {
                            "keyword_text": kw_text,
                            "keyword_phones": data.get("keyword_phones", None),
                            "keyword_phone_ids": phone_ids,
                            "keyword_phone_len": data.get(
                                "keyword_phone_len", len(phone_ids)
                            ),
                        }
                        kw_cache[kw_text] = entry
                    kw_map[src] = entry
        n_with_kw = 0
        for c in self.contents:
            extra = kw_map.get(c["source"])
            if extra is not None:
                c["keyword_text"] = extra["keyword_text"]
                c["keyword_phones"] = extra["keyword_phones"]
                c["keyword_phone_ids"] = extra["keyword_phone_ids"]
                c["keyword_phone_len"] = extra["keyword_phone_len"]
                n_with_kw += 1
        logging.info(
            f"[IndexDSJsonlPhoneTS] {n_with_kw}/{len(self.contents)} 条样本带唤醒词音素线索"
        )

    @staticmethod
    def _resolve_file_list(path: str, **kwargs):
        """复现父类的 file_list 解析逻辑（单 jsonl / jsonl 列表文件 + 数据切分）。"""
        is_training = kwargs.get("is_training", True)
        if path.endswith(".jsonl") or path.endswith(".json"):
            return [path]
        data_split_num = kwargs.get("data_split_num", 1)
        data_split_i = kwargs.get("data_split_i", 0)
        if not is_training:
            data_split_num, data_split_i = 1, 0
        with open(path, encoding="utf-8") as fin:
            file_list_all = fin.readlines()
        num_per_slice = (len(file_list_all) - 1) // data_split_num + 1
        return file_list_all[
            data_split_i * num_per_slice : (data_split_i + 1) * num_per_slice
        ]


@tables.register("dataset_classes", "AudioDatasetPhoneTS")
class AudioDatasetPhoneTS(BosRobustReadMixin, AudioDataset):
    """带唤醒词音素线索的 ASR 数据集（无 reference 音频）。

    BOS 读取走 `BosRobustReadMixin`（重试退避 + worker 内并行），开关见
    `bos_robust_read.py`：`bos_workers` / `bos_read_retry` / `bos_retry_backoff`。
    音频读取本身在父类 `__getitem__` 里，由 mixin 覆盖 `_load_source` 接管。
    """

    def __init__(self, path, index_ds: str = None, frontend=None, tokenizer=None, **kwargs):
        # 默认索引类切换为带唤醒词音素的版本（config 未显式指定时）
        if index_ds is None:
            index_ds = "IndexDSJsonlPhoneTS"
        super().__init__(
            path, index_ds=index_ds, frontend=frontend, tokenizer=tokenizer, **kwargs
        )
        self._init_bos_robust_read(**kwargs)

    def __getitem__(self, index):
        sample = super().__getitem__(index)  # {speech, speech_lengths, text, text_lengths}

        item = self.index_ds[index]
        phone_ids = item.get("keyword_phone_ids", None)
        if phone_ids is not None:
            sample["ref_phoneme"] = torch.tensor(phone_ids, dtype=torch.int64)
            sample["ref_phoneme_lengths"] = torch.tensor(
                [len(phone_ids)], dtype=torch.int32
            )
            # 唤醒词原文透传（字符串，collator 保持为 list），仅供调试/日志
            sample["keyword_text"] = item.get("keyword_text", "")
        # 无 keyword_phone_ids -> 不加字段 -> 走传统 ASR 前向
        return sample

    def collator(self, samples: list = None):
        """先走父类 collate，再对 ref_phoneme 用 padding_value=0 重新 pad。

        父类对 int64/int32 张量统一用 int_pad_value(-1) pad，会把 ref_phoneme
        的 padded 位置填成 -1，与 WordEmbedding(padding_idx=0) 冲突。这里单独把
        ref_phoneme 用 0 重新 pad（padded 位置由 kw_valid 在 cross-att 屏蔽）。

        注意：一个 batch 内可能混有带/不带 ref_phoneme 的样本。若同批既有带线索
        又有传统样本，则 ref_phoneme 只对带线索样本存在，父类 collate 会因为
        key 不齐而只堆叠出现过该 key 的样本 —— 这会与其它字段的 batch 维错位。
        为避免该问题，只要 batch 内出现任一 ref_phoneme，就为缺失样本补一个
        长度为 1、值为 0 的占位（对应空唤醒词），其 ref_phoneme_lengths=0，
        在 _inject_phoneme / cross-att 中被完全屏蔽，等价于无线索前向。
        """
        has_phoneme = any("ref_phoneme" in s for s in samples)
        if has_phoneme:
            for s in samples:
                if "ref_phoneme" not in s:
                    s["ref_phoneme"] = torch.zeros(1, dtype=torch.int64)
                    s["ref_phoneme_lengths"] = torch.tensor([0], dtype=torch.int32)
                    s.setdefault("keyword_text", "")

        # 取出 ref_phoneme 单独处理，其余字段交给父类
        ref_phoneme_list = None
        if has_phoneme:
            ref_phoneme_list = [s.pop("ref_phoneme") for s in samples]

        outputs = super().collator(samples)

        if ref_phoneme_list is not None:
            outputs["ref_phoneme"] = torch.nn.utils.rnn.pad_sequence(
                ref_phoneme_list, batch_first=True, padding_value=0
            )
        return outputs
