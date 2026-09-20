"""Target-Speaker ASR 数据集：在标准 ASR 样本基础上，额外加载一条目标说话人参考音频。

jsonl 每行格式（在原有字段基础上新增 reference / ref_target）：
    {
      "key": "utt-id",
      "source": "/path/to/mix_or_target.wav",   # 主音频（待识别，混合语音）
      "target": "识别文本",
      "source_len": 512, "target_len": 10,
      "reference": "/path/to/enroll.wav",         # 目标说话人参考音频（新增）
      "ref_target": "参考音频对应文本"             # 参考音频文本（新增，目前不使用，留作扩展）
    }

- reference 音频与 source 走同一套读取逻辑（支持本地 / BOS）。
- reference 特征：默认 80 维 kaldi-fbank + 逐句 mean 归一（与 CAMPPlus/ERes2Net/ECAPA 约定一致），
  与 ASR 的 LFR/CMVN 特征相互独立。
- ref_target 目前仅透传为字符串，不参与训练。

不改动原有 AudioDataset / IndexDSJsonl，通过子类实现，未启用时零影响。
"""

import json
import logging

import torch
import torchaudio.compliance.kaldi as Kaldi

from funasr.register import tables
from funasr.datasets.audio_datasets.bos_robust_read import BosRobustReadMixin
from funasr.datasets.audio_datasets.datasets import AudioDataset
from funasr.datasets.audio_datasets.index_ds import IndexDSJsonlRankFull


@tables.register("index_ds_classes", "IndexDSJsonlTSASR")
class IndexDSJsonlTSASR(IndexDSJsonlRankFull):
    """在 IndexDSJsonl 基础上额外解析 reference / ref_target 字段。"""

    def __init__(self, path: str, **kwargs):
        super().__init__(path, **kwargs)
        # 父类已解析并过滤生成 self.contents，但丢弃了 reference/ref_target。
        # 这里重新扫描一遍 jsonl，按 source 建映射并补回引用字段。
        ref_map = {}
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
                    ref_map[src] = {
                        "reference": data.get("reference", None),
                        "ref_target": data.get("ref_target", data.get("ref_text", "")),
                    }
        n_with_ref = 0
        for c in self.contents:
            extra = ref_map.get(c["source"])
            if extra is not None:
                c["reference"] = extra["reference"]
                c["ref_target"] = extra["ref_target"]
                if extra["reference"] is not None:
                    n_with_ref += 1
        logging.info(
            f"[IndexDSJsonlTSASR] {n_with_ref}/{len(self.contents)} 条样本带 reference 音频"
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


@tables.register("dataset_classes", "AudioDatasetTSASR")
class AudioDatasetTSASR(BosRobustReadMixin, AudioDataset):
    """带目标说话人参考音频的 ASR 数据集。

    BOS 读取走 `BosRobustReadMixin`（重试退避 + worker 内并行），开关见
    `bos_robust_read.py`：`bos_workers` / `bos_read_retry` / `bos_retry_backoff`。
    主音频经父类 `_load_source` 钩子接管，参考音频见 `_load_ref_waveform`。
    """

    def __init__(self, path, index_ds: str = None, frontend=None, tokenizer=None, **kwargs):
        # 默认索引类切换为带 reference 的版本（config 未显式指定时）
        if index_ds is None:
            index_ds = "IndexDSJsonlTSASR"
        super().__init__(
            path, index_ds=index_ds, frontend=frontend, tokenizer=tokenizer, **kwargs
        )
        self._init_bos_robust_read(**kwargs)
        # 参考音频特征模式：
        #   "waveform"    -> 传原始波形，特征在 spk_encoder 内部计算（如 ECAPASpeechBrain）
        #   "kaldi_fbank" -> dataset 侧提 80 维 kaldi-fbank（如仓库内 ECAPATDNN/CAMPPlus）
        self.ref_feature = str(kwargs.get("ref_feature", "kaldi_fbank")).lower()
        # 参考音频特征参数（kaldi_fbank 模式下使用）
        self.ref_feat_dim = int(kwargs.get("ref_feat_dim", 80))
        self.ref_cmn = bool(kwargs.get("ref_cmn", True))  # 逐句 mean 归一
        self.ref_fs = int(kwargs.get("ref_fs", 16000))
        # reference 缺失时的处理："error"（默认，直接报错）| "zero"（用占位）
        self.ref_missing = str(kwargs.get("ref_missing", "error")).lower()

    def _load_ref_waveform(self, ref_source: str):
        """按本地 / BOS 读取参考音频，返回 1D 波形张量（BOS 下带重试退避）。"""
        return self._load_waveform_robust(ref_source, fs=self.ref_fs)

    def _extract_ref_feature(self, wav: torch.Tensor) -> torch.Tensor:
        """80 维 kaldi-fbank（+逐句 mean 归一），与 CAMPPlus/ECAPA 约定一致。返回 (T, feat_dim)。"""
        feat = Kaldi.fbank(wav.unsqueeze(0), num_mel_bins=self.ref_feat_dim)
        if self.ref_cmn:
            feat = feat - feat.mean(dim=0, keepdim=True)
        return feat

    def __getitem__(self, index):
        sample = super().__getitem__(index)  # {speech, speech_lengths, text, text_lengths}

        item = self.index_ds[index]
        ref_source = item.get("reference", None)

        if ref_source is None:
            if self.ref_missing == "zero":
                if self.ref_feature == "waveform":
                    ref_speech = torch.zeros(self.ref_fs, dtype=torch.float32)  # 1s 静音占位
                else:
                    ref_speech = torch.zeros(1, self.ref_feat_dim, dtype=torch.float32)
            else:
                raise ValueError(
                    f"AudioDatasetTSASR: 样本缺少 'reference' 字段 (index={index}, "
                    f"source={item.get('source')})。请在 jsonl 中补充 reference，"
                    f"或设置 dataset_conf.ref_missing: zero。"
                )
        else:
            wav = self._load_ref_waveform(ref_source)
            if self.ref_feature == "waveform":
                ref_speech = wav  # (T_samples,) 原始波形，特征在 spk_encoder 内计算
            else:
                ref_speech = self._extract_ref_feature(wav)  # (T_ref, feat_dim)

        ref_speech_lengths = torch.tensor([ref_speech.shape[0]], dtype=torch.int32)

        sample["ref_speech"] = ref_speech
        sample["ref_speech_lengths"] = ref_speech_lengths
        # 参考文本目前不使用，仅透传（字符串，collator 会保持为 list）
        sample["ref_text"] = item.get("ref_target", "")
        return sample
