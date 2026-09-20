"""V2（实验2）数据集：唤醒词段自注册的 target-speaker ASR。

一条音频里唤醒词与指令连在一起，本数据集把它切成两段、各走各的通路：

    wav ── [kw_start, kw_end] ──> ref_speech（波形）──> spk_encoder ──> FiLM 条件
        └─ [cmd_start, end]   ──> fbank ──> speech ──> encoder（target 只含指令词）

即「唤醒词段」只作为说话人线索，不进 ASR 输入；target 也不含唤醒词。推理时 KWS 触发后
唤醒词边界天然已知，所以这套口径可直接部署，无需额外注册音频。

jsonl 每行（由 script/align/make_v2_jsonl.py 产出）：
    {
      "key": "utt-id",
      "source": "/path/to/mix.wav",
      "target": "指令词文本",              # 不含唤醒词；负样本为空串（-> 单个 <unk>）
      "source_len": 391, "target_len": 7,  # source_len = 指令段的 10ms 帧数
      "kw_start": 0.29, "kw_end": 1.25,    # 唤醒词本身（秒，该 wav 自身时间轴）
      "ref_start": 0.0, "ref_end": 1.75,   # 注册段 = 唤醒词前后各 0.5s，后界不越过指令首字
      "cmd_start": 1.25,                   # 指令段起点（= kw_end），一直取到句末
      "emb_key": "utt-id"                  # 取哪一条的说话人 embedding；负样本指向别人
    }

两种 reference 供给方式：
  - `spk_emb_store`（推荐）：script/align/extract_spk_emb.py 离线提好的 memmap，
    dataset 只按 `emb_key` 查表，返回 `spk_emb`（192 维）。spk_encoder 完全不进计算图。
    负样本正是靠 `emb_key` 指向另一位说话人来构造的，必须走这条路。
  - 不配 `spk_emb_store`：退回读波形、返回 `ref_speech`，由模型内部现算 embedding。

与 datasets_tsasr.py（外部 reference 音频）的区别：参考音频不是另一个文件，而是同一条
音频的一个切片，所以**只读一次 wav**，不产生额外 I/O。

不改动原有 AudioDataset / IndexDSJsonl，通过子类实现，未启用时零影响。
"""

import json
import logging
import os

import numpy as np
import torch

from funasr.register import tables
from funasr.datasets.audio_datasets.bos_robust_read import BosRobustReadMixin
from funasr.datasets.audio_datasets.datasets import AudioDataset
from funasr.datasets.audio_datasets.index_ds import IndexDSJsonlRankFull
from funasr.utils.load_utils import extract_fbank


class SpkEmbStore:
    """离线预提说话人 embedding 的只读查表（memmap，按 key 索引）。

    目录里需要这几样（extract_spk_emb.py --merge 产出）：
        spk_emb.f32        float32 (N, D) 裸数组
        spk_emb.keys       N 行 key，与数组逐行对应
        spk_emb.meta.json  {"num": N, "dim": D, "mean_file": "spk_emb.mean.npy"}
        spk_emb.mean.npy   全局均值（norm="center_l2" 时用）
    memmap 在每个 dataloader worker 里 lazy 打开，避免 fork 后句柄失效。

    norm（默认 "center_l2"）：
        "center_l2"  (x - 全局均值) 再 L2 归一。说话人验证的标准前处理；实测把
                     「同说话人 base↔aug」vs「异说话人」的近似 EER 从 20.0% 降到 13.5%
                     （全局均值范数占 embedding 范数的 39%，是不含身份信息的共同分量）。
                     同时把输入尺度拉到 1，否则原始范数约 365，会让 FiLM 生成器的
                     有效学习率被放大几百倍。
        "l2"         只做 L2 归一。
        "none"       原样返回（仅用于对照实验）。
    """

    def __init__(self, path: str, norm: str = "center_l2"):
        self.dir = path if os.path.isdir(path) else os.path.dirname(path)
        self.norm = str(norm).lower()
        with open(os.path.join(self.dir, "spk_emb.meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        self.num = int(meta["num"])
        self.dim = int(meta["dim"])
        self.mean = None
        if self.norm == "center_l2":
            mean_path = os.path.join(self.dir, meta.get("mean_file", "spk_emb.mean.npy"))
            if not os.path.isfile(mean_path):
                raise FileNotFoundError(
                    f"norm=center_l2 需要 {mean_path}；请重跑 extract_spk_emb.sh merge"
                )
            self.mean = torch.from_numpy(np.load(mean_path).astype(np.float32))
        with open(os.path.join(self.dir, "spk_emb.keys"), encoding="utf-8") as f:
            self.key2row = {line.rstrip("\n"): i for i, line in enumerate(f) if line.strip()}
        self._mm = None

    def _mat(self):
        if self._mm is None:
            self._mm = np.memmap(
                os.path.join(self.dir, "spk_emb.f32"),
                dtype=np.float32,
                mode="r",
                shape=(self.num, self.dim),
            )
        return self._mm

    def __contains__(self, key):
        return key in self.key2row

    def get(self, key):
        """按 key 取归一化后的说话人 embedding。

        Args:
            key: 样本 key。

        Returns:
            torch.Tensor: (D,) embedding；key 不存在时返回 None。
        """
        row = self.key2row.get(key)
        if row is None:
            return None
        emb = torch.from_numpy(np.array(self._mat()[row], dtype=np.float32))
        if self.norm == "none":
            return emb
        if self.mean is not None:
            emb = emb - self.mean
        return emb / (emb.norm() + 1e-8)


@tables.register("index_ds_classes", "IndexDSJsonlSelfEnrollTS")
class IndexDSJsonlSelfEnrollTS(IndexDSJsonlRankFull):
    """在 IndexDSJsonl 基础上额外解析 kw_start / kw_end / cmd_start。

    父类只保留 key/source/target/source_len/target_len，会丢掉其余字段，
    所以这里重扫一遍 jsonl，按 source 建映射补回时间戳（与 IndexDSJsonlTSASR 同一手法）。
    """

    def __init__(self, path: str, **kwargs):
        super().__init__(path, **kwargs)
        # 父类只保留 source/prompt/target/source_len/target_len，会丢掉 key 与时间戳，
        # 所以重扫一遍 jsonl 补回来。映射键用 (source, target)：同一条 mixture 会被正样本
        # 与负样本共用（source 相同），但负样本 target 为空串，两者不会撞。
        seg_map, n_dup = {}, 0
        for file_json in self._resolve_file_list(path, **kwargs):
            with open(file_json.strip(), encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if "source" not in data:
                        continue
                    src = data["source"].replace("/cpfs01", "/cpfs_speech/data")
                    mkey = (src, data.get("target", ""))
                    val = (
                        data.get("ref_start", data.get("kw_start", None)),
                        data.get("ref_end", data.get("kw_end", None)),
                        data.get("cmd_start", None),
                        data.get("emb_key", data.get("key")),
                    )
                    if mkey in seg_map and seg_map[mkey] != val:
                        n_dup += 1
                    seg_map[mkey] = val
        if n_dup:
            logging.warning(
                f"[IndexDSJsonlSelfEnrollTS] {n_dup} 条 (source,target) 重复且元数据不同，"
                f"后者覆盖前者；请检查 jsonl 是否有重复样本。"
            )

        n_with_seg = n_with_emb = 0
        for c in self.contents:
            kw_start, kw_end, cmd_start, emb_key = seg_map.get(
                (c["source"], c.get("target", "")), (None, None, None, None)
            )
            c["kw_start"] = kw_start
            c["kw_end"] = kw_end
            # cmd_start 缺省时退化为 kw_end（对齐脚本正常时两者都在）
            c["cmd_start"] = cmd_start if cmd_start is not None else kw_end
            c["emb_key"] = emb_key
            if kw_start is not None and kw_end is not None:
                n_with_seg += 1
            if emb_key is not None:
                n_with_emb += 1
        logging.info(
            f"[IndexDSJsonlSelfEnrollTS] {len(self.contents)} 条样本，"
            f"{n_with_seg} 条带唤醒词时间戳，{n_with_emb} 条带 emb_key"
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


@tables.register("dataset_classes", "AudioDatasetSelfEnrollTS")
class AudioDatasetSelfEnrollTS(BosRobustReadMixin, AudioDataset):
    """唤醒词段自注册的 TS-ASR 数据集：一次读 wav，切「唤醒词段」+「指令段」。

    BOS 读取走 `BosRobustReadMixin`（重试退避 + worker 内并行），开关见
    `bos_robust_read.py`：`bos_workers` / `bos_read_retry` / `bos_retry_backoff`。
    注意开并行读后 `spk_emb_store` 的 memmap 会被多线程访问（只读，安全）。
    """

    # 唤醒词段短于此值（秒）视为异常，退化为 wav[:kw_end]
    MIN_ENROLL_SEC = 0.2

    def __init__(self, path, index_ds: str = None, frontend=None, tokenizer=None, **kwargs):
        if index_ds is None:
            index_ds = "IndexDSJsonlSelfEnrollTS"
        super().__init__(
            path, index_ds=index_ds, frontend=frontend, tokenizer=tokenizer, **kwargs
        )
        self._init_bos_robust_read(**kwargs)
        # 参考特征模式：waveform（ECAPASpeechBrain，内部自己算特征）| kaldi_fbank（仓库内 ECAPATDNN）
        self.ref_feature = str(kwargs.get("ref_feature", "waveform")).lower()
        self.ref_feat_dim = int(kwargs.get("ref_feat_dim", 80))
        self.ref_cmn = bool(kwargs.get("ref_cmn", True))
        # 时间戳缺失时的处理："error"（默认）| "full"（整条音频当指令段、1s 静音当注册段）
        self.seg_missing = str(kwargs.get("seg_missing", "error")).lower()
        self._n_fallback_enroll = 0
        # 离线预提 embedding 目录（推荐）：设置后不再读注册段波形，直接按 emb_key 查表
        self.spk_emb_store = None
        store_path = kwargs.get("spk_emb_store", None)
        if store_path:
            self.spk_emb_store = SpkEmbStore(
                store_path, norm=kwargs.get("emb_norm", "center_l2")
            )
            logging.info(
                f"[AudioDatasetSelfEnrollTS] 使用离线 embedding: {store_path} "
                f"({self.spk_emb_store.num} x {self.spk_emb_store.dim}, "
                f"norm={self.spk_emb_store.norm})"
            )

    def _extract_ref_feature(self, wav: torch.Tensor) -> torch.Tensor:
        """80 维 kaldi-fbank + 逐句 mean 归一（CAMPPlus/ECAPATDNN 约定）。返回 (T, dim)。"""
        import torchaudio.compliance.kaldi as Kaldi

        feat = Kaldi.fbank(wav.unsqueeze(0), num_mel_bins=self.ref_feat_dim)
        if self.ref_cmn:
            feat = feat - feat.mean(dim=0, keepdim=True)
        return feat

    def __getitem__(self, index):
        item = self.index_ds[index]
        wav = self._load_waveform_robust(item["source"])

        # index_ds 的 kw_start/kw_end 槽位存的是**注册段窗口** ref_start/ref_end
        # （由 make_v2_jsonl.py 按「唤醒词前后各 0.5s、后界不越过指令首字」算好）
        ref_start, ref_end = item.get("kw_start"), item.get("kw_end")
        cmd_start = item.get("cmd_start")
        if ref_start is None or ref_end is None:
            if self.seg_missing != "full":
                raise ValueError(
                    f"AudioDatasetSelfEnrollTS: 样本缺少 ref_start/ref_end (index={index}, "
                    f"source={item.get('source')})。请用 script/align/make_v2_jsonl.py 生成 "
                    f"jsonl，或设置 dataset_conf.seg_missing: full。"
                )
            ref_start, ref_end, cmd_start = 0.0, 0.0, 0.0
        if cmd_start is None:
            cmd_start = ref_end

        n = wav.numel()
        i_ref_s = min(max(int(ref_start * self.fs), 0), n)
        i_ref_e = min(max(int(ref_end * self.fs), i_ref_s), n)
        i_cmd = min(max(int(cmd_start * self.fs), 0), n)

        # ---- 指令段 -> ASR 输入（唤醒词结束处到句末）----
        cmd_wav = wav[i_cmd:]
        if cmd_wav.numel() < int(0.1 * self.fs):
            # 极端情况（时间戳越界）：退回整条音频，宁可多识别也不要空输入
            cmd_wav = wav
        if self.preprocessor_speech:
            cmd_wav = self.preprocessor_speech(cmd_wav, fs=self.fs)
        speech, speech_lengths = extract_fbank(
            cmd_wav, data_type=self.data_type, frontend=self.frontend, is_final=True
        )  # speech: [1, T, d]

        # ---- 注册段 -> 说话人线索 ----
        out = {}
        if self.spk_emb_store is not None:
            emb_key = item.get("emb_key")
            emb = self.spk_emb_store.get(emb_key) if emb_key else None
            if emb is None:
                raise ValueError(
                    f"AudioDatasetSelfEnrollTS: emb_key={emb_key!r} 不在预提 embedding 里"
                    f"（index={index}, source={item.get('source')}）。"
                    f"请重跑 script/align/extract_spk_emb.sh，或去掉 dataset_conf.spk_emb_store。"
                )
            out["spk_emb"] = emb
        else:
            enroll = wav[i_ref_s:i_ref_e]
            if enroll.numel() < int(self.MIN_ENROLL_SEC * self.fs):
                enroll = wav[:i_ref_e] if i_ref_e > 0 else wav[: self.fs]
                self._n_fallback_enroll += 1
                if self._n_fallback_enroll in (1, 10, 100, 1000):
                    logging.warning(
                        f"[AudioDatasetSelfEnrollTS] 注册段过短，已退化 "
                        f"{self._n_fallback_enroll} 次（最近: ref=[{ref_start},{ref_end}]，"
                        f"source={item.get('source')}）"
                    )
            if enroll.numel() == 0:
                enroll = torch.zeros(self.fs, dtype=torch.float32)
            ref_speech = (
                enroll if self.ref_feature == "waveform" else self._extract_ref_feature(enroll)
            )
            out["ref_speech"] = ref_speech
            out["ref_speech_lengths"] = torch.tensor([ref_speech.shape[0]], dtype=torch.int32)

        # ---- 文本（只含指令词；负样本为空串 -> 单个 <unk>）----
        target = item["target"]
        if self.preprocessor_text:
            target = self.preprocessor_text(target)
        if self.tokenizer:
            ids = self.tokenizer.encode(target)
            text = torch.tensor(ids, dtype=torch.int64)
        else:
            ids = target
            text = ids
        text_lengths = torch.tensor([len(ids)], dtype=torch.int32)

        out.update(
            {
                "speech": speech[0, :, :],
                "speech_lengths": speech_lengths,
                "text": text,
                "text_lengths": text_lengths,
            }
        )
        return out

