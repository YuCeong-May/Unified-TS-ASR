# -*- coding: utf-8 -*-
"""实验4（V4）：DAE-TSE / KCE 的关键词**随机片段**采样数据集。

复刻 `kce/src/kce/data/factory.py` 的三个函数：
`sample_keyword_from_whole_label`（从整条转写里随机截连续片段作线索）、
`inject_special_token`（线索两端加 `psok`/`peok`，且 `with_trans` 时在 CTC 目标的片段
原位插入同一对括号）、
`make_mix_wav` / `shift_wav`（可选的 batch 内干扰混音）。

与实验3（`datasets_char_cue_tsasr.AudioDatasetCharCueTS`）的区别，也是 V4 的**全部**
设计要点：
  - V3 的线索恒为 `target` 的字面前缀（唤醒词），任务不用线索也完全可解，
    cross-attention 从未收到「定位目标」的梯度；实测线索反而把 CER 从 10.04% 打到 18.02%。
  - V4 的线索是转写里的随机连续片段，且 CTC 目标必须在该片段**原位**吐出
    `psok … peok`。模型不知道括号该打在哪，除非真的定位了线索。
  - **`mix_conf` 必须开、且 `prob: 1.0`。** 这是 2026-08-08 复盘出的关键：DAE-TSE 是
    100% 两说话人混音（`speech_interference.use: True` 且**没有 `prob` 字段**即概率 1），
    标签只含目标说话人。不混音时线索在信息上是冗余的——第一版 V4（`mix_conf: null`）
    实测 `bracket_hit` 真线索 100.0% / 无线索 0.27%，证明线索确实到达了模型、
    cross-attention 也工作，但 CER 31.12% vs 31.41% 毫无收益：**任务本身不需要线索。**

数据（2026-08-08 起改用 `exp/dae_v4_work/*.jsonl`，不再复用 V1 的 `exp/g2p_work`）：
  - 只要干净 TTS（`seg_*/base`），不要离线加噪的 `seg_*/aug`——它会和在线混音叠成两层噪声。
  - 唤醒词段可按 `cmd_start` 时间戳在**加载时**裁掉（`trim_wake_word: true`），不重切音频
    存盘；`false` 时整条音频进模型、并把 `kw_text` 补回 target 前缀，TTS 行就与
    aishell/wenetspeech 完全同构（整条音频 + 整条转写 + 转写里的随机片段当线索）。
  - 混入 aishell1 / aishell2 / wenetspeech：只有 TTS 时 batch 内混音的两条都带唤醒词、
    且唤醒词只有 20 种，线索会失去区分性；开放域干扰源绝大多数不含唤醒词，
    「哪一路是目标」只能靠线索判断——这正是 DAE-TSE 的任务设置。
  - 构造脚本 `script/data/make_v4_dae_jsonl.py`，schema 是两种模式的超集。

结构上的一个刻意选择：`__getitem__` 返回**波形**而不是 fbank，`extract_fbank` 搬到
`collator` 里批量做。map-style + batch_sampler 时 `collate_fn` 就在 worker 进程内执行，
CPU 成本不变，而这是 batch 内混音唯一不需要第二次 BOS 读的结构。

线索 id 空间：直接在 `tokenizer.encode(target)` 的 **id 序列**上截片段，而不是先切字符
再编码——这样线索 id 与 CTC 目标 id 天然对齐，不依赖 `CharTokenizer` 的分词细节。
但**片段的边界与长度按「单位」算**（2026-08-12 改）：一个汉字 = 1 个单位，一个空格分隔的
拉丁词 = 1 个单位，所以 `hi colmo` 是 2 个单位而不是 7 个 token。见 `_encode_units`。

BOS 读取的健壮化（空 body 判定、指数退避重试、worker 内线程池并行）原本写在本文件里，
已抽到 `bos_robust_read.BosRobustReadMixin` 供 V1/V2/V3 共用，本类只是继承它。
"""
# =============================================================================
# Adapted from the DAE-TSE / KCE project:
#   repo   : https://github.com/GnafiY/DAE-TSE
#   branch : cursor/cloud-agent-1784704881858-cfrvr
#   source : kce/src/kce/data/factory.py
# All credit to the original authors.
# =============================================================================
import json
import logging
import random
import re
from array import array

import torch

from funasr.datasets.audio_datasets.bos_robust_read import BosRobustReadMixin
from funasr.datasets.audio_datasets.datasets import AudioDataset
from funasr.register import tables
from funasr.utils.load_utils import extract_fbank

# 线索长度的计数单位（见 `AudioDatasetKwsSpan._encode_units`）：
# 一个汉字算一个单位，一串连续的非汉字（拉丁词/数字）算一个单位。
_CJK_CHAR = re.compile(r"[\u4e00-\u9fa5]")
_UNIT_PIECE = re.compile(r"[\u4e00-\u9fa5]|[^\u4e00-\u9fa5]+")

# \u9010\u5355\u4f4d\u5bf9\u9f50\u53ef\u80fd\u5b58\u653e\u7684\u5b57\u6bb5\u540d\uff0c\u6309\u53ef\u4fe1\u5ea6\u4ece\u9ad8\u5230\u4f4e\uff08`forced_align` \u81ea\u5e26\u5355\u4f4d\u6587\u672c\u3001\u80fd\u4e0e
# `target` \u9010\u5b57\u6838\u5bf9\uff1b`timestamp` \u662f\u5bf9\u9f50\u6a21\u578b\u7684\u539f\u59cb\u8f93\u51fa\uff0c\u6ca1\u6709\u6587\u672c\uff0c\u53ea\u80fd\u6309\u4f4d\u5bf9\u9f50\uff09\u3002
# \u987a\u5e8f\u53ea\u51b3\u5b9a\u300c\u5148\u8bd5\u54ea\u4e2a\u300d\uff0c\u53d6\u7684\u662f**\u7b2c\u4e00\u4e2a\u80fd\u89e3\u6790\u6210\u529f**\u7684\uff0c\u89c1 `IndexDSJsonlKwsSpanV4`\u3002
_ALIGN_FIELDS = ("forced_align", "alignment", "char_timestamps", "timestamp")


def _split_units(text):
    """\u628a\u8f6c\u5199\u5207\u6210**\u5355\u4f4d**\u5e8f\u5217\uff08\u7eaf\u6587\u672c\u53e3\u5f84\uff0c\u4e0d\u4f9d\u8d56 tokenizer\uff09\u3002

    \u4e0e `AudioDatasetKwsSpan._encode_units` \u7684\u5207\u6cd5\u4e00\u81f4\uff1a\u5148\u6309\u7a7a\u683c\u5206\u8bcd\uff0c\u4e0d\u542b\u6c49\u5b57\u7684\u8bcd
    \uff08\u62c9\u4e01\u8bcd\u3001\u6570\u5b57\u4e32\uff09\u6574\u4f53\u7b97\u4e00\u4e2a\u5355\u4f4d\uff0c\u542b\u6c49\u5b57\u7684\u8bcd\u518d\u6309\u300c\u4e00\u4e2a\u6c49\u5b57 / \u4e00\u4e32\u8fde\u7eed\u975e\u6c49\u5b57\u300d\u5207\u3002
    \u552f\u4e00\u5dee\u522b\u662f `_encode_units` \u8fd8\u6709\u4e00\u6761 `len(tokens) != len(word)` \u7684\u56de\u9000\uff08`ip\u7535\u8bdd`
    \u8fd9\u7c7b\u8bcd\u8868\u91cc\u771f\u6709\u7684\u4e2d\u897f\u6df7\u6392\u8bcd\u4f1a\u6574\u8bcd\u7b97\u4e00\u4e2a\u5355\u4f4d\uff09\uff0c\u8fd9\u91cc\u6ca1\u6709 tokenizer \u590d\u73b0\u4e0d\u51fa\u6765\uff1b
    \u771f\u649e\u4e0a\u65f6\u5355\u4f4d\u4e2a\u6570\u5bf9\u4e0d\u4e0a\uff0c\u8c03\u7528\u65b9\u6309\u300c\u5bf9\u9f50\u4e0d\u53ef\u7528\u300d\u4e22\u6389\u90a3\u4e00\u884c\uff0c\u4e0d\u4f1a\u9519\u4f4d\u3002

    Args:
        text: \u8f6c\u5199\u6587\u672c\u3002

    Returns:
        list[str]\uff1a\u5355\u4f4d\u6587\u672c\uff1b\u7a7a\u683c\u53ea\u662f\u8bcd\u7684\u5206\u9694\uff0c\u4e0d\u8ba1\u5165\u4efb\u4f55\u5355\u4f4d\u3002
    """
    units = []
    for word in str(text).split():
        if not _CJK_CHAR.search(word):
            units.append(word)
            continue
        units.extend(m.group() for m in _UNIT_PIECE.finditer(word))
    return units


@tables.register("index_ds_classes", "IndexDSJsonlKwsSpanV4")
class IndexDSJsonlKwsSpanV4(torch.utils.data.Dataset):
    """V4 专用索引：单遍解析 `exp/dae_v4_work/*.jsonl`，保留唤醒词时间戳与语料名。

    不复用 `IndexDSJsonlRankFull`：它只保留固定 5 个字段，补回 `cmd_start`/`kw_text`
    要像 `IndexDSJsonlPhoneTS` 那样再扫一遍 jsonl，而 V4 的 train.jsonl 有约 340 万行，
    第二遍 json 解析要多花好几分钟。这里直接一遍解析完。

    `source_len` 的口径（由 `make_v4_dae_jsonl.py` 保证）：
      - TTS 行是 **`cmd_start` 之后**那段的 10ms 帧数（V2 强制对齐时就这么存的）；
      - 开放域行是整条音频的帧数，且 `cmd_start == 0.0`。
    所以 `trim_wake_word: true` 时 `source_len` 恰好就是喂进模型的长度，batch sampler
    的 token 预算直接可用；`trim_wake_word: false` 时唤醒词那段也进模型，必须把
    `cmd_start` 折算成帧数加回去（`target_len` 同理要加 `len(kw_text)`，因为
    `AudioDatasetKwsSpan` 会把唤醒词补回 target），否则 sampler 会低估 TTS 行
    约 100 帧（约 1 s）而超预算。

    Args:
        path: V4 jsonl 路径（也支持「每行一个 jsonl 路径」的列表文件）。
        **kwargs: 取 `max/min_source_length`、`max/min_target_length`、
            `max_token_length`、`trim_wake_word`、`data_split_num`/`data_split_i`。
    """

    def __init__(self, path: str, **kwargs):
        super().__init__()
        max_src = int(kwargs.get("max_source_length", 2048))
        min_src = int(kwargs.get("min_source_length", 0))
        max_tgt = int(kwargs.get("max_target_length", 2048))
        min_tgt = int(kwargs.get("min_target_length", 0))
        max_tok = int(kwargs.get("max_token_length", 2200))
        trim = bool(kwargs.get("trim_wake_word", True))
        streaming_wake_word = bool(kwargs.get("streaming_wake_word", False))
        speaker_field = str(kwargs.get("speaker_field", "speaker"))

        contents = []
        n_raw = 0
        per_corpus = {}
        n_no_align = n_short = n_mismatch = n_wide_unit = 0
        used_fields = set()
        for file_json in self._resolve_file_list(path, **kwargs):
            with open(file_json.strip(), encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    n_raw += 1
                    units = None
                    if streaming_wake_word:
                        # 多个候选字段时**逐个试到能用为止**，而不是取第一个存在的：
                        # 同一行可能既有 `timestamp`（对齐模型的原始输出，可能与当前
                        # `target` 不同源）又有 `forced_align`（带文本、经过校验），
                        # 硬按固定优先级取第一个存在的，会让整份数据因为一个坏字段全灭。
                        cands = [(k, d[k]) for k in _ALIGN_FIELDS if d.get(k)]
                        if not cands:
                            # 开放域行没有唤醒词及其真实边界，不能伪造两阶段样本。
                            n_no_align += 1
                            continue
                        used = None
                        for name, raw in cands:
                            units = self._pack_units(raw, d.get("target", ""))
                            if units is not None:
                                used = name
                                break
                        if units is None:
                            n_mismatch += 1
                            if n_mismatch == 1:
                                # 整批失败时（0/N 通过）原先只剩一句「0 行通过」，看不出
                                # 对齐字段长什么样。把第一条打出来，schema 一眼可辨。
                                logging.warning(
                                    "[IndexDSJsonlKwsSpanV4] %s 首条对齐不可用："
                                    "source=%s target=%r（%d 个单位）；试过的字段 %s。"
                                    "支持的 schema 见 _pack_units：dict+秒 / [文本, 秒] / "
                                    "[start_ms, end_ms] int 毫秒 / [start_ms, end_ms, 文本]",
                                    file_json.strip(), d.get("source"),
                                    d.get("target", ""),
                                    len(_split_units(d.get("target", ""))),
                                    "；".join(
                                        f"{k}: type={type(v).__name__} len={len(v)} "
                                        f"前 3 项={v[:3]!r}" for k, v in cands
                                    ),
                                )
                            continue
                        used_fields.add(used)
                        ends, char_lens = units
                        # kws_len <= full_len - 1：唤醒段之后必须还留有指令段。
                        if len(ends) < 2:
                            n_short += 1
                            continue
                        if max(char_lens) > 255:
                            n_wide_unit += 1
                            continue
                    cmd_start = float(d.get("cmd_start", 0.0))
                    source_len = int(d.get("source_len", 1))
                    target_len = int(d.get("target_len", len(d.get("target", ""))))
                    if streaming_wake_word:
                        # 整条音频与完整 target 都进模型，jsonl 原值就是口径；
                        # target 额外带 psok/peok 两个定位括号。
                        target_len += 2
                    elif not trim:
                        # 不裁剪 -> 唤醒词那段也进模型，音频长度和 target 都要加回来
                        source_len += int(round(cmd_start * 100))
                        target_len += len(d.get("kw_text", ""))
                    if not (min_src <= source_len <= max_src):
                        continue
                    if not (min_tgt <= target_len <= max_tgt):
                        continue
                    if source_len + target_len > max_tok:
                        continue
                    corpus = d.get("corpus", "other")
                    speaker = d.get(speaker_field, d.get("speaker_id"))
                    if speaker in (None, ""):
                        raise ValueError(
                            f"Missing {speaker_field!r} at {file_json}; "
                            "same-speaker mixing cannot be excluded safely."
                        )
                    per_corpus[corpus] = per_corpus.get(corpus, 0) + 1
                    item = {
                        "source": d["source"],
                        "target": d["target"],
                        "source_len": source_len,
                        "target_len": target_len,
                        "cmd_start": cmd_start,
                        "kw_text": d.get("kw_text", ""),
                        "corpus": corpus,
                        # Speaker ids are corpus-scoped: different corpora can both use
                        # simple ids such as "0001" for unrelated people.
                        "speaker": f"{corpus}:{speaker}",
                    }
                    if units is not None:
                        # 紧凑存储：1.1M 行 × ~15 个 3 键 dict 在 num_workers=8 下内存吃不消。
                        item["unit_ends"] = units[0]
                        item["unit_char_lens"] = units[1]
                    contents.append(item)
        self.contents = contents
        logging.info(
            "[IndexDSJsonlKwsSpanV4] %d/%d 行通过过滤（trim_wake_word=%s, streaming=%s），"
            "跳过：无对齐 %d / 单位数<2 %d / 对齐不可用（文本或单位数对不上）%d / 超长单位 %d；"
            "对齐字段 %s；按语料 %s",
            len(contents), n_raw, trim, streaming_wake_word,
            n_no_align, n_short, n_mismatch, n_wide_unit,
            sorted(used_fields) or "-", sorted(per_corpus.items()),
        )

    @staticmethod
    def _pack_units(raw, target):
        """把 forced_align 压成 ``(array('f') 每单位结束秒, bytes 每单位字符数)``。

        条目是**单位**（一个汉字，或一个拉丁词），不是字符。按条目类型分派三种 schema：

        - ``{"text": "打", "end_time": 0.56}``：本项目 ``*_forced_align.jsonl`` 的
          `forced_align`，文本齐全、时间戳是**秒**；
        - ``["打", 0.56]``：文本在前，时间戳是**秒**；
        - ``[320, 560]``：FunASR 时间戳模型的原始输出（`run_fa.py` 落盘的 `timestamp`，
          见 `funasr/utils/timestamp_tools.ts_prediction_lfr6_standard` 的返回值），
          只有 ``[start_ms, end_ms]``、**没有文本**，而且是 **int 毫秒**；
          ``[start_ms, end_ms, text]`` 三元组按同样口径收。
          这一类按位与 target 的单位序列对齐：单位文本从 target 反推（`_split_units`，
          口径同 `AudioDatasetKwsSpan._encode_units`），所以单位个数必须逐位相等。

        毫秒 schema 要求数值是 ``int``（FunASR 原样输出就是 ``int(timestamp * 1000)``）：
        浮点一律拒收，免得把一份「秒」的数据当成毫秒、边界整体缩到千分之一。
        也正因为两种时间单位混在同一函数里，返回值统一折算成**秒**——下游
        `_sample_streaming_wake` 的返回值和 `cmd_start` 都是秒。

        任一条目不符合上述 schema、`concat(单位文本) != 去空白 target`、或毫秒 schema
        的单位个数对不上时返回 None，由调用方跳过。这是契约守卫：宁可丢这一行，也不能
        让 `unit_ends` 与 target 错位——错位会让「唤醒段音频」与「唤醒文本线索」变成
        两段不同的话，而 loss 照样下降。

        Args:
            raw: jsonl 里的对齐字段（list）。
            target: 该行的转写。

        Returns:
            (array('f') 每单位结束秒, bytes 每单位字符数) 或 None。
        """
        ends = array("f")
        char_lens = bytearray()
        texts = []
        n_ms = 0
        for entry in raw:
            if isinstance(entry, dict):
                text = str(entry.get("text", ""))
                end, scale = entry.get("end_time", entry.get("end")), 1.0
            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                head, tail = entry
                if isinstance(head, str):
                    text, end, scale = head, tail, 1.0
                elif isinstance(head, int) and isinstance(tail, int):
                    # FunASR 时间戳：[start_ms, end_ms]，文本不在条目里
                    if head > tail:
                        return None
                    text, end, scale, n_ms = "", tail, 1e-3, n_ms + 1
                else:
                    return None
            elif (isinstance(entry, (list, tuple)) and len(entry) == 3
                  and isinstance(entry[-1], str) and isinstance(entry[0], int)
                  and isinstance(entry[1], int)):
                # [start_ms, end_ms, text]
                if entry[0] > entry[1]:
                    return None
                text, end, scale, n_ms = entry[-1], entry[1], 1e-3, n_ms + 1
            else:
                return None
            if end is None:
                return None
            ends.append(float(end) * scale)
            char_lens.append(min(len(text), 255))
            texts.append(text)
        if n_ms:
            # 无文本的毫秒行：整行必须是同一种 schema，且时间戳与 target 的单位
            # 一一对应。若某条坏时间戳被丢掉（对齐模型对超长 token 会插 <sil>），
            # 个数就对不上——那时整行作废，绝不错位。
            units = _split_units(target)
            if n_ms != len(raw) or len(ends) != len(units):
                return None
            if any(b < a for a, b in zip(ends, ends[1:])):
                return None
            return ends, bytes(min(len(u), 255) for u in units)
        if "".join(texts) != "".join(str(target).split()):
            return None
        return ends, bytes(char_lens)

    @staticmethod
    def _resolve_file_list(path: str, **kwargs):
        """单 jsonl 直接返回；列表文件按 `data_split_num`/`data_split_i` 切片。"""
        if path.endswith(".jsonl") or path.endswith(".json"):
            return [path]
        num = int(kwargs.get("data_split_num", 1))
        idx = int(kwargs.get("data_split_i", 0))
        if not kwargs.get("is_training", True):
            num, idx = 1, 0
        with open(path, encoding="utf-8") as fin:
            all_files = fin.readlines()
        per = (len(all_files) - 1) // num + 1
        return all_files[idx * per: (idx + 1) * per]

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, index):
        return self.contents[index]

    def get_source_len(self, data_dict):
        """喂进模型那段音频的 10ms 帧数（构造时已按 `trim_wake_word` 折算好）。"""
        return data_dict.get("source_len", 1)

    def get_target_len(self, data_dict):
        """CTC 目标字数（不含 `with_trans` 的 2 个括号 token）。"""
        return data_dict.get("target_len", 0)


@tables.register("dataset_classes", "AudioDatasetKwsSpan")
class AudioDatasetKwsSpan(BosRobustReadMixin, AudioDataset):
    """随机片段关键词线索 + `psok/peok` 括号标记的 CTC 数据集。

    Args:
        path: jsonl 路径。V4 用 `exp/dae_v4_work/*.jsonl`（干净 TTS + 开放域语料，
            带 `cmd_start`/`kw_text`/`corpus`）；也兼容 V1 的 `exp/g2p_work/*.jsonl`。
        index_ds: 索引类。V4 用 `IndexDSJsonlKwsSpanV4`；不传时沿用旧默认
            `IndexDSJsonlPhoneTS`（它把 `keyword_text` 补回来）。
        trim_wake_word: 按时间戳裁掉唤醒词段，音频取 `wav[cmd_start:]`。
            True = 严格复刻 DAE-TSE（音频里没有唤醒词，线索只能是转写的随机片段，
            所以此时 `kw_prefix_prob` 会被强制视作 0）；
            False = 整条音频进模型，且**把 `kw_text` 补回 target 前缀**，于是 TTS 行与
            aishell/wenetspeech 完全同构（整条音频 + 整条转写），线索是转写里的随机片段。
            V4 的 jsonl `target` 不含唤醒词，这个前缀只在 False 时补；V1 schema
            （`keyword_text`）的 target 本来就含唤醒词全文，不补。
        frontend: WavFrontend，在 `collator` 里批量算 fbank。
        tokenizer: CharTokenizer，必需。
        keyword_length_range: 随机片段的长度范围，单位是**汉字/拉丁词**（不是 token，
            见 `_encode_units`）：`hi colmo` 算 2，`我是 luna` 算 3。DAE-TSE 是
            [2, 6] 个**词**，中文按时长折算成 [3, 10] 个字（英文 2-6 词约 0.6-2.0 s，
            中文约 0.2 s/字）。纯中文语料下这个口径与旧的 token 口径**完全等价**。
        continuous: 只支持 True（连续片段），与原实现一致。
        positive_prob: 线索确实出现在本条音频里的概率。**默认 1.0，严格复刻**；
            <1 时按 DAE-TSE 的做法从别的样本借线索作负样本。
        with_trans: 正样本在 CTC 目标的片段原位插入 `psok … peok`（定位监督的来源）。
            注意送进 cue encoder 的线索**无条件**加这对括号，与 `with_trans` 无关，
            这是 `inject_special_token` 的行为。
        use_filler_label_for_neg: 负样本的 CTC 目标换成 `<no_kws/> × L`（`L` = 整条转写
            的 token 数；官方 DAE-TSE `kce/src/kce/data/factory.py:671-674` 是 `L // 3`，
            本实现按 `L`，见 `_inject` 的说明）。默认 False = 沿用 4-3/4-4 的
            口径：负样本仍要转写整句，线索只影响括号。True 时任务变成「线索没出现在音频
            里 -> 不出内容，只吐拒识标签」，即部署语义（KWS 没命中就不该出字）。
            官方评测侧还有配套折叠（`kce/src/kce/eval/decoder.py:174`
            `map_filler_label_for_neg`）：负样本的 ref 无条件改写成**单个**拒识符，
            hyp 只要含拒识符或全 blank 也折叠成单个 —— 即训练吐 `L` 个、
            评测只判「有没有拒识符」，长度在指标上不参与。
            ⚠ **负样本比例（`1 - positive_prob`）此时是个危险旋钮**：那部分样本的转写监督
            整句被占位符替换，比例过高会伤 ASR 并给 blank 路径过强先验。
        kw_prefix_prob: 用真唤醒词前缀而非随机片段的概率。`trim_wake_word=True` 时
            无效（唤醒词已不在音频里，拿它当线索就是无声学依据的负样本）；
            `False` 时唤醒词已被补回 target 前缀，此路径可用（`kw_pos = 0`，括号合法），
            但要「TTS 与开放域完全同构」就该保持 0。
        psok_id / peok_id: 关键词起止的**专用** token（`conf/tokens_kws8408.json` 里的
            `<psok>` / `<peok>` = 8404 / 8405）。曾复用 `<s>` / `</s>`(1/2)，但那两个是
            正常转写的句首句尾符，语义会混；现在词表扩到 8408，两处含义分开。
        no_kws_id: **拒识标签**「线索没出现在这条音频里」用的 token
            （`<no_kws/>` = 8406）。2026-08-22 从 `punk_id`(=`<unk>`) 换过来：`<unk>` 同时
            承载「转写里有词表外的字」和「该拒识」两种含义，混在一个 id 上既没法在评测里
            分开统计，也让 CTC 把 OOV 的声学证据和拒识的**无**声学证据学到同一行输出上。
            `<no_tgt_spk/>`(8407) 是它在说话人维度的对应物，由
            `datasets_speaker_online_enroll.py` 使用；两个拒识原因分开，才能分别看漏检。
        punk_id: 负样本**线索**连续取样失败时的兜底占位符（仍是 `<unk>` = 8403）。
            注意它现在只用在线索侧、不再进 CTC 标签：拒识标签走 `no_kws_id`。
            这里保留 `<unk>` 是对的——那个位置要的语义是「一个无意义、不可能是任何真实
            转写子序列的符号」，而不是「该拒识」。
        mix_conf: batch 内干扰混音配置，None 表示不混（run A）。见 `_mix_batch`。
            `segment_prob > 0` 时，有唤醒/指令边界（`cmd_start > 0` 且
            `trim_wake_word: false`）的行会走**两段各自独立**的干扰源与幅度权重，
            即唤醒段与指令段噪声不同、信噪比不同。见 `_mix_segment`。
        spk2id: 说话人 -> 类别 id 的 json 映射路径，给模型侧的说话人分类辅助头
            （`model_conf.spk_ce_conf`，见 `funasr/models/tsasr/spk_ce_head.py`）提供
            标签。None（默认）= 不发 `spk_id` 列，行为与改动前逐位相同。
            用 `script/data/make_spk2id.py` 从**训练** jsonl 生成，train 与 valid
            **必须指同一份文件**，否则辅助头是在两个不同的标签空间上打分。
            不在映射里的说话人 -> -1 -> 在 CE 里走 ignore_index。aishell 的 dev
            说话人与 train 不相交，所以 valid 的 `acc_spk` 恒为 0 是**预期现象**，
            选 ckpt 仍然只看 CTC 的 `acc`。
            映射的 key 必须是索引里的 `f"{corpus}:{speaker}"` 形式（aishell1 与
            aishell2 的说话人 id 都是裸数字，不加语料前缀会撞号），`make_spk2id.py`
            用的就是这个口径。

    BOS 读取（重试退避 + worker 内并行）来自 `BosRobustReadMixin`，
    开关 `bos_workers` / `bos_read_retry` / `bos_retry_backoff` 见 `bos_robust_read.py`。
    """

    # 负样本线索的重采样次数上限（避开混合语音里已出现的文本）
    _NEG_CUE_TRIES = 16

    def __init__(
        self,
        path,
        index_ds: str = None,
        frontend=None,
        tokenizer=None,
        trim_wake_word: bool = False,
        keyword_length_range=(3, 10),
        continuous: bool = True,
        positive_prob: float = 1.0,
        with_trans: bool = True,
        use_filler_label_for_neg: bool = False,
        kw_prefix_prob: float = 0.5,
        psok_id: int = 8404,
        peok_id: int = 8405,
        punk_id: int = 8403,
        no_kws_id: int = 8406,
        mix_conf: dict = None,
        spk2id: str = None,
        streaming_wake_word: bool = False,
        wake_word_length_range=(2, 5),
        **kwargs,
    ):
        if index_ds is None:
            index_ds = "IndexDSJsonlPhoneTS"
        super().__init__(
            path,
            index_ds=index_ds,
            frontend=frontend,
            tokenizer=tokenizer,
            trim_wake_word=trim_wake_word,
            # 必须显式转发：它在本类是具名参数，不放进 kwargs 就到不了 index，
            # 索引便不会解析 forced_align（unit_ends/unit_char_lens 全缺），
            # 直到第一个 batch 取数时才 KeyError。
            streaming_wake_word=streaming_wake_word,
            **kwargs,
        )
        self._init_bos_robust_read(**kwargs)
        if self.tokenizer is None:
            raise ValueError("AudioDatasetKwsSpan 需要 tokenizer 来编码 CTC 目标与线索")
        if not continuous:
            raise NotImplementedError("AudioDatasetKwsSpan 只支持 continuous=True")
        if frontend is None:
            raise ValueError("AudioDatasetKwsSpan 需要 frontend（fbank 在 collator 里算）")

        self.trim_wake_word = bool(trim_wake_word)
        self.streaming_wake_word = bool(streaming_wake_word)
        self.wake_word_length_range = tuple(int(x) for x in wake_word_length_range)
        if self.streaming_wake_word:
            if len(self.wake_word_length_range) != 2 or not (
                2 <= self.wake_word_length_range[0] <= self.wake_word_length_range[1] <= 5
            ):
                raise ValueError("wake_word_length_range must be [2, 5] or a sub-range")
            if self.trim_wake_word:
                raise ValueError("streaming_wake_word requires trim_wake_word=false")
        self.keyword_length_range = list(keyword_length_range)
        self.positive_prob = float(positive_prob)
        self.with_trans = bool(with_trans)
        self.use_filler_label_for_neg = bool(use_filler_label_for_neg)
        # 裁掉唤醒词后它已不在音频里，用它当线索等于无声学依据的负样本，强制关掉
        self.kw_prefix_prob = 0.0 if self.trim_wake_word else float(kw_prefix_prob)
        if self.trim_wake_word and float(kw_prefix_prob) > 0:
            logging.info(
                "[AudioDatasetKwsSpan] trim_wake_word=True，kw_prefix_prob %.2f -> 0.0",
                float(kw_prefix_prob),
            )
        self.psok_id = int(psok_id)
        self.peok_id = int(peok_id)
        self.punk_id = int(punk_id)
        self.no_kws_id = int(no_kws_id)
        # 词表没换就报错，而不是等到 CTCLoss 里变成一句 CUDA assert：8406 在
        # tokens_kws8406.json 下越界，且 filler 路径只在负样本上走，可能跑几百步才炸。
        if self.use_filler_label_for_neg:
            n_vocab = len(getattr(self.tokenizer, "token_list", []) or [])
            if n_vocab and not 0 <= self.no_kws_id < n_vocab:
                raise ValueError(
                    f"no_kws_id={self.no_kws_id} 超出 {n_vocab} 词表范围；"
                    "把 tokenizer_conf.token_list 指到 conf/tokens_kws8408.json"
                )
            if n_vocab and self.tokenizer.token_list[self.no_kws_id] != "<no_kws/>":
                raise ValueError(
                    f"token_list[{self.no_kws_id}] 是 "
                    f"{self.tokenizer.token_list[self.no_kws_id]!r}，应为 '<no_kws/>'"
                )
        self.mix_conf = mix_conf or None
        self._shuffle_val_for_mix(kwargs)
        # 唤醒词种类极少（本项目 20 种 / 151 万条），缓存避免重复分词
        self._kw_cache = {}
        # 负样本线索连续 `_NEG_CUE_TRIES` 次都撞上混合语音的文本时的兜底次数，
        # 只用于限频告警（每 1000 次打一条），不影响训练
        self._neg_cue_fallback = 0
        if len(self.index_ds) == 0:
            # 空索引不能放过：下游第一个报错是 `_load_spk2id` 的「说话人完全不相交
            # （0/0 命中）」，把「过滤把整份数据滤光了」误诊成「spk2id 配错了」。
            # 真因一定在上面那行 `[IndexDSJsonlKwsSpanV4] 0/N 行通过过滤` 的计数里。
            raise ValueError(
                f"{path} 过滤后一条都不剩；看上面那行 "
                "`[IndexDSJsonlKwsSpanV4] 0/N 行通过过滤` 的分项计数定位原因"
                "（对齐不可用 -> 紧跟的 WARNING 打了首条的对齐 schema）"
            )
        self.spk2id = self._load_spk2id(spk2id)

    def _shuffle_val_for_mix(self, kwargs):
        """打散验证集的取数顺序，否则 `mix_conf` 在 dev 上等于没开。

        dev 的 jsonl 按说话人排序（本项目相邻两行 99.8% 同人，2.2 万行只有 45 个说话人），
        而验证集 sampler 是 `is_training=False` 构造的，`shuffle = shuffle and is_training`
        （`samplers.py:86`）恒为假 -> 顺序取数 -> 一个 microbatch 里几乎没有第二个说话人
        -> `_pick_partner` 恒返回 None -> 样本**静默**保持干净，dev 与 train 的输入分布
        完全不是一回事。这里用固定种子的置换把说话人打散，让 dev 也吃到随机混音。

        用固定种子而不是随机置换：多次验证的样本组成、成批方式保持一致，指标才能跨
        checkpoint 比较。只在本份数据是验证集且开了混音时生效，训练集顺序不动。

        Args:
            kwargs: `AudioDatasetKwsSpan.__init__` 的剩余参数（含 `is_training`）。
        """
        if kwargs.get("is_training", True) or not self.mix_conf:
            return
        seed = int(kwargs.get("val_shuffle_seed", 0))
        random.Random(seed).shuffle(self.index_ds.contents)
        logging.info(
            "[AudioDatasetKwsSpan] 验证集顺序按 seed=%d 打散 %d 条（为让混音能找到"
            "不同说话人的干扰源；训练集不受影响）", seed, len(self.index_ds.contents),
        )

    def _load_spk2id(self, path):
        """读 `spk2id.json` 并做三项校验；`path` 为空时返回 None（= 不发 `spk_id`）。

        校验的都是「会让训练在几百步之后才炸成一句 CUDA assert」的错误：
        id 必须是连续的 `0..N-1`（`nn.Linear` 的 odim 就是 N，留空洞等于白占类别，
        缺号则某个 id 越界）；映射对本份数据的覆盖率为 0 时直接报错，而不是安静地
        让每个 batch 都全 ignore、`loss_spk_ce` 恒为 0。

        Args:
            path: json 路径，形如 `{"aishell1:S0002": 0, ...}`。

        Returns:
            dict[str, int] 或 None。
        """
        if not path:
            return None
        with open(path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"spk2id={path} 不是一个非空 dict")
        mapping = {str(k): int(v) for k, v in mapping.items()}
        ids = sorted(mapping.values())
        if ids != list(range(len(ids))):
            raise ValueError(
                f"spk2id={path} 的 id 不是连续的 0..{len(ids) - 1}；"
                "请用 script/data/make_spk2id.py 重新生成"
            )
        known = sum(1 for i in range(len(self.index_ds)) if self.index_ds[i]["speaker"] in mapping)
        total = len(self.index_ds)
        logging.info(
            "[%s] spk2id=%s：%d 个说话人；本份数据 %d/%d 条命中映射"
            "（未命中 -> -1 -> CE 忽略）。model_conf.spk_ce_conf.num_speakers 必须填 %d",
            type(self).__name__,
            path,
            len(mapping),
            known,
            total,
            len(mapping),
        )
        if known == 0:
            raise ValueError(
                f"spk2id={path} 与这份数据的说话人完全不相交（0/{total} 命中）；"
                "多半是 key 少了 `corpus:` 前缀，或者映射来自另一份 jsonl"
            )
        return mapping

    # ---------------------------------------------------------------- 采样
    def _encode_keyword(self, text: str):
        """把唤醒词文本编码成 id 列表（按文本缓存）。

        Args:
            text: 唤醒词文本。

        Returns:
            list[int]：char id 列表。
        """
        ids = self._kw_cache.get(text)
        if ids is None:
            ids = list(self.tokenizer.encode(text))
            self._kw_cache[text] = ids
        return ids

    def _encode_units(self, text: str):
        """把文本编码成 id 序列，同时给出「单位 -> token 区间」的切分。

        **单位** = 一个汉字，或一个空格分隔的拉丁词/数字串。线索长度按单位计数，
        于是 `hi colmo` 是 2 个单位（而按 token 是 7 个：`colmo` 不在 seg_dict，
        退化成 5 个单字母），`我是 luna` 是 3 个单位。这么改有两个目的：
          1. `keyword_length_range` 在中英混说数据上口径一致——中文数字、英文数词；
          2. 片段边界永远落在词边界上，不会截出 `un@@` 这种半个词的碎片当线索。
        纯中文文本下每个单位恰好 1 个 token，与旧的 token 口径**逐位等价**。

        逐词调 `text2tokens` 是安全的：`seg_tokenize` 本来就按空格切开、**逐词独立**
        查 seg_dict，所以拼起来与整句编码逐元素相同。含汉字的词几乎必然不在 seg_dict
        里（312,968 条里只有 13 条多字含汉字词条，全是 `ip电话`/`pc机` 这类中西混排），
        于是走逐字回退、token 与字符 1:1，可以直接按字符切单位；那 13 条对不上，
        用 `len(tokens) != len(word)` 兜住，整词算一个单位。

        Args:
            text: 转写文本。

        Returns:
            (ids, spans)：`ids` 与 `self.tokenizer.encode(text)` 逐元素相等；
            `spans` 是 list[(start, end)]，按序无重叠地覆盖整个 `ids`。
        """
        ids, spans = [], []
        for word in text.strip().split(" "):
            wid = self.tokenizer.tokens2ids(self.tokenizer.text2tokens(word))
            base = len(ids)
            ids.extend(wid)
            if not _CJK_CHAR.search(word) or len(wid) != len(word):
                spans.append((base, base + len(wid)))
                continue
            off = base
            for m in _UNIT_PIECE.finditer(word):
                spans.append((off, off + len(m.group())))
                off += len(m.group())
        return ids, spans

    def _sample_span(self, spans):
        """在**单位**序列上随机截一段连续片段，复刻 `sample_keyword_from_whole_label`。

        Args:
            spans: `_encode_units` 给出的单位 -> token 区间。

        Returns:
            (kw_pos, kw_len)，仍是 **token** 口径的起点与长度，`_inject` 用它在 CTC
            目标里原位插括号。
        """
        n_unit = len(spans)
        if n_unit == 0:
            return 0, 0
        kw_len = min(random.randint(*self.keyword_length_range), n_unit)
        u0 = random.randint(0, n_unit - kw_len)
        start, end = spans[u0][0], spans[u0 + kw_len - 1][1]
        return start, end - start

    def _pick_keyword(self, ids, spans, keyword_text):
        """决定线索取「真唤醒词」还是「转写里的随机连续片段」。

        `kw_pos = -1` 表示线索**不是** `ids` 的子串（`trim_wake_word=True` 时 V4 的
        target 不含唤醒词），此时不能打 `psok/peok` 括号：括号是「线索在转写里的位置」的
        监督，线索不在转写里就没有位置可标。这半样本的定位监督来自混音——干扰源没有唤醒词，
        模型必须找到唤醒词才知道该转写哪一路。`trim_wake_word=False` 时唤醒词已被补回
        target 前缀，走这条路也会命中 `ids[:n] == kw_ids`。

        Args:
            ids: list[int]，整条转写的 id 序列。
            spans: `_encode_units` 给出的单位 -> token 区间。
            keyword_text: 该条的唤醒词文本，可能为空。

        Returns:
            (kw_ids, kw_pos, kw_len)。`kw_pos >= 0` 时 `kw_ids == ids[kw_pos:kw_pos+kw_len]`。
        """
        if keyword_text and random.random() < self.kw_prefix_prob:
            kw_ids = self._encode_keyword(keyword_text)
            n = len(kw_ids)
            if n > 0:
                # target 含唤醒词全文（V1 的 jsonl，或 trim_wake_word=False 补过前缀的
                # V4）-> 是前缀，可以打括号；否则 kw_pos = -1，不打括号
                if n <= len(ids) and ids[:n] == kw_ids:
                    return kw_ids, 0, n
                return list(kw_ids), -1, n
        kw_pos, kw_len = self._sample_span(spans)
        return ids[kw_pos: kw_pos + kw_len], kw_pos, kw_len

    @staticmethod
    def _contains(haystack, needle) -> bool:
        """`needle` 是否为 `haystack` 的连续子序列（都是 id 列表）。

        用于负样本线索的合法性判定：线索必须**不在**混合语音的任何一路转写里，
        否则「线索没出现」这个前提是假的，标签就是噪声。
        朴素扫描，两边长度分别约 20 / 2~5，成本可忽略。
        """
        n, m = len(haystack), len(needle)
        if m == 0 or m > n:
            return False
        head = needle[0]
        for s in range(n - m + 1):
            if haystack[s] == head and list(haystack[s: s + m]) == list(needle):
                return True
        return False

    def _negative_keyword(self, index, forbid_ids=()):
        """负样本线索：从别的样本里借一段（`positive_prob < 1` 时才会走到）。

        **必须避开 `forbid_ids`**（2026-08-11 修）：借来的片段只有 2~5 个 token，
        撞进本条转写、或撞进 batch 内混进来的干扰源转写的概率并不低（常用二字词），
        一旦撞上，这条「线索不在音频里」的负样本就是错标签——尤其在
        `use_filler_label_for_neg: true` 下，它会教模型把真正说出来的内容也丢掉。
        `__getitem__` 只能查本条自己的转写（配对还没发生），干扰源那一路由
        `collator` 在混音之后复查并按需重采样。

        Args:
            index: 当前样本下标，用于避开自己。
            forbid_ids: list[list[int]]，候选片段不允许是其中任何一条的子序列。

        Returns:
            list[int]：借来的线索 id 列表。
        """
        n = len(self.index_ds)
        for _ in range(self._NEG_CUE_TRIES):
            other = random.randrange(n)
            if other == index:
                continue
            other_ids, other_spans = self._encode_units(self.index_ds[other]["target"])
            if len(other_spans) < self.keyword_length_range[0]:
                continue
            pos, length = self._sample_span(other_spans)
            cand = other_ids[pos: pos + length]
            if any(self._contains(f, cand) for f in forbid_ids):
                continue
            return cand
        # 兜底：`punk` 单 token 当线索。它不是任何真实转写的子序列，所以「线索不在
        # 音频里」这个前提仍然成立，只是线索退化成一个无意义符号。
        self._neg_cue_fallback += 1
        if self._neg_cue_fallback % 1000 == 1:
            logging.warning(
                "[AudioDatasetKwsSpan] 负样本线索连续 %d 次撞上混合语音的转写，"
                "已退化为 punk 线索 %d 次",
                self._NEG_CUE_TRIES, self._neg_cue_fallback,
            )
        return [self.punk_id]

    def _wrap_cue(self, kw_ids):
        """给送进 cue encoder 的线索两端加 `psok`/`peok`，复刻 `inject_special_token`。

        原实现（`kce/src/kce/data/factory.py`）对 `new_keyword` 无条件 `insert(0, psok)` /
        `insert(len, peok)`，正负样本一视同仁；只有写进 CTC 标签那一步才看 `with_trans`
        和 `positive`。所以线索长度恒为「片段长度 + 2」。

        这两行不只是形式对齐：cross-attention 在线索维度上做 softmax，纯片段（4 字线索
        L=4）时每帧的注意力下界就是 0.25，没有任何一行能吸收「这一帧不匹配任何线索字」
        的质量；括号行正好当这个 sink。

        Args:
            kw_ids: list[int]，线索片段的 id 序列。

        Returns:
            list[int]：`[psok] + kw_ids + [peok]`。
        """
        return [self.psok_id] + list(kw_ids) + [self.peok_id]

    def _inject(self, ids, kw_pos, kw_len, positive):
        """复刻 `inject_special_token`：正样本在片段原位插入 `psok … peok`。

        负样本两种口径：
          - `use_filler_label_for_neg`：`<no_kws/> × L`（`L` = 整条转写的 token 数）。
            官方 DAE-TSE `factory.py:673` 的原式是 `L // 3`，那里 `L` 是音素数
            （3 音素≈1 音节≈200 ms），我们的 `L` 是汉字数（1 字≈1 音节），照搬 `//3`
            会让 filler 序列稀疏三倍，故按 `L` 给——单位换算之后这才是等价的。
            `max(..., 1)` 兜住空转写：长度 0 的目标会让 `_calc_cer_ctc` 的 ref 长度为 0
            （误报全额进分子不进分母、`acc` 可为负，而 `acc` 正是 ckpt 选择依据）。
            2026-08-22 起这个符号是 `<no_kws/>`(8406) 而不再是 `<unk>`(8403)：
            见类 docstring 的 `no_kws_id`。
          - 不开（4-3/4-4 的口径）：负样本照样转写整句，线索只影响括号。

        Args:
            ids: list[int]，整条转写的 id 序列。
            kw_pos: 片段起点，`-1` 表示线索不在 `ids` 里（不打括号）。
            kw_len: 片段长度。
            positive: 线索是否真的出现在本条音频里。

        Returns:
            list[int]：CTC 目标 id 序列。
        """
        if not positive and self.use_filler_label_for_neg:
            return [self.no_kws_id] * min(len(ids), 1)
        if positive and self.with_trans and kw_pos >= 0:
            return (
                ids[:kw_pos]
                + [self.psok_id]
                + ids[kw_pos: kw_pos + kw_len]
                + [self.peok_id]
                + ids[kw_pos + kw_len:]
            )
        return list(ids)

    # -------------------------------------------------------------- getitem
    def _load_source(self, item):
        """读音频，`trim_wake_word` 时按时间戳裁成 `wav[cmd_start:]`。

        裁剪放在**加载时**而不是离线切好存盘：`cmd_start` 只是一个数，重切一版音频
        要多占 2 TB BOS 空间且与 V2 的对齐结果失同步。`cmd_start = kw_end`（见
        `script/align/make_v2_jsonl.py`），所以唤醒词之后的那段停顿被保留——
        部署时 KWS 只给唤醒词右边界，训练时保留这段停顿才能让模型学会在其中不乱吐字。

        开放域语料 `cmd_start == 0.0`，这里天然退化成不裁剪，无需按 `corpus` 分支。

        Args:
            item: index_ds 的一条记录，用到 `source` 与 `cmd_start`。

        Returns:
            torch.Tensor：(T_wav,) f32。
        """
        waveform = self._load_waveform_robust(item["source"])
        if not self.trim_wake_word:
            return waveform
        cmd_start = float(item.get("cmd_start", 0.0))
        if cmd_start <= 0.0:
            return waveform
        offset = int(round(cmd_start * self.fs))
        # 裁完不足 0.1 s 说明这条的时间戳有问题（对齐失败/target 极短），原样返回，
        # 让长度过滤而不是空张量去处理它
        if offset >= waveform.numel() - self.fs // 10:
            logging.warning(
                "[AudioDatasetKwsSpan] cmd_start=%.2fs 裁完只剩 %d 采样点，放弃裁剪 source=%s",
                cmd_start, max(waveform.numel() - offset, 0), item["source"],
            )
            return waveform
        return waveform[offset:]

    def _cmd_offset(self, item, waveform) -> int:
        """唤醒段在**返回的波形**里占多少采样点（0 表示这条没有唤醒/指令边界）。

        只有 `trim_wake_word: false` + TTS 行（`cmd_start > 0`）才有边界；开放域语料
        `cmd_start == 0`，裁剪模式下唤醒段已被切走，两种情况都返回 0。
        两段都不足 0.1 s 时也返回 0（退回整条单一干扰），避免对极短段做能量归一。

        Args:
            item: index_ds 的一条记录。
            waveform: `_load_source` 之后的波形。

        Returns:
            int：唤醒段采样点数。
        """
        if self.trim_wake_word:
            return 0
        cmd_start = float(item.get("cmd_start", 0.0))
        if cmd_start <= 0.0:
            return 0
        off = int(round(cmd_start * self.fs))
        min_seg = self.fs // 10
        if off < min_seg or waveform.numel() - off < min_seg:
            return 0
        return off

    def _sample_streaming_wake(self, item):
        """确定唤醒段边界：按**单位**采样（一个汉字，或一个拉丁词），不是字符。

        `forced_align` 的条目是单位，之前用字符切文本、用单位切时间戳，在含拉丁词的行
        （约 3.4%）上唤醒文本与唤醒音频边界会错位。这里全程按单位切。

        train 与 dev 走**同一条**随机路径，每次取数都重采，dev 不做任何特殊化：早期版本
        让 dev 读 jsonl 里预置的 `wake_word_length` 来固定边界，代价是 dev 的唤醒词长度
        分布与训练（以及线上随机的唤醒词）不一致，验证指标不再代表实际形态。两者都截到
        ``kws_len <= full_len - 1``，保证指令段非空；索引阶段已保证单位数 >= 2 且
        对齐文本与 target 一致（见 `IndexDSJsonlKwsSpanV4._pack_units`）。

        Returns:
            (wake_len, n_chars, wake_end)：唤醒段单位数、对应字符数、结束时刻（秒）。
        """
        ends = item["unit_ends"]
        char_lens = item["unit_char_lens"]
        wake_len = random.randint(*self.wake_word_length_range)
        wake_len = max(1, min(wake_len, len(ends) - 1))
        return wake_len, sum(char_lens[:wake_len]), float(ends[wake_len - 1])

    def __getitem__(self, index):
        """返回**整条波形**与标签（fbank、唤醒/指令切分都在 `collator` 里做）。

        Args:
            index: 样本下标。

        Returns:
            dict：`waveform` (T_wav,) f32、`text`/`text_lengths`、
            `cue_token`/`cue_token_lengths`、`kws_target`、`kw_pos`
            （`spk2id` 非空时另加 `spk_id`），
            外加下划线开头的**中间量**（`collator` 里 pop 掉，不进 batch）：
            `_ids` 干净转写 id、`_cue_raw` 未加括号的线索、`_cmd_off` 唤醒段采样点数、
            `_index` 样本下标、`_source` 音频路径、`_target` 预处理后的转写文本、
            `_speaker`，`streaming_wake_word` 时另加 `_wake_end`（唤醒段结束秒，
            `collator` 混音后用它定位边界）。
        """
        item = self.index_ds[index]
        waveform = self._load_source(item)
        wake_end = None
        wake_text = ""
        if self.streaming_wake_word:
            # 波形**不在这里切**：collator 先整条混音，再用「混音偏移 + 真实时间戳」定边界。
            wake_len, n_wake_chars, wake_end = self._sample_streaming_wake(item)
        if self.preprocessor_speech:
            waveform = self.preprocessor_speech(waveform, fs=self.fs)
            waveform = torch.as_tensor(waveform).reshape(-1).to(torch.float32)

        target = item["target"]
        if self.streaming_wake_word:
            # CTC 目标是完整的「唤醒词 + 指令词」，唤醒段也要出文本，
            # 并在唤醒词两侧带 psok/peok 定位括号；不补任何前缀。
            target = "".join(str(target).split())
            if self.preprocessor_text:
                target = self.preprocessor_text(target)
            wake_text = target[:n_wake_chars]
            ids, spans = self._encode_units(target)
            wake_ids = self._encode_units(wake_text)[0]
            label = ([self.psok_id] + wake_ids + [self.peok_id]
                     + self._encode_units(target[n_wake_chars:])[0])
            positive = True
            kw_ids = wake_ids
            kw_pos, kw_len = -1, len(kw_ids)
        else:
            # V4 schema（有 kw_text）的 target 不含唤醒词，是配合 trim_wake_word=True 的口径。
            # 不裁剪时音频里那约 1 s 唤醒词就没有对应标签，CTC 会被教成「把唤醒词删掉」；
            # 补回前缀后 TTS 行与 aishell/wenetspeech 完全同构：整条音频 + 整条转写，
            # 线索是转写里的随机片段（可以跨唤醒词/指令词边界，不做特殊处理）。
            # V1 schema 用的是 keyword_text，它的 target 本来就含唤醒词全文，不能再补。
            kw_prefix = item.get("kw_text") or "" if not self.trim_wake_word else ""
            if kw_prefix:
                target = kw_prefix + target
            if self.preprocessor_text:
                target = self.preprocessor_text(target)
            ids, spans = self._encode_units(target)
            positive = random.random() < self.positive_prob
            # V4 的 jsonl 用 kw_text，V1 的用 keyword_text
            kw_text = item.get("kw_text") or item.get("keyword_text") or ""
            kw_ids, kw_pos, kw_len = self._pick_keyword(ids, spans, kw_text)
            if not positive:
                # 这里只能避开本条自己的转写；混进来的干扰源那一路由 collator 复查
                kw_ids = self._negative_keyword(index, forbid_ids=(ids,))
                kw_pos, kw_len = -1, len(kw_ids)
            label = self._inject(ids, kw_pos, kw_len, positive)
        cue_ids = self._wrap_cue(kw_ids)

        sample = {
            "waveform": waveform,
            "text": torch.tensor(label, dtype=torch.int64),
            "text_lengths": torch.tensor([len(label)], dtype=torch.int32),
            "cue_token": torch.tensor(cue_ids, dtype=torch.int64),
            "cue_token_lengths": torch.tensor([len(cue_ids)], dtype=torch.int32),
            "kws_target": torch.tensor([1 if positive else 0], dtype=torch.int64),
            "kw_pos": torch.tensor([kw_pos], dtype=torch.int64),
            "_ids": ids,
            "_cue_raw": list(kw_ids),
            "_cmd_off": self._cmd_offset(item, waveform),
            "_index": index,
            "_source": item["source"],
            "_target": target,
            "_speaker": item["speaker"],
        }
        if wake_end is not None:
            sample["_wake_end"] = wake_end
        if self.spk2id is not None:
            # 不在映射里 -> -1，模型侧折进 CE 的 ignore_index。
            # 负样本（`kws_target == 0`）的 `spk_id` 仍照发：那条样本该不该参与 CE
            # 由模型侧用 `kws_target` 当 valid 掩码决定，两件事分开，
            # 这样同一份数据换个权重/换个 valid 口径不用重跑 dataset。
            sample["spk_id"] = torch.tensor(
                [self.spk2id.get(item["speaker"], -1)], dtype=torch.int64
            )
        return sample

    # ------------------------------------------------------------------ 混音
    @staticmethod
    def _pick_partner(i, n, speakers=None, exclude=()):
        """从 `[0, n)` 里随机取一个不同说话人的下标。

        段级路径用 `exclude` 强制两段的干扰源**不同**。若当前 microbatch 没有
        合法的不同说话人伙伴则返回 ``None``，调用方保留干净目标，绝不退化为同说话人混音。
        """
        pool = [
            k for k in range(n)
            if k != i and k not in exclude
            and (speakers is None or speakers[k] != speakers[i])
        ]
        if not pool:
            return None
        return random.choice(pool)

    @staticmethod
    def _fit_interf(interf, length, shift_prob, shift_seconds):
        """把干扰源裁/补到**恰好** `length` 个采样点，保证目标段长度不变。

        段级混音不能沿用整条路径那种「补齐到两者较长者」的做法：唤醒段只 1~3 s，
        与一条 6 s 的干扰对齐会在句子中间插进几秒纯干扰、把指令段整体推后。
        比 `length` 长 -> 随机起点裁一窗；短 -> 尾部补零。
        `shift_prob` 的概率在段内前置静音（只作用于干扰路，前置到目标路会推走目标语音）。
        """
        if random.random() < shift_prob:
            delay = random.randint(16000, 16000 * shift_seconds)
            # 段可能比 1 s 还短，直接前置 1~3 s 等于把干扰推出窗口（= 这段没加干扰），
            # 故把延迟压到段长的一半以内，保证至少一半重叠。
            delay = min(delay, max(length // 2, 1))
            interf = torch.cat([interf.new_zeros(delay), interf])
        if interf.numel() > length:
            start = random.randint(0, interf.numel() - length)
            return interf[start: start + length]
        if interf.numel() < length:
            return torch.cat([interf, interf.new_zeros(length - interf.numel())])
        return interf

    @staticmethod
    def _mix_segment(target, interf, a, b):
        """段级混音：`a * target + b * (interf 归一到 target 段的能量)`，SIR = a / b。

        与整条路径（照搬 `make_mix_wav`：**两路都**等能量归一后各乘 `U(lo,hi)`）的差别
        是只归一化干扰路。目标段自己的能量不动、幅度只乘一个整条统一的 `a`，这样两段
        拼回去时目标语音没有台阶——若两段各自做等能量归一，唤醒段与指令段的目标语音
        会被缩放到不同电平，模型能直接从台阶位置读出边界，成为捷径。
        `b` 每段独立抽 -> 唤醒段与指令段的信噪比不同（本次要的就是这个）。
        """
        t_rms = target.norm(p=2).item()
        x_rms = interf.norm(p=2).item()
        if t_rms <= 0.01 or x_rms <= 0.01:
            return target * a
        return target * a + interf * (b * t_rms / x_rms)

    def _mix_batch(self, waveforms, cmd_offs=None, speakers=None):
        """batch 内干扰混音（run B），数学照搬 `make_mix_wav` / `shift_wav`。

        对每条以 `prob` 的概率从同一 microbatch 里另取一条当干扰：各源先 L2 等能量
        归一，再各乘 `U(*ratio_range)` 的幅度权重求和，**仅在削波时**把权重重归一化
        成和为 1；另有 `shift_prob` 的概率给其中一路前置 1~`shift_seconds` 秒静音。

        偏离：配对受 `sort_size` 的长度排序影响而相关，混音双方时长相近——反而让
        `max_or_min: max` 的补齐几乎免费。

        `segment_prob > 0` 时多一条**段级**分支：`cmd_offs[i] > 0`（有唤醒/指令边界，
        即 `cmd_start > 0` 且 `trim_wake_word: false`）的行以 `segment_prob` 的概率
        把波形按边界切两段，**各自**抽一个（互不相同的）干扰源、各自抽幅度权重，
        于是唤醒段与指令段的噪声内容与信噪比都不同；两段拼回后总长度不变。
        数学见 `_mix_segment` / `_fit_interf`（与整条路径的两处偏离都写在那儿）。

        Args:
            waveforms: list[(T_i,) f32]，原始波形。
            cmd_offs: list[int] 或 None，第 i 条的唤醒段采样点数（0 = 没有边界，
                走整条路径）。由 `_cmd_offset` 算出、`collator` 传入。

        Returns:
            (list[(T_i',) f32], list[list[int]], list[int])：混音后的波形（未被选中的
            原样返回）、每条实际用到的干扰源下标（未混音的是空列表），以及每条**目标路**
            被前置的静音采样点数（没前置则为 0）。第二个返回值给 `collator` 复查负样本线索
            是否撞上了干扰源的转写；第三个给 `collator` 修正唤醒/指令边界——整条路径会随机
            给某一路前置静音，选到目标路时目标语音整体后移，字级时间戳必须同步平移。
        """
        cfg = self.mix_conf
        prob = float(cfg.get("prob", 0.5))
        lo, hi = cfg.get("ratio_range", (0.1, 0.9))
        shift_prob = float(cfg.get("shift_prob", 0.3))
        shift_seconds = int(cfg.get("shift_seconds", 3))
        min_len = int(cfg.get("min_len", 16000))
        seg_prob = float(cfg.get("segment_prob", 0.0))
        if seg_prob > 0 and self.streaming_wake_word:
            # 段级路径要在混音前就按边界切分，与「先整条混音、再按偏移切」不兼容。
            raise ValueError(
                "mix_conf.segment_prob must be 0 when streaming_wake_word is enabled"
            )
        n = len(waveforms)
        if n < 2:
            return waveforms, [[] for _ in waveforms], [0] * n

        out, partners, delays = [], [], []
        for i, primary in enumerate(waveforms):
            if random.random() >= prob:
                out.append(primary)
                partners.append([])
                delays.append(0)
                continue

            off = int(cmd_offs[i]) if cmd_offs else 0
            if seg_prob > 0 and off > 0 and random.random() < seg_prob:
                # 目标语音的整体幅度 a 两段共用（不然拼接处有台阶），干扰权重每段独立
                a = random.uniform(lo, hi)
                used, segs = [], []
                for seg in (primary[:off], primary[off:]):
                    j = self._pick_partner(i, n, speakers=speakers, exclude=used)
                    if j is None:
                        # Do not partly mix a sample: retaining the original waveform is
                        # safer than adding an amplitude discontinuity at the boundary.
                        segs = None
                        break
                    interf = self._fit_interf(
                        waveforms[j], seg.numel(), shift_prob, shift_seconds
                    )
                    segs.append(self._mix_segment(seg, interf, a, random.uniform(lo, hi)))
                    used.append(j)
                if segs is None:
                    out.append(primary)
                    partners.append([])
                    delays.append(0)
                    continue
                mix = torch.cat(segs)
                # 整条路径削波时把权重重归一成和为 1；段级按段做会重新引入台阶，
                # 故改成整条同乘一个 1/peak（相对电平全部保持）。
                peak = mix.abs().max().item()
                if peak > 1.0:
                    mix = mix / peak
                out.append(mix)
                partners.append(used)
                delays.append(0)
                continue

            j = self._pick_partner(i, n, speakers=speakers)
            if j is None:
                out.append(primary)
                partners.append([])
                delays.append(0)
                continue
            wavs = [primary, waveforms[j]]

            # shift_wav(fixed_shift_duration)：给随机一路前置静音
            target_delay = 0
            if random.random() < shift_prob:
                k = random.randrange(len(wavs))
                delay = random.randint(16000, 16000 * shift_seconds)
                wavs[k] = torch.cat([wavs[k].new_zeros(delay), wavs[k]])
                if k == 0:
                    target_delay = delay

            rms = [w.norm(p=2).item() for w in wavs]
            rms = [r if r > 0.01 else 1.0 for r in rms]
            max_rms = max(rms)
            length = max(max(w.numel() for w in wavs), min_len)
            wavs = [
                torch.cat([w, w.new_zeros(length - w.numel())]) if w.numel() < length else w[:length]
                for w in wavs
            ]
            wavs = [wavs[k] * (max_rms / rms[k]) for k in range(len(wavs))]

            ratios = [random.uniform(lo, hi) for _ in wavs]
            mix = sum(wavs[k] * ratios[k] for k in range(len(wavs)))
            if mix.max() > 1 or mix.min() < -1:
                total = sum(ratios)
                ratios = [r / total for r in ratios]
                mix = sum(wavs[k] * ratios[k] for k in range(len(wavs)))
            out.append(mix)
            partners.append([j])
            delays.append(target_delay)
        return out, partners, delays

    # ---------------------------------------------------------------- collate
    def _wake_split(self, waveform, wake_end, delay, source):
        """混音后的唤醒/指令切分点（采样点）。

        `wake_end` 是**原始**目标语音里的字级时间戳；整条混音时若前置静音选到了目标路，
        目标语音整体后移 `delay` 个采样点，边界必须同步平移。两端各留 100 ms，
        保证两段都出得了 fbank 帧；越界说明时间戳与音频不匹配，记日志后夹紧。
        """
        split = int(round(float(wake_end) * self.fs)) + int(delay)
        lo, hi = self.fs // 10, waveform.numel() - self.fs // 10
        if hi < lo:
            raise ValueError(f"waveform too short to split at {wake_end:.3f}s: {source}")
        if not lo <= split <= hi:
            logging.warning(
                "[AudioDatasetKwsSpan] wake boundary %.3fs(+%d) 越界 [%d, %d]，已夹紧：%s",
                float(wake_end), int(delay), lo, hi, source,
            )
            split = min(max(split, lo), hi)
        return split

    @staticmethod
    def _describe_mix(sources, targets, partners):
        """给每条样本生成一行**人类可读**的混音溯源串，只供 `_maybe_log_sample` 打印。

        训练日志里那三行 `cue/ref/hyp` 看不出「hyp 里多出来的字是从哪来的」——100% 在线
        混音下，音频里同时有第二个说话人在说话，模型吐出对方的字是最常见的错误形态。
        把干扰源的路径与转写一起打出来，就能当场判定「这是把干扰人的话抄下来了」还是
        「模型自己幻觉」。上面那条 `ref: 不必了我自己能够找到` / `hyp: 我秘了我自己不够找到`
        就是典型：`我秘` 与 `不够` 到底来自干扰源还是声学混淆，只看三行是没法判的。

        `partners` 的长度自带混音模式信息（`_mix_batch` 的返回约定）：
            0 -> 这条没被选中混音；1 -> 整条路径；2 -> 段级（唤醒段/指令段各一个干扰源）。

        Args:
            sources: list[str]，各条的音频路径（`item["source"]`）。
            targets: list[str]，各条补过唤醒词前缀 / 预处理后的转写文本。
            partners: list[list[int]]，`_mix_batch` 返回的干扰源 batch 内下标。

        Returns:
            list[str]：与输入等长，每条是多行串（首行是模式，随后每个源两行）。
        """
        def one(tag, k):
            return f"    {tag} {sources[k]}\n         {targets[k]}"

        infos = []
        for i, part in enumerate(partners):
            if not part:
                mode = "未混音"
            elif len(part) > 1:
                mode = f"段级混音（唤醒段/指令段各一个干扰源，共 {len(part)} 个）"
            else:
                mode = "整条混音（1 个干扰源）"
            lines = [mode, one("tgt ", i)]
            lines += [one(f"itf{k}", j) for k, j in enumerate(part)]
            infos.append("\n".join(lines))
        return infos

    def collator(self, samples: list = None):
        """批量算 fbank + 父类 collate + `cue_token` 用 0 重新 pad。

        父类对 int 张量统一用 `int_pad_value`(-1) pad，会与
        `WordEmbedding(padding_idx=0)` 冲突，故 `cue_token` 单独用 0 pad；padded 位置
        由 `cue_token_lengths` 在 cross-attention 的 mask 里屏蔽。

        还负责**负样本线索的复查**：`__getitem__` 挑负样本线索时只知道本条自己的转写，
        而混进来的干扰说话人是这里才配对的。若线索恰好是某个干扰源转写的子串，那条
        「线索没出现在音频里」的前提就不成立（音频里真有人念了它），标签却是空的 ->
        监督相反。故拿到 `partners` 后重新判定，命中就换一个线索（连同 `cue_token_lengths`）。
        `_ids` / `_cue_raw` / `_cmd_off` / `_index` / `_source` / `_target` 是
        `__getitem__` 附带的内部字段，**必须在 `super().collator` 之前全部 pop 掉**
        （父类只认张量）。

        Args:
            samples: `__getitem__` 的输出列表。

        Returns:
            dict：`speech`(B,T,80) f32、`speech_lengths`(B,1) i32、`text`(B,L) i64 pad -1、
            `text_lengths`(B,1) i32、`cue_token`(B,L_k) i64 pad 0、
            `cue_token_lengths`(B,1) i32、`kws_target`(B,1) i64、`kw_pos`(B,1) i64、
            `spk_id`(B,1) i64（仅 `spk2id` 非空时；父类的通用逻辑直接堆叠，无需特判），
            以及 **`mix_info`（list[str]，长度 B，非张量）**：每条的混音溯源串，
            模型侧 `_maybe_log_sample` 打印用。它会跟着 `model(**batch)` 进 forward 的
            `**kwargs`，`to_device` 对非张量原样返回、`describe_batch` 也有非张量分支，
            所以不需要改 trainer；不用它的模型（父类以外）忽略即可。
        """
        waveforms = [s.pop("waveform") for s in samples]
        wake_ends = [s.pop("_wake_end", None) for s in samples]
        ids_list = [s.pop("_ids") for s in samples]
        cue_raws = [s.pop("_cue_raw") for s in samples]
        cmd_offs = [s.pop("_cmd_off") for s in samples]
        indices = [s.pop("_index") for s in samples]
        sources = [s.pop("_source") for s in samples]
        targets = [s.pop("_target") for s in samples]
        speakers = [s.pop("_speaker") for s in samples]
        cue_list = [s.pop("cue_token") for s in samples]

        partners = [[] for _ in samples]
        delays = [0] * len(samples)
        if self.mix_conf:
            # 先整条混音，再按「混音偏移 + 真实时间戳」切唤醒/指令段：反过来做
            # （先切再只混指令段）会让唤醒段永远干净，锁定说话人的任务变得平凡。
            waveforms, partners, delays = self._mix_batch(waveforms, cmd_offs, speakers)
            for i, part in enumerate(partners):
                if not part or int(samples[i]["kws_target"]) == 1:
                    continue
                forbid = [ids_list[i]] + [ids_list[j] for j in part]
                if not any(self._contains(f, cue_raws[i]) for f in forbid):
                    continue
                kw_ids = self._negative_keyword(indices[i], forbid_ids=forbid)
                cue_ids = self._wrap_cue(kw_ids)
                cue_list[i] = torch.tensor(cue_ids, dtype=torch.int64)
                samples[i]["cue_token_lengths"] = torch.tensor(
                    [len(cue_ids)], dtype=torch.int32
                )

        wake_waveforms = None
        if self.streaming_wake_word:
            if any(end is None for end in wake_ends):
                raise RuntimeError("streaming_wake_word batch has no wake boundary")
            wake_waveforms = []
            for i, wav in enumerate(waveforms):
                split = self._wake_split(wav, wake_ends[i], delays[i], sources[i])
                wake_waveforms.append(wav[:split])
                waveforms[i] = wav[split:]

        speech, speech_lengths = extract_fbank(
            waveforms, data_type=self.data_type, frontend=self.frontend, is_final=True
        )

        outputs = super().collator(samples)
        outputs["speech"] = speech
        outputs["speech_lengths"] = speech_lengths.reshape(-1, 1).to(torch.int32)
        if wake_waveforms is not None:
            # 两段各自算 fbank 会在边界丢掉跨界的 1~2 帧（25 ms 窗 / 10 ms 移），
            # 推理侧同样按边界切，口径一致。
            wake_speech, wake_lengths = extract_fbank(
                wake_waveforms, data_type=self.data_type, frontend=self.frontend, is_final=True
            )
            outputs["wake_speech"] = wake_speech
            outputs["wake_speech_lengths"] = wake_lengths.reshape(-1, 1).to(torch.int32)
        outputs["cue_token"] = torch.nn.utils.rnn.pad_sequence(
            cue_list, batch_first=True, padding_value=0
        )
        outputs["mix_info"] = self._describe_mix(sources, targets, partners)
        return outputs
