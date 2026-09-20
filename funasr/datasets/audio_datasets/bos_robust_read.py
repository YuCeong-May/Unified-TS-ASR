# -*- coding: utf-8 -*-
"""BOS 音频读取的健壮化 mixin：重试退避 + worker 内并行读。

抽取自 V4（`datasets_kws_span_tsasr.AudioDatasetKwsSpan`）在 2026-08-07 三次训练崩溃后
沉淀下来的读取路径，供全部 TS-ASR 数据集类共用。**不影响** vanilla `AudioDataset` /
`AudioDatasetHotword`——它们不继承本 mixin，行为逐字节不变。

解决两个独立问题：

1. **BOS 偶发返回空 body 且不抛异常。** `get_object(...).read()` 在突发负载下会返回
   `b""`（实测 2292/2292 条失败全是 `bytes=0 head=b''`），既不是截断也不是对象损坏——
   事后单线程复读全部完好。这些字节进 `load_audio_text_image_video` 后，tier1
   （`torchaudio.load`）/ tier2（`sf.read`）都在裸 `except:` 下，tier3
   `_load_audio_ffmpeg` 又在 `file.lower()` 上假定入参是路径字符串，于是一律以
   `AttributeError: '_io.BytesIO' object has no attribute 'lower'` 收尾，真因被完全掩盖，
   最后打死 worker、拖死整个多卡任务。
   对策：**先自己判空**（给出明确的 IOError）+ 指数退避重试 + 每次失败丢弃 `BosClient`
   让下次读重建连接。失败是**成簇爆发**的（5 秒内 1507 条，集中在验证 dataloader 启动
   的瞬间），毫秒级连发的重试等于都落在同一个坏窗口里，所以**退避是必需的**，不是可选项。
   重试语义中性：同一条样本重新读，不跳过、不替换。

2. **单 worker 的 BOS 读是串行的。** 单条样本 106 ms 里 BOS 读占 89 ms
   （TTFB 17 ms + 231 KB 传输 59 ms，单流仅 3.9 MB/s），而 worker 是「发请求-等-解码」
   严格串行 -> 单 worker 只有 9.4 条/s。`__getitems__` 让 PyTorch 把**整个 batch** 的
   下标一次性交过来，在 worker 内用线程池并行拉取，单 worker 提到约 29 条/s（约 3x）。
   8 线程即到进程级天花板（16/32 线程无增益，GIL），跨进程则严格线性。
   全局在飞请求数 = `num_workers × bos_workers`。

代价（已接受）：并行读改变了 `random` 的消费顺序，逐样本的可复现性不再成立。
"""
import io
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from funasr.utils.load_utils import load_audio_text_image_video

# 小于这个字节数不可能是任何真实音频（wav 头本身就 44 字节），直接判为读失败。
_MIN_AUDIO_BYTES = 44


class BosRobustReadMixin:
    """给 `AudioDataset` 子类加上「重试退避 + 并行读」的 BOS 读取路径。

    用法：把它放在基类**左边**（`class Foo(BosRobustReadMixin, AudioDataset)`），
    并在 `super().__init__(...)` 之后调用 `self._init_bos_robust_read(**kwargs)`。

    暴露给 `dataset_conf` 的三个开关：
        bos_workers: 每个 dataloader worker 内并行拉 BOS 的线程数（默认 8）。
            设 <=1 或非 BOS 存储时退回串行路径。
        bos_read_retry: 单条音频「读+解码」失败后的重试次数（默认 4，即最多读 5 次）。
        bos_retry_backoff: 重试前的基础退避秒数（默认 0.5），实际等待
            `backoff * 2**attempt * U(0.5, 1.5)`，跨 worker 抖开避免同步重试。
    """

    def _init_bos_robust_read(self, **kwargs):
        """读取三个开关并初始化线程池占位。必须在 `super().__init__` 之后调用。"""
        self.bos_workers = int(kwargs.get("bos_workers", 8))
        self.bos_read_retry = max(0, int(kwargs.get("bos_read_retry", 4)))
        self.bos_retry_backoff = float(kwargs.get("bos_retry_backoff", 0.5))
        # 线程池必须在 worker 进程内 lazy 创建：dataset 会被 pickle 发给 worker，
        # 而 ThreadPoolExecutor 不可 pickle；fork 后父进程的池对象也没有活线程。
        self._pool = None
        self._pool_pid = None

    # ------------------------------------------------------------- 线程池
    def __getstate__(self):
        """pickle 到 worker 前丢掉线程池（不可 pickle）。"""
        state = self.__dict__.copy()
        state["_pool"] = None
        state["_pool_pid"] = None
        return state

    def _get_pool(self):
        """worker 内 lazy 创建线程池；按 pid 判断，fork 后自动重建。"""
        pid = os.getpid()
        if self._pool is None or self._pool_pid != pid:
            self._pool = ThreadPoolExecutor(
                max_workers=self.bos_workers,
                thread_name_prefix=f"bosio{pid}",
            )
            self._pool_pid = pid
        return self._pool

    def __getitems__(self, indices):
        """整批取样本，BOS 读取用 worker 内线程池并行（PyTorch 的 batch fetch 钩子）。

        `_MapDatasetFetcher.fetch` 见到 dataset 有 `__getitems__` 就会把**整个 batch**
        的下标一次性交过来（否则退化成 `[self.dataset[i] for i in ...]` 的串行列表推导，
        单 worker 独自串行拉完整批，约 120 条 x 106 ms ≈ 12.7 s）。

        Args:
            indices: 一个 batch 的样本下标列表。

        Returns:
            list：与 `__getitem__` 同构、顺序与 `indices` 一致的样本列表。
        """
        if (
            getattr(self, "bos_workers", 0) <= 1
            or self.data_storage != "bos"
            or len(indices) < 2
        ):
            return [self[i] for i in indices]
        return list(self._get_pool().map(self.__getitem__, indices))

    # ------------------------------------------------------------- 读 + 重试
    @staticmethod
    def _wav_bytes_report(raw) -> str:
        """给出 BOS 读回字节的完整性摘要，只用于日志（不抛异常）。

        把「字节数」和「wav 头自称的长度」一起打进日志，失败时能一眼分辨
        「读回来的字节本身不完整」还是「字节完好但进程内解码失败」。

        Args:
            raw: BOS 读回的原始字节。

        Returns:
            str：形如 `bytes=231044 declared=231044 head=b'RIFF'`。
        """
        n = len(raw)
        if n < 8:
            return f"bytes={n} head={raw[:4]!r}"
        declared = int.from_bytes(raw[4:8], "little") + 8
        return f"bytes={n} declared={declared} head={raw[:4]!r}"

    def _load_waveform_robust(self, source_path, fs=None):
        """读一条音频并解码成 1-D f32 波形，失败按 `bos_read_retry` 退避重读。

        每次失败都丢弃当前 `BosClient`，下次尝试重建连接——失败是成簇爆发的，
        且簇内所有对象事后单线程复读均完好，所以「换连接 + 等一会儿再来」是对症的。
        本地存储时不重试（本地读失败是真错误，重试只会掩盖它）。

        Args:
            source_path: jsonl 里的 source（本地路径形态，BOS 下按前缀映射成 key）。
            fs: 目标采样率，None 用 `self.fs`。

        Returns:
            torch.Tensor：(T_wav,) f32。

        Raises:
            最后一次尝试的原始异常（重试用尽仍失败时原样抛出）。
        """
        fs = self.fs if fs is None else fs
        if self.data_storage != "bos":
            return self._decode_waveform(source_path, fs)

        for attempt in range(self.bos_read_retry + 1):
            report = ""
            try:
                raw = self._read_source_bytes(source_path)
                report = self._wav_bytes_report(raw)
                # 先自己判空：BOS 会静默返回 b""，若直接喂给 load_audio，
                # 真因会被 load_utils 的裸 except + file.lower() 换成 AttributeError。
                if len(raw) < _MIN_AUDIO_BYTES:
                    raise IOError(
                        f"BOS 返回的字节过短，不可能是音频（{report}），"
                        f"source={source_path}"
                    )
                return self._decode_waveform(io.BytesIO(raw), fs)
            except Exception as err:  # noqa: BLE001 - 读/解码的任何失败都重试
                logging.warning(
                    "[%s] 读取音频失败(第 %d/%d 次)，source=%s, %s: %s: %s",
                    type(self).__name__, attempt + 1, self.bos_read_retry + 1,
                    source_path, report or "(未取到字节)", type(err).__name__, err,
                )
                self._bos_client = None
                if attempt >= self.bos_read_retry:
                    raise
                time.sleep(
                    self.bos_retry_backoff
                    * (2 ** attempt)
                    * (0.5 + random.random())
                )
        raise IOError(f"读取音频失败，source={source_path}")

    @staticmethod
    def _decode_waveform(source, fs):
        """解码成 1-D f32 张量。`preprocessor_speech` 刻意留在重试循环**之外**。"""
        wav = load_audio_text_image_video(source, fs=fs)
        if isinstance(wav, (list, tuple)):
            wav = wav[0]
        if not isinstance(wav, torch.Tensor):
            wav = torch.as_tensor(wav)
        return wav.reshape(-1).to(torch.float32)

    # ----------------------------------------------------------- 基类读取钩子
    def _load_source(self, item):
        """覆盖 `AudioDataset._load_source`，把主音频的读取换成健壮版本。

        V1/V3 的 `__getitem__` 只是 `super().__getitem__(index)` 再补线索字段，
        所以覆盖这个钩子就等于给它们换上了重试退避路径。

        Args:
            item: index_ds 的一条记录，用到 `source`。

        Returns:
            torch.Tensor：(T_wav,) f32。
        """
        return self._load_waveform_robust(item["source"])
