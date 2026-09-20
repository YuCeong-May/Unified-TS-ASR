"""ShardAudioDataset: 流式加载 webdataset 风格的 shard tar 包。

设计参考 wenet 的 shard 加载流程，做完全的 IterableDataset：

shard list (.list/.listl/.txt) → epoch 内 shuffle → 按 DDP rank 切片
  → 按 DataLoader worker 切片
  → 每个 worker 同时持有 N 个 tar 的字节缓冲（in-memory 缓冲池）
  → 解析得到 (wav_bytes, target) 样本
  → 进入 sample_buffer_size 大小的局部 shuffle 池
  → 取出后做 fbank/分词，yield 单条样本

与现有 IndexDSShard（Map-style，全量索引）的区别：
  - IndexDSShard 启动慢（要读所有 tar 的 member 表），适合中小数据集
  - ShardAudioDataset 启动 O(1)（只读 shard list），适合大数据集，行为与 wenet 对齐
"""

import io
import json
import logging
import os
import random
import tarfile
from collections import deque
from typing import List

import torch
from torch.utils.data import IterableDataset

from funasr.register import tables
from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video


_AUDIO_EXTS = {"wav", "flac", "mp3", "ogg", "m4a", "opus"}
_TEXT_EXTS = {"txt", "text"}


def read_shard_list(path: str) -> List[str]:
    """从 path 解析 shard tar 文件列表。

    支持三种形态：
      - 直接 .tar 路径
      - .list / .listl / .txt：每行 `{"file": "/abs/path/xxx.tar"}` 或纯 tar 路径
    """
    if path.endswith(".tar"):
        return [path]
    files: List[str] = []
    with open(path, encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if isinstance(d, dict) and "file" in d:
                    files.append(d["file"])
                    continue
            except json.JSONDecodeError:
                pass
            files.append(line)
    return files


def _parse_tar_bytes(tar_bytes: bytes, tar_path: str):
    """完全在内存中解析 tar，按 key 配对 (audio_bytes, target)，逐条 yield。"""
    pairs = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r|*") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                name = m.name
                if "." not in name:
                    continue
                key, ext = name.rsplit(".", 1)
                ext = ext.lower()
                if ext not in _AUDIO_EXTS and ext not in _TEXT_EXTS:
                    continue
                f = tf.extractfile(m)
                if f is None:
                    continue
                data = f.read()
                entry = pairs.setdefault(key, {})
                if ext in _AUDIO_EXTS:
                    entry["audio_bytes"] = data
                    entry["audio_ext"] = ext
                else:
                    entry["target"] = data.decode("utf-8", errors="ignore").strip()
                # 如果该 key 的 wav+txt 都齐了，立刻 yield 并释放内存
                if "audio_bytes" in entry and "target" in entry:
                    yield key, entry["audio_bytes"], entry["target"]
                    pairs.pop(key, None)
    except Exception as e:
        logging.warning(f"ShardAudioDataset: failed to parse {tar_path}: {e}")


@tables.register("dataset_classes", "ShardAudioDataset")
class ShardAudioDataset(IterableDataset):
    """webdataset-style shard tar IterableDataset.

    主要 dataset_conf 选项：
      shard_buffer_size:    int  = 4     一次性加载到内存的 tar 数（局部 shuffle 池由它决定）
      shard_prefetch:       int  = 2     预读多少个 tar 在 deque 里待用
      shard_shuffle:        bool = True  每个 epoch 是否打乱 shard 列表（仅 training）
      sample_shuffle:       bool = True  缓冲池内的样本是否再做一次局部 shuffle（仅 training）
      shuffle_seed:         int  = 1234  shuffle 基础种子（实际 seed = base + epoch + ...）
      samples_per_shard:    int  = 3000  __len__ 估计用，每个 tar 的样本数（仅用于步数估算）
      max_target_length:    int  = 2048  长度过滤
      min_target_length:    int  = 0
      max_token_length:     int          可选，target 字符上限
    """

    def __init__(
        self,
        path: str,
        frontend=None,
        tokenizer=None,
        is_training: bool = True,
        int_pad_value: int = -1,
        float_pad_value: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.path = path
        self.shards = read_shard_list(path)
        if not self.shards:
            logging.warning(f"ShardAudioDataset: empty shard list from {path}")

        self.frontend = frontend
        self.tokenizer = tokenizer
        self.is_training = is_training
        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value
        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = "sound"

        # filtering thresholds
        self.max_target_length = int(kwargs.get("max_target_length", 2048))
        self.min_target_length = int(kwargs.get("min_target_length", 0))
        self.max_token_length = int(kwargs.get("max_token_length", 10**9))
        # 音频侧（LFR 后帧数）过滤，与 AudioDataset 的 max_length 行为对齐
        # 设默认值 0 / 10**9 表示不过滤；上层 yaml 显式给值时才生效
        self.max_speech_frame_length = int(kwargs.get("max_speech_frame_length", 10**9))
        self.min_speech_frame_length = int(kwargs.get("min_speech_frame_length", 0))

        # buffer / shuffle
        self.shard_buffer_size = max(1, int(kwargs.get("shard_buffer_size", 4)))
        self.shard_prefetch = max(0, int(kwargs.get("shard_prefetch", 2)))
        self.shard_shuffle = bool(kwargs.get("shard_shuffle", True)) and is_training
        self.sample_shuffle = bool(kwargs.get("sample_shuffle", True)) and is_training
        self.shuffle_seed = int(kwargs.get("shuffle_seed", 1234))

        # 远程存储（可选）：从 BOS 按需拉取 tar 字节，其余 buffer/shuffle/batch 逻辑不变
        #   shard_storage:     "local"（默认）| "bos"
        #   bos_local_prefix:  本地挂载前缀，与 bos://{bos_bucket}/ 一一对应
        #     例：/data/object-store/  <->  bos://example-bucket/
        self.shard_storage = str(kwargs.get("shard_storage", "local")).lower()
        self.bos_endpoint = kwargs.get("bos_endpoint", "http://bj.bcebos.com")
        self.bos_bucket = kwargs.get("bos_bucket", None)
        self.bos_local_prefix = kwargs.get("bos_local_prefix", None)
        self.bos_credentials = os.path.expanduser(
            kwargs.get("bos_credentials", "~/.baidubce/credentials")
        )
        # 在 worker 内 lazy 创建：BosClient 自带 at_fork 句柄，主进程创建会被子进程重置
        self._bos_client = None

        # __len__ 估算
        self.samples_per_shard = int(kwargs.get("samples_per_shard", 3000))
        self.batch_size = int(kwargs.get("batch_size", 1))

        # 与 AudioDataset 对齐的预处理
        self.preprocessor_speech = None
        self.preprocessor_text = None
        if is_training:
            ps = kwargs.get("preprocessor_speech", None)
            if ps:
                cls = tables.preprocessor_classes.get(ps)
                self.preprocessor_speech = cls(**kwargs.get("preprocessor_speech_conf", {}))
            pt = kwargs.get("preprocessor_text", None)
            if pt:
                cls = tables.preprocessor_classes.get(pt)
                self.preprocessor_text = cls(**kwargs.get("preprocessor_text_conf", {}))

        self.epoch = 0
        logging.info(
            f"ShardAudioDataset: {len(self.shards)} shards from {path}, "
            f"storage={self.shard_storage}, "
            f"is_training={is_training}, shard_buffer_size={self.shard_buffer_size}, "
            f"shard_prefetch={self.shard_prefetch}, shard_shuffle={self.shard_shuffle}, "
            f"sample_shuffle={self.sample_shuffle}, "
            f"min_target_length={self.min_target_length}, max_target_length={self.max_target_length}, "
            f"max_token_length={self.max_token_length}, "
            f"min_speech_frame_length={self.min_speech_frame_length}, "
            f"max_speech_frame_length={self.max_speech_frame_length}"
        )

    # ---------------------------------------------------------- trainer interface
    def set_epoch(self, epoch: int):
        """设置当前 epoch 用于数据打乱。"""
        self.epoch = int(epoch)

    # ---------------------------------------------------------- BOS backend
    def _local_to_bos_key(self, tar_path: str) -> str:
        """把本地 tar 路径映射成 BOS object key。

        约定：本地 self.bos_local_prefix 挂载点 <-> bos://{bos_bucket}/。
        例：bos_local_prefix=/data/object-store/, bos_bucket=example-bucket
            /data/object-store/project/datasets/shards/a.tar
              -> key = project/datasets/shards/a.tar
        如果路径不在挂载点下，则退回用 bucket 名做切分兜底。
        """
        p = tar_path
        if self.bos_local_prefix and p.startswith(self.bos_local_prefix):
            return p[len(self.bos_local_prefix):].lstrip("/")
        # 兜底：按 "/{bucket}/" 切分，取其后的部分
        marker = f"/{self.bos_bucket}/"
        idx = p.find(marker)
        if idx >= 0:
            return p[idx + len(marker):]
        return p.lstrip("/")

    def _get_bos_client(self):
        """worker 内 lazy 创建 BosClient（避免主进程 fork 时句柄失效）。"""
        if self._bos_client is None:
            from bostorchconnector._bos_client import BosClient, BosClientConfig

            cfg = BosClientConfig(credentials_path=self.bos_credentials, log_level=2)
            self._bos_client = BosClient(
                endpoint=self.bos_endpoint,
                bos_client_config=cfg,
                dataset_type=1,
            )
        return self._bos_client

    def _read_tar_bytes(self, tar_path: str):
        """读取单个 tar 的字节流：local 直接读盘，bos 走 BosClient.get_object。"""
        if self.shard_storage == "bos":
            key = self._local_to_bos_key(tar_path)
            reader = self._get_bos_client().get_object(self.bos_bucket, key)
            return reader.read()
        with open(tar_path, "rb") as f:
            return f.read()

    def __len__(self):
        try:
            world = torch.distributed.get_world_size()
        except Exception:
            world = 1
        world = max(1, world)
        n_shards = max(1, (len(self.shards) + world - 1) // world)
        return n_shards * self.samples_per_shard

    # ---------------------------------------------------------- shard split
    def _split_shards(self) -> List[str]:
        try:
            world = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        except Exception:
            world = 1
            rank = 0

        info = torch.utils.data.get_worker_info()
        if info is None:
            n_workers, worker_id = 1, 0
        else:
            n_workers, worker_id = info.num_workers, info.id

        shards = list(self.shards)
        if self.shard_shuffle:
            rnd = random.Random(self.shuffle_seed + self.epoch * 1009 + 17)
            rnd.shuffle(shards)

        # Pad to make every rank receive the same number of shards. Without this,
        # ranks with fewer shards may finish early while other ranks are still doing
        # training collectives, which can deadlock NCCL at epoch boundaries.
        if world > 1 and shards:
            total_size = ((len(shards) + world - 1) // world) * world
            shards += shards[: total_size - len(shards)]

        # rank stride 切分（每个 rank 拿到的 shard 数相同）
        shards = shards[rank::world] if world > 1 else shards
        # worker stride 再切一次
        if n_workers > 1:
            shards = shards[worker_id::n_workers]
        return shards

    # ---------------------------------------------------------- buffered shard read
    def _iter_shard_samples(self, my_shards: List[str]):
        """按 shard_buffer_size 一次加载若干 tar，做 in-memory 局部 shuffle 后 yield 样本。

        prefetch deque 保证当前批解析完时下一批已经读完磁盘 IO。
        """
        shard_queue: deque = deque(my_shards)
        bytes_buf: deque = deque()  # (tar_path, tar_bytes)

        def fill_bytes_buf():
            target = self.shard_buffer_size + self.shard_prefetch
            while shard_queue and len(bytes_buf) < target:
                tar_path = shard_queue.popleft()
                if self.shard_storage != "bos" and not os.path.exists(tar_path):
                    logging.warning(f"ShardAudioDataset: missing shard {tar_path}, skip")
                    continue
                try:
                    bytes_buf.append((tar_path, self._read_tar_bytes(tar_path)))
                except Exception as e:
                    logging.warning(f"ShardAudioDataset: failed to read {tar_path}: {e}")

        fill_bytes_buf()

        while bytes_buf:
            sample_pool = []
            n = min(self.shard_buffer_size, len(bytes_buf))
            for _ in range(n):
                tar_path, tar_bytes = bytes_buf.popleft()
                for key, wav_bytes, target in _parse_tar_bytes(tar_bytes, tar_path):
                    sample_pool.append((key, wav_bytes, target))
                # 主动释放
                del tar_bytes

            # 解析下一批 tar 的同时（其实是顺序的，prefetch 主要靠 deque 缓存磁盘已读字节）
            fill_bytes_buf()

            if self.sample_shuffle and sample_pool:
                # 每次缓冲池都换一个种子，避免 epoch 内不同批之间相关
                seed = self.shuffle_seed + self.epoch * 1009 + len(sample_pool) * 31
                random.Random(seed).shuffle(sample_pool)

            for item in sample_pool:
                yield item

    # ---------------------------------------------------------- main iter
    def __iter__(self):
        my_shards = self._split_shards()
        if not my_shards:
            return

        for key, wav_bytes, target in self._iter_shard_samples(my_shards):
            target_len = len(target)
            if target_len < self.min_target_length or target_len > self.max_target_length:
                continue
            if target_len > self.max_token_length:
                continue

            try:
                source = io.BytesIO(wav_bytes)
                data_src = load_audio_text_image_video(source, fs=self.fs)
                if self.preprocessor_speech is not None:
                    data_src = self.preprocessor_speech(data_src, fs=self.fs)
                speech, speech_lengths = extract_fbank(
                    data_src,
                    data_type=self.data_type,
                    frontend=self.frontend,
                    is_final=True,
                )
            except Exception as e:
                logging.warning(f"ShardAudioDataset: decode failed for {key}: {e}")
                continue

            # 音频侧（LFR 后帧数）过滤：避免极短/极长样本撑爆 cross-attention
            n_lfr_frames = int(speech_lengths.item())
            if (
                n_lfr_frames < self.min_speech_frame_length
                or n_lfr_frames > self.max_speech_frame_length
            ):
                continue

            t = target
            if self.preprocessor_text is not None:
                t = self.preprocessor_text(t)
            if self.tokenizer is not None:
                ids = self.tokenizer.encode(t)
                text = torch.tensor(ids, dtype=torch.int64)
            else:
                ids = t
                text = ids
            text_lengths = torch.tensor([len(ids)], dtype=torch.int32)

            yield {
                "speech": speech[0, :, :],
                "speech_lengths": speech_lengths,
                "text": text,
                "text_lengths": text_lengths,
            }

    # ---------------------------------------------------------- collator
    def collator(self, samples):
        """将多个样本拼成一个 batch，对 Tensor 做 padding。"""
        outputs = {}
        for sample in samples:
            for key in sample.keys():
                outputs.setdefault(key, []).append(sample[key])
        for key, data_list in outputs.items():
            if isinstance(data_list[0], torch.Tensor):
                if data_list[0].dtype in (torch.int64, torch.int32):
                    pad_value = self.int_pad_value
                else:
                    pad_value = self.float_pad_value
                outputs[key] = torch.nn.utils.rnn.pad_sequence(
                    data_list, batch_first=True, padding_value=pad_value
                )
        return outputs
