"""IndexDSShard: 从 webdataset 风格的 shard tar 包加载样本。

FunASR 原生 IndexDSJsonl 只支持 jsonl / list-of-jsonl，无法直接读取 tar 包。
本模块实现 IndexDSShard：

- 启动时扫所有 tar，建立内存索引 `{key: (tar_path, wav_offset, wav_size, target)}`，
  wav_offset 用 tarfile 的 offset_data 字段，wav_size 用 member.size。
- __getitem__ 时直接 open(tar_path,'rb') + seek(offset) + read(size)，得到 wav 字节，
  包装成 io.BytesIO 交给 AudioDataset 现有的 load_audio_text_image_video（已支持 file-like）。

每个 tar 内部要求每个样本由同名 .wav 和 .txt 组成（webdataset 标准）。
"""

import io
import json
import logging
import os
import tarfile

import torch

from funasr.register import tables


@tables.register("index_ds_classes", "IndexDSShard")
class IndexDSShard(torch.utils.data.Dataset):
    """从 webdataset 风格 shard tar 包加载样本的索引数据集。

    Args:
        path: 数据路径，支持三种形态：
            - 单个 .tar 文件
            - .list/.txt 文本文件，每行 `{"file": "/abs/path/xxx.tar"}` 或纯 tar 绝对路径
            - 任意文本文件（每行一个 tar 路径，按 JSON 解析失败时按原样处理）
        kwargs: 与 IndexDSJsonl 一致的过滤参数，如 max_token_length 等。
    """

    def __init__(self, path: str, **kwargs):
        super().__init__()
        self.max_source_length = kwargs.get("max_source_length", 2048)
        self.min_source_length = kwargs.get("min_source_length", 0)
        self.max_target_length = kwargs.get("max_target_length", 2048)
        self.min_target_length = kwargs.get("min_target_length", 0)
        self.max_token_length = kwargs.get("max_token_length", 2200)

        tar_files = self._collect_tar_files(path)
        logging.info(f"IndexDSShard: collected {len(tar_files)} tar(s) from {path}")

        self.contents = []
        skipped_missing = 0
        for tar_path in tar_files:
            if not os.path.exists(tar_path):
                skipped_missing += 1
                continue
            try:
                self._index_one_tar(tar_path)
            except Exception as e:
                logging.warning(f"IndexDSShard: failed to index {tar_path}: {e}")
        if skipped_missing:
            logging.warning(
                f"IndexDSShard: skipped {skipped_missing} missing tar(s) "
                f"(data may still be uploading, continuing without them)"
            )

        logging.info(f"IndexDSShard: indexed {len(self.contents)} samples")

    @staticmethod
    def _collect_tar_files(path):
        """从 path 推断 tar 文件列表。"""
        tar_files = []
        if not os.path.exists(path):
            logging.error(f"IndexDSShard: path not found: {path}")
            return tar_files

        if path.endswith(".tar"):
            tar_files = [path]
            return tar_files

        # 当作 list file：每行一个 tar 路径，或 {"file": "..."} json
        with open(path, encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if isinstance(d, dict) and "file" in d:
                        tar_files.append(d["file"])
                        continue
                except json.JSONDecodeError:
                    pass
                # 退化处理：整行当作 tar 路径
                tar_files.append(line)
        return tar_files

    def _index_one_tar(self, tar_path):
        """打开单个 tar，把 (key, wav_offset, wav_size, target) 入索引。"""
        with tarfile.open(tar_path, "r") as tf:
            wav_members = {}
            txt_members = {}
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                name = m.name
                if name.endswith(".wav"):
                    wav_members[name[:-4]] = m
                elif name.endswith(".txt"):
                    txt_members[name[:-4]] = m

            # 一次性读出所有 txt 内容（小文件，集中读避免多次 seek）
            txt_contents = {}
            for key, m in txt_members.items():
                f = tf.extractfile(m)
                if f is None:
                    continue
                txt_contents[key] = f.read().decode("utf-8", errors="ignore").strip()

            for key, wav_m in wav_members.items():
                if key not in txt_contents:
                    continue
                target = txt_contents[key]
                target_len = len(target)
                # source_len 用 wav 字节大小粗略估算（16kHz/16bit/mono ≈ 32000 bytes/s）
                # 单位约为采样点，足够给 batch sampler 做长度过滤和动态批用
                source_len = max(1, wav_m.size // 2)

                if (
                    source_len < self.min_source_length
                    or source_len > self.max_source_length
                ):
                    continue
                if (
                    target_len < self.min_target_length
                    or target_len > self.max_target_length
                ):
                    continue
                if source_len + target_len > self.max_token_length:
                    continue

                self.contents.append(
                    {
                        "key": key,
                        "tar_path": tar_path,
                        "wav_offset": wav_m.offset_data,
                        "wav_size": wav_m.size,
                        "target": target,
                        "source_len": source_len,
                        "target_len": target_len,
                    }
                )

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, index):
        c = self.contents[index]
        # 直接二进制读 tar 文件的 wav 字节段，绕过 tarfile 句柄状态（线程安全、零开销）
        with open(c["tar_path"], "rb") as f:
            f.seek(c["wav_offset"])
            wav_bytes = f.read(c["wav_size"])
        return {
            "source": io.BytesIO(wav_bytes),
            "target": c["target"],
            "source_len": c["source_len"],
            "target_len": c["target_len"],
        }

    def get_source_len(self, data_dict):
        """返回音频源长度。"""
        return data_dict.get("source_len", 1)

    def get_target_len(self, data_dict):
        """返回目标文本长度。"""
        return data_dict.get("target_len", 0)
