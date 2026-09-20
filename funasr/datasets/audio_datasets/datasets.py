"""FunASR 音频数据集模块，提供训练和推理用的数据加载功能。"""
import io
import os
import logging
import torch
import random


from funasr.register import tables
from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video


@tables.register("dataset_classes", "AudioDataset")
class AudioDataset(torch.utils.data.Dataset):
    """
    AudioDataset
    """

    def __init__(
        self,
        path,
        index_ds: str = None,
        frontend=None,
        tokenizer=None,
        is_training: bool = True,
        int_pad_value: int = -1,
        float_pad_value: float = 0.0,
        **kwargs,
    ):
        """Initialize AudioDataset.
        
            Args:
                path: TODO.
                index_ds: TODO.
                frontend: Audio frontend for feature extraction.
                tokenizer: Tokenizer instance for text encoding/decoding.
                is_training: Boolean flag for training.
                int_pad_value: TODO.
                float_pad_value: TODO.
                **kwargs: Additional keyword arguments.
            """
        super().__init__()
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)

        self.preprocessor_speech = None
        self.preprocessor_text = None

        if is_training:
            preprocessor_speech = kwargs.get("preprocessor_speech", None)
            if preprocessor_speech:
                preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
                preprocessor_speech = preprocessor_speech_class(
                    **kwargs.get("preprocessor_speech_conf")
                )
            self.preprocessor_speech = preprocessor_speech
            preprocessor_text = kwargs.get("preprocessor_text", None)
            if preprocessor_text:
                preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
                preprocessor_text = preprocessor_text_class(**kwargs.get("preprocessor_text_conf"))
            self.preprocessor_text = preprocessor_text

        self.frontend = frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = "sound"
        self.tokenizer = tokenizer

        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value

        # 远程存储（可选）：从 BOS 按需拉取音频字节，其余（fbank/分词/collate）逻辑不变。
        #   data_storage:      "local"（默认）| "bos"
        #   bos_local_prefix:  本地挂载前缀，与 bos://{bos_bucket}/ 一一对应
        #     例：/data/object-store/  <->  bos://example-bucket/
        #   jsonl 里的 source 仍写本地挂载路径，运行时按前缀映射成 object key 拉取。
        self.data_storage = str(kwargs.get("data_storage", "local")).lower()
        self.bos_endpoint = kwargs.get("bos_endpoint", "http://bj.bcebos.com")
        self.bos_bucket = kwargs.get("bos_bucket", None)
        self.bos_local_prefix = kwargs.get("bos_local_prefix", None)
        self.bos_credentials = os.path.expanduser(
            kwargs.get("bos_credentials", "~/.baidubce/credentials")
        )
        # 在 worker 内 lazy 创建：BosClient 自带 at_fork 句柄，主进程创建会被子进程重置
        self._bos_client = None

    # ---------------------------------------------------------- BOS backend
    def _local_to_bos_key(self, source_path: str) -> str:
        """把本地音频路径映射成 BOS object key（与 ShardAudioDataset 约定一致）。

        约定：本地 self.bos_local_prefix 挂载点 <-> bos://{bos_bucket}/。
        例：bos_local_prefix=/data/object-store/, bos_bucket=example-bucket
            /data/object-store/project/datasets/xxx.wav -> key = project/datasets/xxx.wav
        路径不在挂载点下时，退回按 "/{bucket}/" 切分兜底。
        """
        p = source_path
        if self.bos_local_prefix and p.startswith(self.bos_local_prefix):
            return p[len(self.bos_local_prefix):].lstrip("/")
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

    def _read_source_bytes(self, source_path: str) -> bytes:
        """从 BOS 读取单个音频文件的字节流。"""
        key = self._local_to_bos_key(source_path)
        reader = self._get_bos_client().get_object(self.bos_bucket, key)
        return reader.read()

    def _load_source(self, item):
        """读取并解码主音频，返回波形。**子类覆盖点。**

        抽出来只为让 TS-ASR 的数据集类能挂上 `BosRobustReadMixin` 的
        「重试退避 + 并行读」路径（见 `bos_robust_read.py`）；本类的行为不变。

        Args:
            item: index_ds 的一条记录，用到 `source`。

        Returns:
            `load_audio_text_image_video` 的返回值（波形）。
        """
        source = item["source"]
        if self.data_storage == "bos":
            # 从 BOS 拉取字节，包成 BytesIO 交给下游解码（load_audio 支持 file-like）
            source = io.BytesIO(self._read_source_bytes(item["source"]))
        return load_audio_text_image_video(source, fs=self.fs)

    def get_source_len(self, index):
        """Get source len.
        
            Args:
                index: TODO.
            """
        item = self.index_ds[index]
        return self.index_ds.get_source_len(item)

    def get_target_len(self, index):
        """Get target len.
        
            Args:
                index: TODO.
            """
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)

    def __len__(self):
        """Internal: len  ."""
        return len(self.index_ds)

    def __getitem__(self, index):
        """Internal: getitem  .
        
            Args:
                index: TODO.
            """
        item = self.index_ds[index]
        data_src = self._load_source(item)
        if self.preprocessor_speech:
            data_src = self.preprocessor_speech(data_src, fs=self.fs)

        speech, speech_lengths = extract_fbank(
            data_src, data_type=self.data_type, frontend=self.frontend, is_final=True
        )  # speech: [b, T, d]

        target = item["target"]
        if self.preprocessor_text:
            target = self.preprocessor_text(target)

        if self.tokenizer:
            ids = self.tokenizer.encode(target)
            text = torch.tensor(ids, dtype=torch.int64)
        else:
            ids = target
            text = ids
        ids_lengths = len(ids)
        text_lengths = torch.tensor([ids_lengths], dtype=torch.int32)

        return {
            "speech": speech[0, :, :],
            "speech_lengths": speech_lengths,
            "text": text,
            "text_lengths": text_lengths,
        }

    def collator(self, samples: list = None):
        """Collator.
        
            Args:
                samples: TODO.
            """
        outputs = {}
        for sample in samples:
            for key in sample.keys():
                if key not in outputs:
                    outputs[key] = []
                outputs[key].append(sample[key])

        for key, data_list in outputs.items():
            if isinstance(data_list[0], torch.Tensor):
                if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:

                    pad_value = self.int_pad_value
                else:
                    pad_value = self.float_pad_value

                outputs[key] = torch.nn.utils.rnn.pad_sequence(
                    data_list, batch_first=True, padding_value=pad_value
                )
        return outputs


@tables.register("dataset_classes", "AudioDatasetHotword")
class AudioDatasetHotword(AudioDataset):
    # for finetuning contextual_paraformer and seaco_paraformer
    def __init__(
        self,
        *args,
        seaco_id: bool = 0,
        **kwargs,
    ):
        """Initialize AudioDatasetHotword.
        
            Args:
                *args: Variable positional arguments.
                **kwargs: Additional keyword arguments.
            """
        super().__init__(*args, **kwargs)
        self.seaco_id = seaco_id

    def __getitem__(self, index):
        """Internal: getitem  .
        
            Args:
                index: TODO.
            """
        item = self.index_ds[index]
        # import pdb;
        # pdb.set_trace()
        source = item["source"]
        data_src = load_audio_text_image_video(source, fs=self.fs)
        if self.preprocessor_speech:
            data_src = self.preprocessor_speech(data_src, fs=self.fs)
        speech, speech_lengths = extract_fbank(
            data_src, data_type=self.data_type, frontend=self.frontend, is_final=True
        )  # speech: [b, T, d]

        target = item["target"]
        if self.preprocessor_text:
            target = self.preprocessor_text(target)
        if self.tokenizer:
            ids = self.tokenizer.encode(target)
            text = torch.tensor(ids, dtype=torch.int64)
        else:
            ids = target
            text = ids
        ids_lengths = len(ids)
        text_lengths = torch.tensor([ids_lengths], dtype=torch.int32)

        def generate_index(
            length,
            hotword_min_length=2,
            hotword_max_length=8,
            sample_rate=0.75,
            double_rate=0.1,
            pre_prob=0.0,
            pre_index=None,
            pre_hwlist=None,
        ):
            """Generate index.
            
                Args:
                    length: TODO.
                    hotword_min_length: TODO.
                    hotword_max_length: TODO.
                    sample_rate: TODO.
                    double_rate: TODO.
                    pre_prob: TODO.
                    pre_index: TODO.
                    pre_hwlist: TODO.
                """
            if length < hotword_min_length:
                return [-1]
            if random.random() < sample_rate:
                if pre_prob > 0 and random.random() < pre_prob and pre_index is not None:
                    return pre_index
                if length == hotword_min_length:
                    return [0, length - 1]
                elif (
                    random.random() < double_rate
                    and length > hotword_max_length + hotword_min_length + 2
                ):
                    # sample two hotwords in a sentence
                    _max_hw_length = min(hotword_max_length, length // 2)
                    # first hotword
                    start1 = random.randint(0, length // 3)
                    end1 = random.randint(
                        start1 + hotword_min_length - 1, start1 + _max_hw_length - 1
                    )
                    # second hotword
                    start2 = random.randint(end1 + 1, length - hotword_min_length)
                    end2 = random.randint(
                        min(length - 1, start2 + hotword_min_length - 1),
                        min(length - 1, start2 + hotword_max_length - 1),
                    )
                    return [start1, end1, start2, end2]
                else:  # single hotword
                    start = random.randint(0, length - hotword_min_length)
                    end = random.randint(
                        min(length - 1, start + hotword_min_length - 1),
                        min(length - 1, start + hotword_max_length - 1),
                    )
                    return [start, end]
            else:
                return [-1]

        hotword_indx = generate_index(text_lengths[0])
        return {
            "speech": speech[0, :, :],
            "speech_lengths": speech_lengths,
            "text": text,
            "text_lengths": text_lengths,
            "hotword_indx": hotword_indx,
            "seaco_id": self.seaco_id,
        }

    def collator(self, samples: list = None):
        """Collator.
        
            Args:
                samples: TODO.
            """
        outputs = {}
        hotword_indxs = []
        seaco_id = samples[0]["seaco_id"]
        for sample in samples:
            for key in sample.keys():
                if key == "seaco_id":
                    continue
                elif key == "hotword_indx":
                    hotword_indxs.append(sample[key])
                else:
                    if key not in outputs:
                        outputs[key] = []
                    outputs[key].append(sample[key])

        for key, data_list in outputs.items():
            if isinstance(data_list[0], torch.Tensor):
                if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:
                    pad_value = self.int_pad_value
                else:
                    pad_value = self.float_pad_value
                outputs[key] = torch.nn.utils.rnn.pad_sequence(
                    data_list, batch_first=True, padding_value=pad_value
                )

        hotword_list, hotword_lengths = [], []
        text = outputs["text"]
        seaco_label_pad = torch.ones_like(text) * -1 if seaco_id else None
        for b, (hotword_indx, one_text, length) in enumerate(
            zip(hotword_indxs, text, outputs["text_lengths"])
        ):
            length = length[0]
            if seaco_label_pad is not None:
                seaco_label_pad[b][:length] = seaco_id
            if hotword_indx[0] != -1:
                start, end = int(hotword_indx[0]), int(hotword_indx[1])
                hotword = one_text[start : end + 1]
                hotword_list.append(hotword)
                hotword_lengths.append(end - start + 1)
                if seaco_label_pad is not None:
                    seaco_label_pad[b][start : end + 1] = one_text[start : end + 1]
                if len(hotword_indx) == 4 and hotword_indx[2] != -1:
                    # the second hotword if exist
                    start, end = int(hotword_indx[2]), int(hotword_indx[3])
                    hotword_list.append(one_text[start : end + 1])
                    hotword_lengths.append(end - start + 1)
                    if seaco_label_pad is not None:
                        seaco_label_pad[b][start : end + 1] = one_text[start : end + 1]
        hotword_list.append(torch.tensor([1]))
        hotword_lengths.append(1)
        hotword_pad = torch.nn.utils.rnn.pad_sequence(
            hotword_list, batch_first=True, padding_value=0
        )
        outputs["hotword_pad"] = hotword_pad
        outputs["hotword_lengths"] = torch.tensor(hotword_lengths, dtype=torch.int32)
        if seaco_label_pad is not None:
            outputs["seaco_label_pad"] = seaco_label_pad
        return outputs
