import logging
import torch

from funasr.register import tables


# @tables.register("dataloader_classes", "DataloaderMapStyle")
def DataloaderMapStyle(frontend=None, tokenizer=None, **kwargs):
    # dataset
    """Dataloadermapstyle.
    
        Args:
            frontend: Audio frontend for feature extraction.
            tokenizer: Tokenizer instance for text encoding/decoding.
            **kwargs: Additional keyword arguments.
        """
    logging.info("Build dataloader")
    dataset_class = tables.dataset_classes.get(kwargs.get("dataset", "AudioDataset"))
    dataset_tr = dataset_class(
        kwargs.get("train_data_set_list"),
        frontend=frontend,
        tokenizer=tokenizer,
        is_training=True,
        **kwargs.get("dataset_conf"),
    )
    dataset_val = dataset_class(
        kwargs.get("valid_data_set_list"),
        frontend=frontend,
        tokenizer=tokenizer,
        is_training=False,
        **kwargs.get("dataset_conf"),
    )

    # dataloader
    batch_sampler = kwargs["dataset_conf"].get("batch_sampler", "BatchSampler")
    batch_sampler_val = None
    if batch_sampler is not None:
        batch_sampler_class = tables.batch_sampler_classes.get(batch_sampler)
        batch_sampler = batch_sampler_class(dataset_tr, **kwargs.get("dataset_conf"))
        batch_sampler_val = batch_sampler_class(
            dataset_val, is_training=False, **kwargs.get("dataset_conf")
        )

    dataloader_tr = torch.utils.data.DataLoader(
        dataset_tr, collate_fn=dataset_tr.collator, **batch_sampler
    )
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, collate_fn=dataset_val.collator, **batch_sampler_val
    )

    return dataloader_tr, dataloader_val


@tables.register("dataloader_classes", "DataloaderMapStyle")
class DataloaderMapStyle:
    def __init__(self, frontend=None, tokenizer=None, **kwargs):
        # dataset
        """Initialize DataloaderMapStyle.
        
            Args:
                frontend: Audio frontend for feature extraction.
                tokenizer: Tokenizer instance for text encoding/decoding.
                **kwargs: Additional keyword arguments.
            """
        logging.info("Build dataloader")

        dataset_class = tables.dataset_classes.get(kwargs.get("dataset", "AudioDataset"))
        dataset_tr = None
        # split dataset
        self.data_split_num = kwargs["dataset_conf"].get("data_split_num", 1)
        if self.data_split_num == 1:
            dataset_tr = dataset_class(
                kwargs.get("train_data_set_list"),
                frontend=frontend,
                tokenizer=tokenizer,
                is_training=True,
                **kwargs.get("dataset_conf"),
            )
        dataset_val = dataset_class(
            kwargs.get("valid_data_set_list"),
            frontend=frontend,
            tokenizer=tokenizer,
            is_training=False,
            **kwargs.get("dataset_conf"),
        )

        self.dataset_tr = dataset_tr
        self.dataset_val = dataset_val
        self.kwargs = kwargs

        self.dataset_class = dataset_class
        self.frontend = frontend
        self.tokenizer = tokenizer
        self.kwargs = kwargs

    def build_iter(self, epoch=0, data_split_i=0, start_step=0, **kwargs):

        # reload dataset slice
        """Build iter.
        
            Args:
                epoch: TODO.
                data_split_i: TODO.
                start_step: TODO.
                **kwargs: Additional keyword arguments.
            """
        if self.data_split_num > 1:
            del self.dataset_tr
            self.dataset_tr = self.dataset_class(
                self.kwargs.get("train_data_set_list"),
                frontend=self.frontend,
                tokenizer=self.tokenizer,
                is_training=True,
                **self.kwargs.get("dataset_conf"),
                data_split_i=data_split_i,
            )

        # dataloader
        batch_sampler = self.kwargs["dataset_conf"].get("batch_sampler", "BatchSampler")
        batch_sampler_val = None
        if batch_sampler is not None:
            batch_sampler_class = tables.batch_sampler_classes.get(batch_sampler)
            batch_sampler = batch_sampler_class(
                self.dataset_tr, start_step=start_step, **self.kwargs.get("dataset_conf")
            )
            batch_sampler_val = batch_sampler_class(
                self.dataset_val, is_training=False, **self.kwargs.get("dataset_conf")
            )

        batch_sampler["batch_sampler"].set_epoch(epoch)
        batch_sampler_val["batch_sampler"].set_epoch(epoch)
        dataloader_tr = torch.utils.data.DataLoader(
            self.dataset_tr, collate_fn=self.dataset_tr.collator, **batch_sampler
        )
        dataloader_val = torch.utils.data.DataLoader(
            self.dataset_val, collate_fn=self.dataset_val.collator, **batch_sampler_val
        )

        return dataloader_tr, dataloader_val


@tables.register("dataloader_classes", "DataloaderIterable")
def DataloaderIterable(frontend=None, tokenizer=None, **kwargs):
    """Dataloaderiterable.

        Args:
            frontend: Audio frontend for feature extraction.
            tokenizer: Tokenizer instance for text encoding/decoding.
            **kwargs: Additional keyword arguments.
        """
    logging.info("Build dataloader")
    dataset_class = tables.dataset_classes.get(kwargs.get("dataset", "LargeDataset"))
    dataset_tr = dataset_class(
        kwargs.get("train_data_set_list"),
        frontend=frontend,
        tokenizer=tokenizer,
        is_training=True,
        **kwargs.get("dataset_conf"),
    )
    dataset_val = dataset_class(
        kwargs.get("valid_data_set_list"),
        frontend=frontend,
        tokenizer=tokenizer,
        is_training=False,
        **kwargs.get("dataset_conf"),
    )

    return dataset_tr, dataset_val


class _SetEpochProxy:
    """让 IterableDataset 的 DataLoader 也能响应 trainer 的
    `dataloader.batch_sampler.set_epoch(epoch)`。

    内部把 set_epoch 直接转发到 dataset.set_epoch。
    """

    def __init__(self, dataset):
        self._dataset = dataset

    def set_epoch(self, epoch):
        """转发 set_epoch 到内部 dataset。"""
        if hasattr(self._dataset, "set_epoch"):
            self._dataset.set_epoch(epoch)


class _ShardLoaderWrapper:
    """对 torch.utils.data.DataLoader 的轻量包装。

    PyTorch 不允许在 DataLoader 实例化后再赋值 batch_sampler 属性，
    所以这里再包一层，把 trainer 期望的 `loader.batch_sampler.set_epoch(epoch)`
    路由到 dataset.set_epoch；其余属性/方法（__iter__、__len__、dataset 等）
    全部代理到内部的真实 DataLoader。
    """

    def __init__(self, loader, dataset, start_step=0):
        self._loader = loader
        self._start_step = max(0, int(start_step))
        self.batch_sampler = _SetEpochProxy(dataset)

    def __iter__(self):
        for batch_idx, batch in enumerate(self._loader):
            if batch_idx < self._start_step:
                continue
            yield batch

    def __len__(self):
        return max(0, len(self._loader) - self._start_step)

    def __getattr__(self, name):
        # __getattr__ 只在常规属性查找失败时调用，
        # 所以 batch_sampler / _loader 等已显式定义的字段不会进来。
        return getattr(self._loader, name)


@tables.register("dataloader_classes", "DataloaderShard")
class DataloaderShard:
    """webdataset-style shard 的 IterableDataset 适配 dataloader。

    与 DataloaderMapStyle 接口一致：暴露 data_split_num 和 build_iter()，
    但 dataloader_train/val 是基于 IterableDataset 的 DataLoader，并把
    `dataloader.batch_sampler` 伪装成一个能接受 set_epoch 的代理对象，
    `__len__` 由数据集的样本数估算除以 batch_size 得到。
    """

    def __init__(self, frontend=None, tokenizer=None, **kwargs):
        logging.info("Build DataloaderShard")
        dataset_class = tables.dataset_classes.get(
            kwargs.get("dataset", "ShardAudioDataset")
        )

        ds_conf = kwargs.get("dataset_conf", {}) or {}
        self.batch_size = int(ds_conf.get("batch_size", 1))
        self.num_workers = int(ds_conf.get("num_workers", 2))
        self.pin_memory = bool(ds_conf.get("pin_memory", True))
        self.prefetch_factor = int(ds_conf.get("prefetch_factor", 2))
        # 注意：IterableDataset 下若 persistent_workers=True，主进程的 set_epoch
        # 不会传给已 fork 的 worker；默认 False 让每个 epoch 重新 fork 一次。
        self.persistent_workers = bool(ds_conf.get("persistent_workers", False))

        self.dataset_tr = dataset_class(
            kwargs.get("train_data_set_list"),
            frontend=frontend,
            tokenizer=tokenizer,
            is_training=True,
            **ds_conf,
        )
        self.dataset_val = dataset_class(
            kwargs.get("valid_data_set_list"),
            frontend=frontend,
            tokenizer=tokenizer,
            is_training=False,
            **ds_conf,
        )

        # Iterable shard 数据集不做 split 切片，data_split_num 固定为 1
        self.data_split_num = 1

    def _build_loader(self, dataset, is_training: bool, start_step=0):
        kwargs = dict(
            batch_size=self.batch_size,
            collate_fn=dataset.collator,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=is_training,
        )
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
            kwargs["persistent_workers"] = self.persistent_workers
        loader = torch.utils.data.DataLoader(dataset, **kwargs)
        # 用 wrapper 暴露 batch_sampler.set_epoch(...)
        return _ShardLoaderWrapper(loader, dataset, start_step=start_step)

    def build_iter(self, epoch=0, data_split_i=0, start_step=0, **kwargs):
        """构建训练和验证的 DataLoader 迭代器。"""
        if start_step > 0:
            logging.info(f"DataloaderShard resume from batch step: {start_step}")
        self.dataset_tr.set_epoch(epoch)
        self.dataset_val.set_epoch(epoch)
        loader_tr = self._build_loader(
            self.dataset_tr, is_training=True, start_step=start_step
        )
        loader_val = self._build_loader(self.dataset_val, is_training=False)
        return loader_tr, loader_val
