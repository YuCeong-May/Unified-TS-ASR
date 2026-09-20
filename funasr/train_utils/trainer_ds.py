import math
import os
import time
import torch
import logging
import traceback
from tqdm import tqdm
from datetime import datetime
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from contextlib import nullcontext, contextmanager
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from funasr.train_utils.device_funcs import to_device
from funasr.train_utils.recursive_op import recursive_average
from funasr.train_utils.average_nbest_models import average_checkpoints
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
import funasr.utils.misc as misc_utils

try:
    import wandb
except:
    wandb = None


@contextmanager
def maybe_autocast(dtype=None, use_deepspeed=False):
    """Maybe autocast.

        Args:
            dtype: TODO.
            use_deepspeed: TODO.
        """
    if use_deepspeed:
        # DeepSpeed 通过 bf16/fp16 config 自行管理混精度；
        # engine.forward 内部已有 autocast_if_enabled 处理，外面不要重复开 autocast，
        # 否则与 engine 内部的 autocast 冲突（engine 发现外层已 enabled 时会按自己的
        # torch_autocast 配置覆盖或关掉，导致 input 未 cast 而 weight 已 cast → dtype mismatch）。
        yield
    else:
        if dtype == torch.float16 or dtype == torch.bfloat16:
            with autocast(enabled=True, dtype=dtype):
                yield
        else:
            yield


class Trainer:
    """
    A simple trainer class for training a PyTorch model, saving checkpoints at the end of each epoch,
    and optionally resuming from a saved checkpoint.

    Attributes:
        max_epoch (int): Maximum number of epochs for training.
        model (torch.nn.Module): The model to be trained.
        optim (torch.optim.Optimizer): The optimizer to use for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        dataloader_train (torch.utils.data.DataLoader): DataLoader for the training dataset.
        dataloader_val (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        output_dir (str): Directory where model checkpoints will be saved.
        resume (str, optional): Path to a checkpoint to resume training from.
    """

    def __init__(
        self,
        rank=0,
        local_rank=0,
        world_size=1,
        use_ddp: bool = False,
        use_fsdp: bool = False,
        use_fp16: bool = False,
        use_bf16: bool = False,
        use_deepspeed: bool = False,
        output_dir: str = "./",
        **kwargs,
    ):
        """
        Initializes the Trainer class with the model, optimizer, scheduler, dataloaders, and other settings.

        Args:
            model (torch.nn.Module): The model to be trained.
            optim (torch.optim.Optimizer): The optimizer to use for training.
            scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
            dataloader_train (torch.utils.data.DataLoader): The DataLoader for the training dataset.
            dataloader_val (torch.utils.data.DataLoader): The DataLoader for the validation dataset.
            **kwargs: Additional keyword arguments:
                      max_epoch (int): The maximum number of epochs for training.
                      output_dir (str): The directory where model checkpoints will be saved. Default is './'.
                      resume (str, optional): The file path to a checkpoint to resume training from.
        """
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.use_ddp = use_ddp
        self.use_fsdp = use_fsdp

        self.device = kwargs.get("device", "cuda")

        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.resume = kwargs.get("resume", True)
        self.start_epoch = 0
        self.max_epoch = kwargs.get("max_epoch", 100)

        # self.kwargs = kwargs
        self.log_interval = kwargs.get("log_interval", 50)
        self.batch_total = 0
        # 训练中因 forward/CUDA 异常或非有限 loss 而被全体 rank 一起跳过的 batch 计数
        self.skipped_batches = 0
        # 连续跳过计数：若连续多个 batch 都失败，通常意味着 CUDA 上下文已 sticky 损坏，
        # 进程内跳过无意义，需要人工重启。超过阈值时会打醒目告警。
        self.consecutive_skips = 0
        self.max_consecutive_skips = int(kwargs.get("max_consecutive_skips", 20))
        self.dtype = torch.float32
        self.use_fp16 = use_fp16
        self.use_bf16 = use_bf16
        if self.use_fp16:
            self.dtype = torch.float16
        if self.use_bf16:
            self.dtype = torch.bfloat16
        self.save_checkpoint_interval = kwargs.get("save_checkpoint_interval", 5000)
        self.validate_interval = kwargs.get("validate_interval", 5000)
        self.keep_nbest_models = kwargs.get("keep_nbest_models", 500)
        self.avg_keep_nbest_models_type = kwargs.get("avg_keep_nbest_models_type", "acc")
        self.avg_nbest_model = kwargs.get("avg_nbest_model", 10)
        self.accum_grad = kwargs.get("accum_grad", 1)
        self.grad_clip = kwargs.get("grad_clip", 10.0)
        self.grad_clip_type = kwargs.get("grad_clip_type", 2.0)

        self.train_acc_avg = 0.0
        self.train_loss_avg = 0.0
        self.val_acc_avg = 0.0
        self.val_loss_avg = 0.0
        self.best_acc_idx = 0
        self.saved_ckpts = {}
        self.step_or_epoch = -1
        self.best_step_or_epoch = ""
        self.val_acc_step_or_epoch = {}
        self.val_loss_step_or_epoch = {}

        self.reset_gpu_cache = kwargs.get("reset_gpu_cache", False)
        self.start_data_split_i = 0
        self.start_step = 0
        self.step_in_epoch = 0
        self.use_wandb = kwargs.get("use_wandb", False)
        if self.use_wandb:
            wandb.login(key=kwargs.get("wandb_token"))
            wandb.init(
                config=kwargs,
                project=kwargs.get("wandb_project", "my_project"),
                entity=kwargs.get("wandb_team", "my_team"),
                name=kwargs.get("wandb_exp_name", "my_exp"),
                dir=output_dir,
                job_type="training",
                reinit=True,
            )
        tensorboard_dir = os.path.join(output_dir, "tensorboard")
        if rank == 0:
            os.makedirs(tensorboard_dir, exist_ok=True)
            try:
                from tensorboardX import SummaryWriter

                self.writer = SummaryWriter(tensorboard_dir)
            except:
                self.writer = None
        else:
            self.writer = None

        self.use_deepspeed = use_deepspeed
        self.deepspeed_config = kwargs.get("deepspeed_config", "")
        excludes = kwargs.get("excludes", None)
        if excludes is not None:
            if isinstance(excludes, str):
                excludes = excludes.split(",")
        self.excludes = excludes
        effective_save_name_excludes = kwargs.get("effective_save_name_excludes", None)
        if effective_save_name_excludes is not None:
            if isinstance(effective_save_name_excludes, str):
                effective_save_name_excludes = effective_save_name_excludes.split(",")
        self.effective_save_name_excludes = effective_save_name_excludes

        # ckpt 异步备份到 BOS：本地照常保存、不删，云盘只是备份。
        # train_conf:
        #   ckpt_bos_backup_uri: bos://example-bucket/checkpoints/run/  # 结尾的 / 见下
        #   ckpt_bos_backup_bcecmd: /path/to/bcecmd
        self.ckpt_backup = None
        backup_uri = kwargs.get("ckpt_bos_backup_uri", None)
        # 所有 rank 都要知道开关（上传前的 barrier 必须全员参与），但只有 rank0 真的传
        self.ckpt_backup_enabled = bool(backup_uri)
        if backup_uri and self.rank == 0:
            from funasr.train_utils.ckpt_backup import CkptBackup

            self.ckpt_backup = CkptBackup(
                output_dir=output_dir,
                uri=backup_uri,
                bcecmd=kwargs.get("ckpt_bos_backup_bcecmd", None),
                concurrency=int(kwargs.get("ckpt_bos_backup_concurrency", 10)),
            )

    def _update_ckpt_pointer(self, link_name, target_name):
        """把 output_dir/<link_name> 更新为指向同目录下 <target_name> 的相对符号链接。

        `model.pt`（latest）和 `model.pt.best` 原来是把同一份 state 重复存一遍，
        deepspeed 下一次 checkpoint 要写 3 份（每份 3.3GB）。改成符号链接后只写真实 ckpt
        一份，指针切换是原子的 rename，且用相对路径，整个 exp 目录可以整体搬移。

        Args:
            link_name: 指针名，如 "model.pt" / "model.pt.best"。
            target_name: 真实 ckpt 名（同目录下），如 "model.pt.ep3.4000"。
        """
        link = os.path.join(self.output_dir, link_name)
        target = os.path.join(self.output_dir, target_name)
        if not os.path.exists(target):
            logging.warning(f"ckpt 指针 {link_name} 的目标 {target_name} 不存在，跳过更新")
            return

        tmp = f"{link}.tmp_link"
        if os.path.islink(tmp):
            os.unlink(tmp)
        os.symlink(target_name, tmp)
        if os.path.islink(link):
            os.replace(tmp, link)  # 原子替换旧链接
        else:
            if os.path.exists(link):
                # 老版本留下的真实拷贝（内容与某个 model.pt.ep* 完全相同），删掉换成链接
                logging.info(f"用符号链接替换重复的 ckpt 拷贝: {link}")
                misc_utils.smart_remove(link)
            os.replace(tmp, link)
        logging.info(f"ckpt 指针: {link_name} -> {target_name}")

    def _remove_worst_ckpt(self, protected=()):
        """按 keep_nbest_models 删掉指标最差的 ckpt，跳过被符号链接引用的那些。

        Args:
            protected: 不允许删除的 ckpt 名集合（model.pt / model.pt.best 的目标），
                否则链接会悬空、断掉 resume 与 eval。
        """
        if self.keep_nbest_models <= 0 or len(self.saved_ckpts) <= self.keep_nbest_models:
            return
        candidates = [k for k in self.saved_ckpts if k not in set(protected)]
        if not candidates:
            return
        if self.avg_keep_nbest_models_type == "acc":
            key = min(candidates, key=self.saved_ckpts.get)
        else:
            key = max(candidates, key=self.saved_ckpts.get)
        del self.saved_ckpts[key]
        filename = os.path.join(self.output_dir, key)
        logging.info(f"Delete: {filename}")
        if os.path.exists(filename):
            misc_utils.smart_remove(filename)
        if self.ckpt_backup is not None:
            # 本地淘汰的同时把云盘备份也删掉，保持两边都是 keep_nbest_models 份
            self.ckpt_backup.remove(key)

    def save_checkpoint(
        self,
        epoch,
        step=None,
        model=None,
        optim=None,
        scheduler=None,
        scaler=None,
        step_in_epoch=None,
        **kwargs,
    ):
        """
        Saves a checkpoint containing the model's state, the optimizer's state,
        and the scheduler's state at the end of the given epoch. This method is
        intended to be called at the end of each epoch to save the training progress.

        Args:
            epoch (int): The epoch number at which the checkpoint is being saved.
        """
        if self.use_ddp or self.use_fsdp:
            dist.barrier()
        step_in_epoch = None if step is None else step_in_epoch
        if self.use_deepspeed:

            logging.info(f"Save checkpoint: {epoch}, rank: {self.local_rank}\n")
            # self.step_or_epoch += 1
            state = {
                "epoch": epoch,
                # "state_dict": model.state_dict(),
                # "optimizer": optim.state_dict(),
                # "scheduler": scheduler.state_dict(),
                "saved_ckpts": self.saved_ckpts,
                "val_acc_step_or_epoch": self.val_acc_step_or_epoch,
                "val_loss_step_or_epoch": self.val_loss_step_or_epoch,
                "best_step_or_epoch": self.best_step_or_epoch,
                "avg_keep_nbest_models_type": self.avg_keep_nbest_models_type,
                "step": step,
                "step_in_epoch": step_in_epoch,
                "data_split_i": kwargs.get("data_split_i", 0),
                "data_split_num": kwargs.get("data_split_num", 1),
                "batch_total": self.batch_total,
                "train_loss_avg": kwargs.get("train_loss_avg", 0),
                "train_acc_avg": kwargs.get("train_acc_avg", 0),
            }
            step = step_in_epoch
            if hasattr(model, "module"):
                state["state_dict"] = model.module.state_dict()

            if scaler:
                state["scaler_state"] = scaler.state_dict()
            # Create output directory if it does not exist
            os.makedirs(self.output_dir, exist_ok=True)
            if step is None:
                ckpt_name = f"model.pt.ep{epoch}"
            else:
                ckpt_name = f"model.pt.ep{epoch}.{step}"
            filename = os.path.join(self.output_dir, ckpt_name)

            # torch.save(state, filename)
            with torch.no_grad():
                model.save_checkpoint(save_dir=self.output_dir, tag=ckpt_name, client_state=state)
            logging.info(f"\nCheckpoint saved to {filename}\n")
            # tag 目录里每个 rank 各写一份 optim_states，rank0 必须等大家都写完再上传
            if self.ckpt_backup_enabled and dist.is_available() and dist.is_initialized():
                dist.barrier()
            if self.ckpt_backup is not None:
                self.ckpt_backup.upload(ckpt_name)
            # 只写这一份真实 ckpt：deepspeed 自己维护的 output_dir/latest 里现在是真实 tag
            # （如 model.pt.ep3.4000），不再是含义模糊的 "model.pt"。
            # model.pt 变成指向最新 ckpt 的符号链接，供 resume / average_nbest 使用。
            if self.rank == 0:
                self._update_ckpt_pointer("model.pt", ckpt_name)
            if self.best_step_or_epoch == "":
                self.best_step_or_epoch = ckpt_name

            if self.avg_keep_nbest_models_type == "acc":
                if (
                    self.val_acc_step_or_epoch[ckpt_name]
                    >= self.val_acc_step_or_epoch[self.best_step_or_epoch]
                ):
                    self.best_step_or_epoch = ckpt_name
                    if self.rank == 0:
                        self._update_ckpt_pointer("model.pt.best", ckpt_name)
                    logging.info(
                        f"Update best acc: {self.val_acc_step_or_epoch[self.best_step_or_epoch]:.4f}, "
                        f"model.pt.best -> {ckpt_name}"
                    )
                else:
                    logging.info(
                        f"No improvement in acc: {self.val_acc_step_or_epoch[ckpt_name]:.4f} < {self.val_acc_step_or_epoch[self.best_step_or_epoch]:.4f}, {os.path.join(self.output_dir, self.best_step_or_epoch)}"
                    )
            elif self.avg_keep_nbest_models_type == "loss":
                if (
                    self.val_loss_step_or_epoch[ckpt_name]
                    <= self.val_loss_step_or_epoch[self.best_step_or_epoch]
                ):
                    self.best_step_or_epoch = ckpt_name
                    if self.rank == 0:
                        self._update_ckpt_pointer("model.pt.best", ckpt_name)
                    logging.info(
                        f"Update best loss: {self.val_loss_step_or_epoch[self.best_step_or_epoch]:.4f}, "
                        f"model.pt.best -> {ckpt_name}"
                    )
                else:
                    logging.info(
                        f"No improvement in loss: {self.val_loss_step_or_epoch[ckpt_name]:.4f} > {self.val_loss_step_or_epoch[self.best_step_or_epoch]:.4f}, {os.path.join(self.output_dir, self.best_step_or_epoch)}"
                    )
            else:
                print("Undo")
            if self.rank == 0:
                self.saved_ckpts[ckpt_name] = getattr(
                    self, f"val_{self.avg_keep_nbest_models_type}_step_or_epoch"
                )[ckpt_name]
                # ckpt_name 是 model.pt 的目标、best_step_or_epoch 是 model.pt.best 的目标，
                # 删了它们会让链接悬空。
                self._remove_worst_ckpt(protected=(ckpt_name, self.best_step_or_epoch))

        elif self.use_fsdp:
            raise NotImplementedError(
                "FSDP checkpoint saving is not yet implemented. Use DDP or DeepSpeed instead."
            )
        elif self.rank == 0:
            logging.info(
                f"Save checkpoint: {epoch}, rank: {self.rank}, local_rank: {self.local_rank}\n"
            )
            # self.step_or_epoch += 1
            state = {
                "epoch": epoch,
                "optimizer": optim.state_dict(),
                "scheduler": scheduler.state_dict(),
                "saved_ckpts": self.saved_ckpts,
                "val_acc_step_or_epoch": self.val_acc_step_or_epoch,
                "val_loss_step_or_epoch": self.val_loss_step_or_epoch,
                "best_step_or_epoch": self.best_step_or_epoch,
                "avg_keep_nbest_models_type": self.avg_keep_nbest_models_type,
                "step": step,
                "step_in_epoch": step_in_epoch,
                "data_split_i": kwargs.get("data_split_i", 0),
                "data_split_num": kwargs.get("data_split_num", 1),
                "batch_total": self.batch_total,
                "train_loss_avg": kwargs.get("train_loss_avg", 0),
                "train_acc_avg": kwargs.get("train_acc_avg", 0),
            }
            step = step_in_epoch
            if hasattr(model, "module"):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            if self.effective_save_name_excludes is not None:
                logging.info(f"effective_save_name_excludes: {self.effective_save_name_excludes}")
                dst_state_dict = {}
                for k in state_dict.keys():
                    for k_ex in self.effective_save_name_excludes:
                        k_tmp = k.replace("module.", "")
                        if k.startswith(k_ex):
                            logging.info(f"key: {k} matching: {k_ex}, not save it")
                            break
                    else:
                        dst_state_dict[k] = state_dict[k]
                state["state_dict"] = dst_state_dict
            else:
                state["state_dict"] = state_dict

            if scaler:
                state["scaler_state"] = scaler.state_dict()
            # Create output directory if it does not exist
            os.makedirs(self.output_dir, exist_ok=True)
            if step is None:
                ckpt_name = f"model.pt.ep{epoch}"
            else:
                ckpt_name = f"model.pt.ep{epoch}.{step}"
            filename = os.path.join(self.output_dir, ckpt_name)
            torch.save(state, filename)

            logging.info(f"\nCheckpoint saved to {filename}\n")
            if self.ckpt_backup is not None:
                self.ckpt_backup.upload(ckpt_name)
            # model.pt / model.pt.best 都用符号链接指向真实 ckpt，不再重复存同一份 state
            self._update_ckpt_pointer("model.pt", ckpt_name)
            if self.best_step_or_epoch == "":
                self.best_step_or_epoch = ckpt_name

            if self.avg_keep_nbest_models_type == "acc":
                if (
                    self.val_acc_step_or_epoch[ckpt_name]
                    >= self.val_acc_step_or_epoch[self.best_step_or_epoch]
                ):
                    self.best_step_or_epoch = ckpt_name
                    self._update_ckpt_pointer("model.pt.best", ckpt_name)
                    logging.info(
                        f"Update best acc: {self.val_acc_step_or_epoch[self.best_step_or_epoch]:.4f}, "
                        f"model.pt.best -> {ckpt_name}"
                    )
                else:
                    logging.info(
                        f"No improvement in acc: {self.val_acc_step_or_epoch[ckpt_name]:.4f} < {self.val_acc_step_or_epoch[self.best_step_or_epoch]:.4f}, {os.path.join(self.output_dir, self.best_step_or_epoch)}"
                    )
            elif self.avg_keep_nbest_models_type == "loss":
                if (
                    self.val_loss_step_or_epoch[ckpt_name]
                    <= self.val_loss_step_or_epoch[self.best_step_or_epoch]
                ):
                    self.best_step_or_epoch = ckpt_name
                    self._update_ckpt_pointer("model.pt.best", ckpt_name)
                    logging.info(
                        f"Update best loss: {self.val_loss_step_or_epoch[self.best_step_or_epoch]:.4f}, "
                        f"model.pt.best -> {ckpt_name}"
                    )
                else:
                    logging.info(
                        f"No improvement in loss: {self.val_loss_step_or_epoch[ckpt_name]:.4f} > {self.val_loss_step_or_epoch[self.best_step_or_epoch]:.4f}, {os.path.join(self.output_dir, self.best_step_or_epoch)}"
                    )
            else:
                print("Undo")
            self.saved_ckpts[ckpt_name] = getattr(
                self, f"val_{self.avg_keep_nbest_models_type}_step_or_epoch"
            )[ckpt_name]
            self._remove_worst_ckpt(protected=(ckpt_name, self.best_step_or_epoch))

        if self.use_ddp or self.use_fsdp:
            dist.barrier()

    def resume_checkpoint(
        self,
        model=None,
        optim=None,
        scheduler=None,
        scaler=None,
    ):
        """
        Resumes training from a checkpoint at the given file path.
        Loads the model's state, the optimizer's state, and the scheduler's state.

        Args:
            resume_path (str): The file path to the checkpoint to resume from.
        """
        if self.resume:

            if self.use_deepspeed:
                ckpt = os.path.join(self.output_dir, "model.pt")
                if os.path.exists(ckpt):
                    _, checkpoint = model.load_checkpoint(self.output_dir, "model.pt")
                    self.start_epoch = checkpoint["epoch"]
                    self.saved_ckpts = checkpoint["saved_ckpts"]
                    self.val_acc_step_or_epoch = (
                        checkpoint["val_acc_step_or_epoch"]
                        if "val_acc_step_or_epoch" in checkpoint
                        else {}
                    )
                    self.val_loss_step_or_epoch = (
                        checkpoint["val_loss_step_or_epoch"]
                        if "val_loss_step_or_epoch" in checkpoint
                        else {}
                    )
                    self.best_step_or_epoch = (
                        checkpoint["best_step_or_epoch"]
                        if "best_step_or_epoch" in checkpoint
                        else ""
                    )
                    self.start_data_split_i = (
                        checkpoint["data_split_i"] if "data_split_i" in checkpoint else 0
                    )
                    self.batch_total = (
                        checkpoint["batch_total"] if "batch_total" in checkpoint else 0
                    )
                    self.start_step = checkpoint["step"] if "step" in checkpoint else 0
                    self.start_step = 0 if self.start_step is None else self.start_step
                    self.step_in_epoch = (
                        checkpoint["step_in_epoch"] if "step_in_epoch" in checkpoint else 0
                    )
                    self.step_in_epoch = 0 if self.step_in_epoch is None else self.step_in_epoch
                    print(checkpoint["train_acc_avg"])
                    self.train_acc_avg = (
                        checkpoint["train_acc_avg"] if "train_acc_avg" in checkpoint else 0
                    )
                    self.train_loss_avg = (
                        checkpoint["train_loss_avg"] if "train_loss_avg" in checkpoint else 0
                    )
                    model.to(self.device)
                    print(f"Checkpoint loaded successfully from '{ckpt}'")
                else:
                    print(f"No checkpoint found at '{ckpt}', does not resume status!")
            else:

                ckpt = os.path.join(self.output_dir, "model.pt")
                if os.path.isfile(ckpt):
                    checkpoint = torch.load(ckpt, map_location="cpu")
                    self.start_epoch = checkpoint["epoch"]
                    # self.model.load_state_dict(checkpoint['state_dict'])
                    src_state = checkpoint["state_dict"]
                    dst_state = model.state_dict()
                    for k in dst_state.keys():
                        excludes_flag = False
                        if self.excludes is not None:
                            for k_ex in self.excludes:
                                k_tmp = k.replace("module.", "")
                                if k_tmp.startswith(k_ex):
                                    logging.info(f"key: {k} matching: {k_ex}, excluded")
                                    excludes_flag = True
                                    break
                        if excludes_flag:
                            continue
                        if not k.startswith("module.") and "module." + k in src_state.keys():
                            k_ddp = "module." + k
                        elif k.startswith("module.") and "module." + k not in src_state.keys():
                            k_ddp = k.replace("module.", "", 1)
                        else:
                            k_ddp = k
                        if k_ddp in src_state.keys():
                            dst_state[k] = src_state[k_ddp]
                        else:
                            print(f"Miss key in ckpt: model: {k}, ckpt: {k_ddp}")

                    model.load_state_dict(dst_state)
                    optim.load_state_dict(checkpoint["optimizer"])
                    scheduler.load_state_dict(checkpoint["scheduler"])
                    if scaler is not None and "scaler_state" in checkpoint:
                        scaler.load_state_dict(checkpoint["scaler_state"])

                    self.saved_ckpts = checkpoint["saved_ckpts"]
                    self.val_acc_step_or_epoch = (
                        checkpoint["val_acc_step_or_epoch"]
                        if "val_acc_step_or_epoch" in checkpoint
                        else {}
                    )
                    self.val_loss_step_or_epoch = (
                        checkpoint["val_loss_step_or_epoch"]
                        if "val_loss_step_or_epoch" in checkpoint
                        else {}
                    )
                    self.best_step_or_epoch = (
                        checkpoint["best_step_or_epoch"]
                        if "best_step_or_epoch" in checkpoint
                        else ""
                    )
                    self.start_data_split_i = (
                        checkpoint["data_split_i"] if "data_split_i" in checkpoint else 0
                    )
                    self.batch_total = (
                        checkpoint["batch_total"] if "batch_total" in checkpoint else 0
                    )
                    self.start_step = checkpoint["step"] if "step" in checkpoint else 0
                    self.start_step = 0 if self.start_step is None else self.start_step
                    self.step_in_epoch = (
                        checkpoint["step_in_epoch"] if "step_in_epoch" in checkpoint else 0
                    )
                    self.step_in_epoch = 0 if self.step_in_epoch is None else self.step_in_epoch
                    print(checkpoint["train_acc_avg"])
                    self.train_acc_avg = (
                        checkpoint["train_acc_avg"] if "train_acc_avg" in checkpoint else 0
                    )
                    self.train_loss_avg = (
                        checkpoint["train_loss_avg"] if "train_loss_avg" in checkpoint else 0
                    )
                    model.to(self.device)
                    print(f"Checkpoint loaded successfully from '{ckpt}'")
                else:
                    print(f"No checkpoint found at '{ckpt}', does not resume status!")

        if self.use_ddp or self.use_fsdp:
            dist.barrier()

    def train_epoch(
        self,
        model=None,
        optim=None,
        scheduler=None,
        scaler=None,
        dataloader_train=None,
        dataloader_val=None,
        epoch=None,
        **kwargs,
    ):
        """
        Defines the training process for a single epoch with gradient accumulation.
        Args:
            epoch (int): The current epoch number.
        """
        if self.use_ddp or self.use_fsdp or self.use_deepspeed:
            dist.barrier()
        logging.info(f"Train epoch: {epoch}, rank: {self.rank}\n")
        model.train()

        # Set the number of steps for gradient accumulation
        accum_grad = self.accum_grad
        # Initialize the gradient accumulation
        optim.zero_grad()
        speed_stats = {}

        # epoch 边界同步标志：在任一 rank 数据耗尽时通知全体一起退出，
        # 避免 IterableDataset 各 rank batch 数不等导致的 NCCL collective 错位死锁。
        iterator_stop = torch.zeros(1, device=self.device)
        use_dist = self.use_ddp or self.use_fsdp or self.use_deepspeed

        dataloader_train.batch_sampler.set_epoch(epoch)
        time_beg = time.perf_counter()
        time5 = time_beg
        data_iter = iter(dataloader_train)
        batch_idx = -1
        while True:
            # 先取本地下一个 batch；取不到则只标记本地耗尽，不立即 break。
            try:
                batch = next(data_iter)
                local_stop = 0
            except StopIteration:
                batch = None
                local_stop = 1

            # 每个 step 做一次轻量(Numel=1)的 allreduce 探测：只要有一个 rank 耗尽，
            # 全体在同一 step 一起 break，保证梯度/loss 等后续 collective 永远对齐。
            if use_dist:
                iterator_stop.fill_(local_stop)
                dist.all_reduce(iterator_stop, op=dist.ReduceOp.SUM)
                if iterator_stop.item() > 0:
                    break
            elif local_stop:
                break

            batch_idx += 1
            self.batch_total += 1
            self.step_in_epoch += 1
            loss_dict = {
                "speed_stats": {},
                "epoch": epoch,
                "batch_idx": batch_idx,
                "data_split_i": kwargs.get("data_split_i", 0),
                "data_split_num": kwargs.get("data_split_num", 1),
                "log_step": batch_idx + kwargs.get("start_step", 0),
                "batch_total": self.batch_total,
                "step_in_epoch": self.step_in_epoch,
            }

            time1 = time.perf_counter()
            loss_dict["speed_stats"]["data_load"] = f"{time1-time_beg:0.3f}"

            batch = to_device(batch, self.device, non_blocking=True)

            my_context = nullcontext
            if self.use_ddp or self.use_fsdp:
                # 梯度累积窗口内只在**最后一个** microstep 打开梯度同步，判据必须与
                # update_step 的 `(batch_idx + 1) % accum_grad == 0` 完全对齐。
                # batch_idx 从 0 起算：原来的 `batch_idx % accum_grad != 0` 把同步放在了窗口的
                # 第 0 个 microstep（此时 grad 刚被 set_to_none 清空），于是只有 1/accum_grad 的
                # 梯度做了跨 rank 平均，其余全是本地梯度 -> 各 rank 更新量不同、参数静默漂移，
                # 有效 batch 只有设计值的 ~31%（4 卡 x accum 4）。
                if (batch_idx + 1) % accum_grad != 0:
                    my_context = model.no_sync

            # ============ 单 batch 异常保护 ============
            # 目标：某个 batch 触发 forward/CUDA 异常（或产生 nan/inf loss）时，不让整个训练崩掉，
            # 而是打印详细错误日志后跳过该 batch 继续训练。
            # 分布式关键点：必须"全体 rank 在同一 step 一起跳过"。若某 rank 跳过 backward/step 而
            # 其他 rank 继续，会造成 NCCL collective 错位死锁。因此在 backward 之前用 all_reduce
            # 同步一个错误标志（此刻没有任何 rank 做过 backward，要么全体 backward、要么全体跳过）。
            batch_err = torch.zeros(1, device=self.device)
            err_tb = None
            time2 = time.perf_counter()
            skip_batch = False
            with my_context():
                try:
                    self.forward_step(model, batch, loss_dict=loss_dict)
                    _loss = loss_dict.get("loss", None)
                    if _loss is None or not torch.isfinite(_loss).all():
                        batch_err.fill_(1)
                        err_tb = f"non-finite or missing loss: {_loss}"
                except Exception:
                    batch_err.fill_(1)
                    err_tb = traceback.format_exc()

                time3 = time.perf_counter()

                # 跨 rank 同步错误标志（backward 之前）
                if use_dist:
                    dist.all_reduce(batch_err, op=dist.ReduceOp.SUM)

                if batch_err.item() > 0:
                    skip_batch = True
                else:
                    loss_dict["speed_stats"]["forward_time"] = f"{time3 - time2:0.3f}"
                    self.backward_step(model, scaler, loss_dict=loss_dict)
                    time4 = time.perf_counter()
                    loss_dict["speed_stats"]["backward_time"] = f"{time4 - time3:0.3f}"

            if skip_batch:
                self.skipped_batches += 1
                self.consecutive_skips += 1
                # 详细错误日志：dump 出错 batch 的形状/长度/token 范围 + 完整 traceback
                logging.error(
                    f"[SKIP BATCH] rank={self.rank} epoch={epoch} "
                    f"batch_idx={batch_idx} step_in_epoch={self.step_in_epoch} "
                    f"batch_total={self.batch_total} "
                    f"(skipped_total={self.skipped_batches}, consecutive={self.consecutive_skips})\n"
                    f"  batch_detail: {self._describe_batch(batch)}\n"
                    f"  local_error={'YES' if err_tb else 'no (synced from another rank)'}"
                    + (f"\n  traceback:\n{err_tb}" if err_tb else "")
                )
                # 尝试清理 CUDA：若为 sticky 上下文损坏，这里会再次抛错，说明进程内跳过救不回来。
                try:
                    torch.cuda.synchronize()
                except Exception:
                    logging.error(
                        f"[SKIP BATCH] rank={self.rank} torch.cuda.synchronize() 仍失败，"
                        f"CUDA 上下文可能已 sticky 损坏，进程内跳过无法恢复，需要重启训练进程：\n"
                        f"{traceback.format_exc()}"
                    )
                if self.consecutive_skips >= self.max_consecutive_skips:
                    logging.error(
                        f"[SKIP BATCH] rank={self.rank} 已连续跳过 {self.consecutive_skips} 个 batch "
                        f"(阈值 {self.max_consecutive_skips})，几乎可以确定 CUDA 上下文已损坏，"
                        f"继续跳过无意义。请中止并重启训练（会自动从最近 checkpoint 恢复）。"
                    )
                # 不清空已累积的合法微批梯度（DeepSpeed 内部 micro_step 全体一致推进），直接下一个 batch。
                time_beg = time.perf_counter()
                continue

            # 本 batch 正常，重置连续跳过计数
            self.consecutive_skips = 0

            self.update_step(model, optim, scheduler, scaler, loss_dict=loss_dict)
            total_time = f"{(time.perf_counter() - time5):0.3f}"
            time5 = time.perf_counter()

            loss_dict["speed_stats"]["optim_time"] = f"{time5 - time4:0.3f}"

            loss_dict["speed_stats"]["total_time"] = total_time

            loss_dict["lr"] = scheduler.get_last_lr()[0]
            # step_in_slice 的分子是绝对位置(batch_idx + start_step)，所以分母也要用整个 epoch 的
            # 长度而非续跑剩余量。len(dataloader_train) 已减去 start_step，这里加回来保持同基准，
            # 否则续跑时分子分母基准不一致（如 126200/298142，末尾会 >100%）。
            loss_dict["batch_num_epoch"] = len(dataloader_train) + kwargs.get("start_step", 0)

            self.train_loss_avg = (
                self.train_loss_avg * batch_idx + loss_dict["loss"].detach().cpu().item()
            ) / (batch_idx + 1)
            if "acc" in loss_dict["stats"]:
                self.train_acc_avg = (
                    self.train_acc_avg * batch_idx + loss_dict["stats"]["acc"].detach().cpu().item()
                ) / (batch_idx + 1)

            self.log(loss_dict, tag="train")

            if self.step_in_epoch % self.validate_interval == 0:
                self.validate_epoch(
                    model=model,
                    dataloader_val=dataloader_val,
                    epoch=epoch,
                    writer=self.writer,
                    step=batch_idx + 1 + kwargs.get("start_step", 0),
                    step_in_epoch=self.step_in_epoch,
                )

            if self.step_in_epoch % self.save_checkpoint_interval == 0:
                self.save_checkpoint(
                    epoch,
                    model=model,
                    optim=optim,
                    scheduler=scheduler,
                    scaler=scaler,
                    step=batch_idx + 1 + kwargs.get("start_step", 0),
                    step_in_epoch=self.step_in_epoch,
                    data_split_i=kwargs.get("data_split_i", 0),
                    data_split_num=kwargs.get("data_split_num", 1),
                    train_loss_avg=self.train_loss_avg,
                    train_acc_avg=self.train_acc_avg,
                )

            time_beg = time.perf_counter()

        if self.use_ddp or self.use_fsdp or self.use_deepspeed:
            train_loss_avg = torch.tensor(self.train_loss_avg, dtype=torch.float32).to(self.device)
            train_acc_avg = torch.tensor(self.train_acc_avg, dtype=torch.float32).to(self.device)
            dist.all_reduce(train_loss_avg, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_acc_avg, op=dist.ReduceOp.SUM)
            self.train_loss_avg = train_loss_avg.detach().cpu().item() / self.world_size
            self.train_acc_avg = train_acc_avg.detach().cpu().item() / self.world_size

    def _describe_batch(self, batch):
        """把一个 batch 的关键信息拼成可读字符串，供 [SKIP BATCH] 错误日志 dump。

        只做纯 CPU/元信息读取（shape/dtype/长度/token 范围），尽量避免再触发新的 CUDA kernel；
        对 speech/text 的取 min/max 用 try 包住，即使 CUDA 上下文已坏也不会二次抛出把日志搞没。
        """
        if batch is None:
            return "None"
        parts = []
        try:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    info = f"{k}: shape={tuple(v.shape)} dtype={v.dtype} device={v.device}"
                    try:
                        if k in ("speech_lengths", "text_lengths"):
                            info += f" min={int(v.min())} max={int(v.max())}"
                        elif k == "text":
                            info += f" tok_min={int(v.min())} tok_max={int(v.max())}"
                        elif k == "speech":
                            info += (
                                f" val_min={float(v.min()):.4f} val_max={float(v.max()):.4f}"
                                f" has_nan={bool(torch.isnan(v).any())}"
                                f" has_inf={bool(torch.isinf(v).any())}"
                            )
                    except Exception as e:
                        info += f" <stat_failed: {e}>"
                    parts.append(info)
                else:
                    parts.append(f"{k}: {type(v).__name__}={v!r}")
        except Exception as e:
            return f"<describe_batch failed: {e}>"
        return "\n    " + "\n    ".join(parts)

    def forward_step(self, model, batch, loss_dict={}):
        """Forward step.

            Args:
                model: Model instance or model name.
                batch: TODO.
                loss_dict: TODO.
            """
        if self.use_deepspeed and self.dtype in (torch.float16, torch.bfloat16):
            # DeepSpeed 的 bf16/fp16 config 会把权重**永久** cast 成低精度，并且
            # engine.forward 里的 autocast_if_enabled 会主动关掉外层 torch.autocast
            # （见 deepspeed/runtime/torch_autocast.py：外层 enabled 但 engine 未配
            # torch_autocast 时走 `torch.autocast(enabled=False)`）。因此低精度模式下
            # 必须由调用方把浮点输入 cast 成同一 dtype，否则第一个 Linear 就报
            # "mat1 and mat2 must have the same dtype"。整型输入（text / lengths）不能动。
            batch = {
                k: (v.to(self.dtype) if torch.is_tensor(v) and v.is_floating_point() else v)
                for k, v in batch.items()
            }
        with maybe_autocast(dtype=self.dtype, use_deepspeed=self.use_deepspeed):
            retval = model(**batch)

        loss, stats, weight = retval
        stats = {k: v for k, v in stats.items() if v is not None}

        loss_dict["loss"] = loss
        loss_dict["stats"] = stats
        loss_dict["weight"] = weight

    def backward_step(self, model, scaler, loss_dict={}):
        """Backward step.
        
            Args:
                model: Model instance or model name.
                scaler: TODO.
                loss_dict: TODO.
            """
        loss = loss_dict["loss"]

        if self.use_deepspeed:
            scaled_loss = model.backward(loss)
        else:
            loss = loss / self.accum_grad
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

    def update_step(self, model, optim, scheduler, scaler, loss_dict=None):
        """Update step.
        
            Args:
                model: Model instance or model name.
                optim: TODO.
                scheduler: TODO.
                scaler: TODO.
                loss_dict: TODO.
            """
        batch_idx = loss_dict["batch_idx"]
        if self.use_deepspeed:
            model.step()
        else:
            if (batch_idx + 1) % self.accum_grad == 0:
                # Perform gradient clipping if it is set
                if self.grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=self.grad_clip,
                        norm_type=self.grad_clip_type,
                    )
                    if not torch.isfinite(grad_norm):
                        logging.warning(
                            f"The grad norm is {grad_norm}. Skipping updating the model."
                        )
                        optim.zero_grad()  # Reset gradients
                        return

                # Execute an optimization step (update model parameters)
                if self.use_ddp or self.use_fsdp:
                    dist.barrier()
                if scaler:
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()
                scheduler.step()
                # Clear gradients for the next accumulation stage
                optim.zero_grad(set_to_none=True)

    def validate_epoch(
        self,
        model=None,
        dataloader_val=None,
        epoch=None,
        writer=None,
        **kwargs,
    ):
        """
        Defines the validation process for a single epoch.
        Should be implemented with the actual model validation steps.

        Args:
            epoch (int): The current epoch number.
        """
        self.val_loss_avg = 0.0
        self.val_acc_avg  = 0.0

        if self.use_ddp or self.use_fsdp or self.use_deepspeed:
            dist.barrier()
        logging.info(f"Validate epoch: {epoch}, rank: {self.rank}\n")
        model.eval()

        with torch.no_grad():

            speed_stats = {}
            time_beg = time.perf_counter()
            time5 = time_beg

            dataloader_val.batch_sampler.set_epoch(epoch)
            # 与训练循环同样的 epoch 边界同步：验证集也是 IterableDataset，
            # 各 rank batch 数不等时若不同步退出，结尾的 all_reduce(val_loss) 会死锁。
            use_dist = self.use_ddp or self.use_fsdp or self.use_deepspeed
            val_iterator_stop = torch.zeros(1, device=self.device)
            data_iter = iter(dataloader_val)
            batch_idx = -1
            while True:
                try:
                    batch = next(data_iter)
                    local_stop = 0
                except StopIteration:
                    batch = None
                    local_stop = 1
                if use_dist:
                    val_iterator_stop.fill_(local_stop)
                    dist.all_reduce(val_iterator_stop, op=dist.ReduceOp.SUM)
                    if val_iterator_stop.item() > 0:
                        break
                elif local_stop:
                    break

                batch_idx += 1
                loss_dict = {
                    "speed_stats": {},
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "data_split_i": kwargs.get("data_split_i", 0),
                    "data_split_num": kwargs.get("data_split_num", 1),
                    "log_step": batch_idx + kwargs.get("start_step", 0),
                    "batch_total": self.batch_total,
                    "step_in_epoch": batch_idx + 1,
                    "lr": 0.0,
                }

                time1 = time.perf_counter()
                loss_dict["speed_stats"]["data_load"] = f"{time1 - time_beg:0.3f}"

                batch = to_device(batch, self.device, non_blocking=True)

                time2 = time.perf_counter()

                self.forward_step(model, batch, loss_dict=loss_dict)

                time3 = time.perf_counter()
                loss_dict["speed_stats"]["forward_time"] = f"{time3 - time2:0.3f}"

                total_time = f"{(time.perf_counter() - time5):0.3f}"
                time5 = time.perf_counter()

                loss_dict["speed_stats"]["total_time"] = total_time

                loss_dict["batch_num_epoch"] = len(dataloader_val)

                self.log(loss_dict, tag="val")
                time_beg = time.perf_counter()
                self.val_loss_avg = (
                    self.val_loss_avg * batch_idx + loss_dict["loss"].detach().cpu().item()
                ) / (batch_idx + 1)
                if "acc" in loss_dict["stats"]:
                    self.val_acc_avg = (
                        self.val_acc_avg * batch_idx
                        + loss_dict["stats"]["acc"].detach().cpu().item()
                    ) / (batch_idx + 1)

            if self.use_ddp or self.use_fsdp or self.use_deepspeed:
                val_loss_avg = torch.tensor(self.val_loss_avg, dtype=torch.float32).to(self.device)
                val_acc_avg = torch.tensor(self.val_acc_avg, dtype=torch.float32).to(self.device)
                dist.all_reduce(val_loss_avg, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_acc_avg, op=dist.ReduceOp.SUM)
                self.val_loss_avg = val_loss_avg.detach().cpu().item() / self.world_size
                self.val_acc_avg = val_acc_avg.detach().cpu().item() / self.world_size

        # 验证集的聚合指标必须在这里单独打日志 / 写 tensorboard：`log()` 把所有 add_scalar
        # 都锁在 `(batch_idx + 1) % log_interval == 0`，而验证集的 batch 数常远小于
        # log_interval（本项目 dev 8611 行 / 4 卡、样本数被 batch_size_sample_max 卡在 200
        # → 每卡仅约 11 个 batch），于是 `*/val` 一个点都写不出来。即便凑够了，那也是逐 batch
        # 的噪声值、且横轴 batch_total 在验证期间不推进，多个点会挤在同一 x 上互相覆盖。
        # 这里用的是跨 rank all_reduce 后的 epoch/step 级均值，每次验证恰好一个点。
        # `Update best acc/loss` 只在指标变好时才打，不能替代这条无条件的记录。
        logging.info(
            f"Validate done: epoch: {epoch}, total step: {self.batch_total}, "
            f"(val_loss_avg: {self.val_loss_avg:.4f}), (val_acc_avg: {self.val_acc_avg:.4f})"
        )
        if self.writer is not None:
            self.writer.add_scalar("loss/val", self.val_loss_avg, self.batch_total)
            self.writer.add_scalar("acc/val", self.val_acc_avg, self.batch_total)

        if kwargs.get("step_in_epoch", None) is None:
            ckpt_name = f"model.pt.ep{epoch}"
        else:
            ckpt_name = f'model.pt.ep{epoch}.{kwargs.get("step_in_epoch")}'
        self.val_acc_step_or_epoch[ckpt_name] = self.val_acc_avg
        self.val_loss_step_or_epoch[ckpt_name] = self.val_loss_avg

        if self.use_ddp or self.use_fsdp or self.use_deepspeed:
            dist.barrier()

        model.train()

    def log(
        self,
        loss_dict: dict = None,
        tag="train",
        **kwargs,
    ):
        """Log.
        
            Args:
                loss_dict: TODO.
                tag: TODO.
                **kwargs: Additional keyword arguments.
            """
        loss = loss_dict["loss"].detach().cpu().item()
        epoch = loss_dict["epoch"]
        batch_idx = loss_dict["batch_idx"]
        step_in_epoch = loss_dict["step_in_epoch"]
        batch_total = loss_dict["batch_total"]
        batch_num_epoch = loss_dict["batch_num_epoch"]
        lr = loss_dict["lr"]

        speed_stats = loss_dict["speed_stats"]
        stats = loss_dict["stats"]
        data_split_i = loss_dict["data_split_i"]
        data_split_num = loss_dict["data_split_num"]
        log_step = loss_dict.get("log_step", None)

        if (batch_idx + 1) % self.log_interval == 0:
            batch_idx = log_step if log_step is not None else batch_idx
            gpu_info = (
                "GPU, memory: usage: {:.3f} GB, "
                "peak: {:.3f} GB, "
                "cache: {:.3f} GB, "
                "cache_peak: {:.3f} GB".format(
                    torch.cuda.memory_allocated() / 1024 / 1024 / 1024,
                    torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024,
                    torch.cuda.memory_reserved() / 1024 / 1024 / 1024,
                    torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024,
                )
            )

            loss_avg_epoch = getattr(self, f"{tag}_loss_avg")
            acc_avg_epoch = getattr(self, f"{tag}_acc_avg")
            # CTC-only 模型的 loss 是「每条句子的 -logP」（batch 均值，未按 token 归一），
            # 未训练时能到几百，math.exp 会直接 OverflowError 把训练打挂。
            ppl_avg_epoch = math.exp(loss_avg_epoch) if loss_avg_epoch < 700 else float("inf")
            description = (
                f"{tag}, "
                f"rank: {self.rank}, "
                f"epoch: {epoch}/{self.max_epoch}, "
                f"data_slice: {data_split_i}/{data_split_num}, "
                f"step_in_slice: {batch_idx + 1}/{batch_num_epoch}, step_in_epoch: {step_in_epoch}, total step: {batch_total}, "
                f"(loss_batch_local: {loss:.3f}), "
                f"(loss_epoch_local_avg: {loss_avg_epoch:.3f}), "
                f"(ppl_avg_slice: {ppl_avg_epoch:.3e}), "
                f"(acc_avg_slice: {acc_avg_epoch:.3f}), "
                f"(lr: {lr:.3e}), "
                f"{[(k, round(v.detach().cpu().item(), 3)) for k, v in stats.items()]}, "
                f"{speed_stats}, "
                f"{gpu_info}"
            )
            logging.info(description)

            # ``loss/val`` / ``acc/val`` are reserved for the full-dev-set
            # aggregate emitted at the end of ``validate_epoch``.  Logging a
            # sampled validation batch under the same tag and global step made
            # TensorBoard mix two incompatible values at one x coordinate.
            # Keep sampled validation diagnostics, but give them a distinct
            # namespace so consumers can opt in explicitly.
            metric_suffix = "_batch" if tag == "val" else ""
            description_dict = {
                f"loss{metric_suffix}/{tag}": loss,
                f"lr{metric_suffix}/{tag}": lr,
            }

            # ---- 跨 rank 求全局均值 ----
            # writer 只在 rank0 上，但 all_reduce 必须**所有 rank 一起调用**，
            # 因此聚合逻辑必须放在 writer 判空之外，且各 rank 走到这里的条件
            # `(batch_idx + 1) % log_interval == 0` 必须一致（batch_idx 已同步）。
            use_dist = self.use_ddp or self.use_fsdp or self.use_deepspeed
            global_loss = loss
            global_stats = {}
            if use_dist and dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
                buf = torch.tensor([loss], dtype=torch.float32, device=self.device)
                dist.all_reduce(buf, op=dist.ReduceOp.SUM)
                global_loss = (buf.item() / world_size)
                for key, var in stats.items():
                    # 必须 clone：dtype/device 已匹配时 `.to()` 返回**同一块存储**，
                    # 而 all_reduce 是原地写，会把 loss_dict["stats"][key] 直接改成
                    # 各 rank 之和（×world_size）。而 train_epoch:791 /
                    # validate_epoch:1043 随后正是拿这些张量累计 loss_avg / acc_avg，
                    # 于是被 log 到的那些 step 贡献了 4 倍的值 —— val_acc_avg 虚高，
                    # 而它就是 best_model_criterion=[[valid, acc, max]] 选 ckpt 的依据。
                    val = var.detach().reshape(1).to(dtype=torch.float32,
                                                     device=self.device).clone()
                    dist.all_reduce(val, op=dist.ReduceOp.SUM)
                    global_stats[key] = (val.item() / world_size)
            else:
                for key, var in stats.items():
                    global_stats[key] = var.detach().cpu().item()

            writer = self.writer
            if writer is not None:
                # 只在 rank0 写入；写入的是全局均值（tag 不带 rank 前缀）。
                writer.add_scalar(f"loss{metric_suffix}/{tag}", global_loss, batch_total)
                writer.add_scalar(f"lr{metric_suffix}/{tag}", lr, batch_total)
                for key, val in global_stats.items():
                    writer.add_scalar(f"stats_{key}{metric_suffix}/{tag}", val, batch_total)
                    description_dict[f"stats_{key}{metric_suffix}/{tag}"] = val
                for key, var in speed_stats.items():
                    writer.add_scalar(f"stats_{key}{metric_suffix}/{tag}", float(var), batch_total)
                    description_dict[f"stats_{key}{metric_suffix}/{tag}"] = float(var)
            if self.use_wandb and wandb is not None:
                wandb.log(
                    description_dict,
                    step=batch_total,
                )

    def close(self, writer=None):

        """Close.
        
            Args:
                writer: TODO.
            """
        if self.use_ddp or self.use_fsdp:
            dist.barrier()

        if writer is not None:
            writer.close()

        if self.use_ddp or self.use_fsdp:
            torch.distributed.destroy_process_group()

    def warp_model(self, model, **kwargs):

        """Warp model.
        
            Args:
                model: Model instance or model name.
                **kwargs: Additional keyword arguments.
            """
        # warm start 需要枚举 cue/backbone 参数集合，放在 DDP / deepspeed 包装之前，
        # 保证 named_parameters() 返回原始 Parameter 对象。参数 requires_grad 不变，
        # 主干 lr=0 由 _setup_warm_start 构造的 param group 控制。
        self._setup_warm_start_freeze(model, **kwargs)

        if self.use_deepspeed:
            from deepspeed.runtime.zero.stage_1_and_2 import (
                estimate_zero2_model_states_mem_needs_all_live,
            )
            from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
            from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

            local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
            world_size = int(os.environ.get("WORLD_SIZE", 1))

            # NOTE(xcsong): look in detail how the memory estimator API works:
            #   https://deepspeed.readthedocs.io/en/latest/memory.html#discussion
            if int(os.environ.get("RANK", 0)) == 0:
                logging.info("Estimating model states memory needs (zero2)...")
                estimate_zero2_model_states_mem_needs_all_live(
                    model,
                    num_gpus_per_node=local_world_size,
                    num_nodes=world_size // local_world_size,
                )
                logging.info("Estimating model states memory needs (zero3)...")
                estimate_zero3_model_states_mem_needs_all_live(
                    model,
                    num_gpus_per_node=local_world_size,
                    num_nodes=world_size // local_world_size,
                )
            device = None  # Init device later
            pass  # Init DeepSpeed later

        elif self.use_ddp:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            model = model.cuda(local_rank)
            model = DDP(
                model,
                device_ids=[local_rank],
                find_unused_parameters=kwargs.get("train_conf", {}).get(
                    "find_unused_parameters", False
                ),
            )

        else:
            model = model.to(device=kwargs.get("device", "cuda"))

        return model

    def warp_optim_scheduler(self, model, **kwargs):
        """Warp optim scheduler.
        
            Args:
                model: Model instance or model name.
                **kwargs: Additional keyword arguments.
            """
        from funasr.optimizers import optim_classes
        from funasr.schedulers import scheduler_classes
        from omegaconf import OmegaConf, DictConfig
        import json

        # optim
        logging.info("Build optim")
        optim = kwargs.get("optim", "adam")
        assert optim in optim_classes
        optim_class = optim_classes.get(optim)
        params = self._setup_warm_start(model, **kwargs)
        optim = optim_class(params, **kwargs.get("optim_conf"))

        # scheduler
        logging.info("Build scheduler")
        scheduler = kwargs.get("scheduler", "warmuplr")
        assert scheduler in scheduler_classes
        scheduler_class = scheduler_classes.get(scheduler)
        scheduler = scheduler_class(optim, **(kwargs.get("scheduler_conf") or {}))

        if self.use_deepspeed:
            import deepspeed

            args = OmegaConf.create({"deepspeed_config": self.deepspeed_config})
            with open(self.deepspeed_config, "r") as fin:
                ds_configs = json.load(fin)

            if "bf16" in ds_configs and ds_configs["bf16"]["enabled"]:
                self.dtype = torch.bfloat16

            if "fp16" in ds_configs and ds_configs["fp16"]["enabled"]:
                self.dtype = torch.float16
            if "optimizer" in ds_configs:
                # NOTE(xcsong): Disable custom optimizer if it is set in ds_config,
                # extremely useful when enable cpu_offload, DeepspeedCpuAdam
                # could be 4~5x faster than torch native adam
                optim = None
                if "scheduler" in ds_configs:
                    scheduler = None
                else:

                    def make_scheduler(opt):
                        """Scheduler.
                        
                            Args:
                                opt: TODO.
                            """
                        return scheduler_class(opt, **(kwargs.get("scheduler_conf") or {}))

                    scheduler = make_scheduler

            model, optimizer, _, scheduler = deepspeed.initialize(
                args=args,
                model=model,
                optimizer=optim,
                lr_scheduler=scheduler,
                model_parameters=params,
            )

        return model, optim, scheduler

    # ------------------------------------------------------------------ #
    # warm start：先只训练线索分支（如 FiLM），若干 epoch 后再放开 Paraformer 主干
    #
    # 实现方式：主干 param group lr=0，requires_grad 保持 True。
    # 优点：所有参数始终参与 DDP Reducer / ZeRO 分片，不需要重建任何包装。
    # 代价：主干的反向计算照跑（FiLM 在编码器输入端，本来也必须传梯度回来），
    # Adam 对主干维护 momentum 但不做更新（lr=0）。
    # ------------------------------------------------------------------ #
    def _setup_warm_start_freeze(self, model, **kwargs):
        """按 warm_start_conf 将主干 lr 置零（而非冻结 requires_grad）。

        采用 lr=0 方案而非 requires_grad=False：
        - 所有参数始终 requires_grad=True，ZeRO 分片不会出现空桶。
        - DDP Reducer 收集的参数集合不变，解冻时无需重建包装。
        - 代价：backbone 仍参与反向计算（本来 FiLM 在编码器输入端也必须传梯度回去）。

        warm_start_conf:
            epochs: 1                  # 前几个 epoch 只训线索分支（0/缺省 = 关闭）
            cue_prefixes: [fusion]     # 属于线索分支的参数名前缀
        """
        ws = kwargs.get("warm_start_conf", None) or {}
        try:
            epochs = int(ws.get("epochs", 0) or 0)
        except (TypeError, ValueError):
            epochs = 0
        self.warm_start_epochs = epochs
        self.warm_start_done = epochs <= 0
        self._warm_start_cue_params = []
        self._warm_start_backbone_params = []
        if epochs <= 0:
            return
        if self.use_fsdp:
            raise NotImplementedError(
                "warm_start_conf 不支持 FSDP：FSDP 把参数展平成 FlatParameter，"
                "无法按子模块冻结/解冻。"
            )

        prefixes = list(ws.get("cue_prefixes", ["fusion"]) or ["fusion"])
        cue, backbone = [], []
        n_cue = 0
        for name, p in model.named_parameters():
            if any(name == t or name.startswith(t + ".") for t in prefixes):
                cue.append(p)
                n_cue += p.numel()
            else:
                backbone.append(p)
        if not cue:
            sample = [n for n, _ in list(model.named_parameters())[:8]]
            raise ValueError(
                f"warm_start_conf.cue_prefixes={prefixes} 没匹配到任何参数。"
                f"参数名示例: {sample}"
            )

        # 不再冻结 requires_grad —— 改由 optimizer param group lr=0 控制
        self._warm_start_cue_params = cue
        self._warm_start_backbone_params = backbone
        self._warm_start_base_lr = float(
            kwargs.get("optim_conf", {}).get("lr", 1e-3)
        )
        logging.info(
            f"warm start: 前 {epochs} 个 epoch 只训线索分支 {prefixes}"
            f"（{n_cue/1e6:.2f}M 参数），主干 {len(backbone)} 个张量 "
            f"lr=0（requires_grad 保持 True），epoch {epochs} 起恢复 lr"
        )

    def _setup_warm_start(self, model, **kwargs):
        """给 optimizer 的参数集合：warm start 开启时拆成「线索分支 / 主干」两组，
        主干 lr=0；到点后由 apply_warm_start 把主干 lr 恢复为 base_lr。

        注意 group 顺序（cue=0, backbone=1）被 apply_warm_start 依赖：DeepSpeed ZeRO
        会把 param_groups[i]["params"] 替换成扁平化分片，无法再按 Parameter 身份匹配，
        只能按 group 下标定位（DS 保持 group 顺序与数量不变）。
        """
        if getattr(self, "warm_start_epochs", 0) <= 0:
            return model.parameters()
        base_lr = float(kwargs.get("optim_conf", {}).get("lr", 1e-3))
        self._warm_start_base_lr = base_lr
        self._warm_start_backbone_group_idx = 1
        return [
            {"params": self._warm_start_cue_params, "lr": base_lr},
            {"params": self._warm_start_backbone_params, "lr": 0.0},
        ]

    def apply_warm_start(self, model, optim, scheduler, epoch):
        """每个 epoch 开始时调用：到点了就把主干 lr 恢复为 base_lr。返回 model。

        使用 lr=0 方案后 requires_grad 始终为 True，DDP Reducer / ZeRO 分片无需重建；
        只需要改 optimizer 的 backbone param group lr，**以及** scheduler.base_lrs
        —— WarmupLR.get_lr() 每步都用 base_lrs 重算 lr，若不同步更新，下一次
        scheduler.step() 会立刻把主干 lr 打回 0。
        """
        if getattr(self, "warm_start_done", True):
            return model
        if epoch < self.warm_start_epochs:
            logging.info(
                f"warm start: epoch {epoch} 仍只训线索分支（主干 lr=0），"
                f"epoch {self.warm_start_epochs} 起恢复 lr"
            )
            return model

        base_lr = getattr(self, "_warm_start_base_lr", None)
        idx = getattr(self, "_warm_start_backbone_group_idx", None)
        # DeepSpeed 若在 ds_config 里配了 optimizer，则 optim 为 None，真正的
        # optimizer 挂在 engine 上；否则 optim 就是 torch 原生 optimizer（DS 只是
        # 原地改写它的 param_groups）。
        target_optim = optim if optim is not None else getattr(model, "optimizer", None)

        if base_lr is None or idx is None or target_optim is None:
            logging.warning("warm start: 未找到 optimizer / base_lr，主干 lr 未恢复")
            self.warm_start_done = True
            return model

        if idx >= len(target_optim.param_groups):
            logging.warning(
                f"warm start: param_groups 数量({len(target_optim.param_groups)})"
                f"不足，无法定位主干 group {idx}，主干 lr 未恢复"
            )
            self.warm_start_done = True
            return model

        target_optim.param_groups[idx]["lr"] = base_lr
        target_optim.param_groups[idx]["initial_lr"] = base_lr
        sched_base = getattr(scheduler, "base_lrs", None)
        if sched_base is not None and idx < len(sched_base):
            sched_base[idx] = base_lr
        else:
            logging.warning(
                "warm start: scheduler 无 base_lrs，主干 lr 可能被下一次 step 打回 0"
            )
        self.warm_start_done = True
        logging.info(f"warm start: epoch {epoch} 恢复主干 lr={base_lr}（group {idx}）")
        return model
