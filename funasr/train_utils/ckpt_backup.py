"""把本地 ckpt 异步备份到 BOS（本地文件照常保留，云盘只作备份）。

上传走 `bcecmd bos cp`：它自带 multipart + 多线程，实测比用 bostorchconnector
逐 part `put_object` 快得多，也不用自己处理分片和校验。训练进程只把任务丢进后台
队列，`save_checkpoint` 不会被上传阻塞。

队列只有一个 worker，所以多次 save 的上传串行执行，不会几个 GB 的上传互相抢带宽。
上传失败只记 warning，不影响训练——本地那份 ckpt 始终是权威副本。
"""

import atexit
import logging
import os
import queue
import shutil
import subprocess
import threading
import time

DEFAULT_BCECMD = "bcecmd"
_STOP = object()


def normalize_uri(uri: str) -> str:
    """把 `bos://bucket/prefix`、`bucket/prefix` 统一成 bcecmd 要的 `bos:/bucket/prefix`。"""
    rest = uri
    for p in ("bos://", "bos:/"):
        if rest.startswith(p):
            rest = rest[len(p) :]
            break
    rest = rest.strip("/")
    if not rest:
        raise ValueError(f"ckpt 备份 uri 不合法: {uri}")
    return f"bos:/{rest}"


def find_bcecmd(path: str = None) -> str:
    """定位 bcecmd 可执行文件，找不到就抛错（配置写错时早失败）。"""
    for cand in (path, os.environ.get("BCECMD"), DEFAULT_BCECMD, shutil.which("bcecmd")):
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    raise FileNotFoundError(
        f"找不到 bcecmd（试过 {path} / $BCECMD / {DEFAULT_BCECMD} / PATH），"
        "请设置 train_conf.ckpt_bos_backup_bcecmd"
    )


class CkptBackup:
    """ckpt 的后台 BOS 备份队列。只在 rank0 创建。

    Args:
        output_dir: 本地 ckpt 目录。
        uri: 云端前缀，如 `bos://example-bucket/checkpoints/run`；以 `/` 结尾时
            自动拼上 output_dir 的目录名，避免多个实验的同名 ckpt 互相覆盖。
        bcecmd: bcecmd 路径，None 则自动查找。
        concurrency: bcecmd 上传并发度。
        timeout: 单个 ckpt 上传/删除的超时秒数。
        retries: 失败重试次数。
    """

    def __init__(
        self,
        output_dir: str,
        uri: str,
        bcecmd: str = None,
        concurrency: int = 10,
        timeout: int = 3600,
        retries: int = 2,
    ):
        self.output_dir = os.path.abspath(output_dir)
        if uri.endswith("/"):
            uri += os.path.basename(self.output_dir)
        self.uri = normalize_uri(uri)
        self.bcecmd = find_bcecmd(bcecmd)
        self.concurrency = max(1, int(concurrency))
        self.timeout = int(timeout)
        self.retries = max(0, int(retries))
        self._q = queue.Queue()
        self._lock = threading.Lock()
        self._cancelled = set()
        self._worker = threading.Thread(
            target=self._loop, name="ckpt-bos-backup", daemon=True
        )
        self._worker.start()
        atexit.register(self.close)
        logging.info(f"ckpt 将异步备份到 {self.uri}（本地保留，云盘只作备份）")

    # ------------------------------------------------------------ 对外接口
    def upload(self, name: str):
        """把 output_dir/<name>（文件或 deepspeed 的 tag 目录）排进上传队列。"""
        with self._lock:
            self._cancelled.discard(name)
        self._q.put(("up", name))
        n = self._q.qsize()
        logging.info(f"ckpt 备份入队: {name}" + (f"（队列积压 {n}）" if n > 1 else ""))

    def remove(self, name: str):
        """删掉云端的 <name>（本地淘汰 worst ckpt 时调用）。

        同名上传若还排在队里就直接作废：本地文件此刻已被删，传也传不成。
        """
        with self._lock:
            self._cancelled.add(name)
        self._q.put(("rm", name))

    def close(self, wait: bool = True):
        """停止 worker；wait=True 时等队列里剩下的任务传完。"""
        if self._worker is None:
            return
        self._q.put(_STOP)
        if wait and self._worker.is_alive():
            n = self._q.qsize()
            if n > 1:
                logging.info(f"等待 {n - 1} 个 ckpt 备份任务完成...")
            self._worker.join()
        self._worker = None

    # ------------------------------------------------------------ worker
    def _loop(self):
        while True:
            item = self._q.get()
            if item is _STOP:
                return
            action, name = item
            try:
                if action == "up":
                    with self._lock:
                        cancelled = name in self._cancelled
                    if cancelled:
                        logging.info(f"ckpt {name} 已被本地淘汰，跳过备份")
                        continue
                    self._do_upload(name)
                else:
                    self._do_remove(name)
                    with self._lock:
                        self._cancelled.discard(name)
            except Exception as e:  # 备份失败绝不影响训练
                logging.warning(f"ckpt 备份任务 {action} {name} 失败（忽略）: {e}")

    def _run(self, argv, what: str, src: str = None):
        """跑一次 bcecmd（失败重试），返回 (是否成功, 耗时秒)。

        src 给定时每次重试前确认它还在——ckpt 可能刚被 keep_nbest 淘汰掉，
        这种情况下重试没有意义。
        """
        for attempt in range(self.retries + 1):
            if src is not None and not os.path.exists(src):
                logging.info(f"{what}: 本地已不存在（已被淘汰），放弃备份")
                return False, 0.0
            t = time.perf_counter()
            p = subprocess.run(
                argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=self.timeout
            )
            out = p.stdout.decode("utf-8", "replace").strip()
            if p.returncode == 0:
                return True, time.perf_counter() - t
            if src is not None and not os.path.exists(src):
                # 上传途中被 keep_nbest 淘汰了，不是故障
                logging.info(f"{what}: 本地已不存在（已被淘汰），放弃备份")
                return False, 0.0
            logging.warning(
                f"{what} 第 {attempt + 1} 次失败（returncode {p.returncode}）: "
                f"{out[-500:] if out else '无输出'}"
            )
        return False, 0.0

    def _do_upload(self, name: str):
        src = os.path.join(self.output_dir, name)
        # 目录（deepspeed 的 tag 目录）用 -r 整体上传；bcecmd 把 src 目录内的相对路径
        # 挂到 dst 前缀下，所以 dst 必须显式带上 name 这一层。
        dst = f"{self.uri}/{name}"
        argv = [self.bcecmd, "bos", "cp", "--yes", "--quiet", "--disable-bar"]
        if os.path.isdir(src):
            argv += ["--recursive", "--concurrency", str(self.concurrency)]
        argv += [src, dst]
        size = _tree_size(src)
        ok, cost = self._run(argv, f"ckpt 备份 {name} -> {dst}", src=src)
        if ok:
            logging.info(
                f"ckpt 已备份到 {dst}（{size / 2**20:.0f} MiB, {cost:.1f}s, "
                f"{size / 2**20 / max(cost, 1e-6):.0f} MiB/s）"
            )
        elif os.path.exists(src):
            logging.warning(f"ckpt 备份最终失败，本地副本仍在: {src}")

    def _do_remove(self, name: str):
        dst = f"{self.uri}/{name}"
        # ckpt 可能是文件也可能是目录，-r 对两者都适用
        argv = [self.bcecmd, "bos", "rm", "--yes", "--quiet", "--recursive", dst]
        ok, _ = self._run(argv, f"删除云端 ckpt {dst}")
        if ok:
            logging.info(f"已删除云端 ckpt 备份: {dst}")


def _tree_size(path: str) -> int:
    """文件或目录的总字节数（只用于日志里的速率）。"""
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _, files in os.walk(path):
        for n in files:
            try:
                total += os.path.getsize(os.path.join(root, n))
            except OSError:
                pass
    return total
