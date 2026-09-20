"""可插拔的说话人 embedding 编码器。

约定的通用接口 SpeakerEncoder：
    forward(feats, feats_lengths) -> spk_emb   # (B, T, F) -> (B, D)
    output_size() -> int                       # embedding 维度 D

输入 feats 为 dataset 侧提取好的说话人特征（默认 80 维 kaldi-fbank，逐句 mean 归一），
与 ASR 的 LFR/CMVN 特征不同，二者相互独立。

新增编码器：继承 SpeakerEncoder 并用 @tables.register("spk_encoder_classes", "<Name>") 注册即可，
在 config 里通过 spk_encoder / spk_encoder_conf 切换。
"""

import logging
import os
import torch
import torch.nn as nn

from funasr.register import tables


class SpeakerEncoder(nn.Module):
    """说话人 embedding 编码器基类。

    子类需实现 forward(feats, feats_lengths) -> (B, D)，并保证 output_size() 返回 D。
    基类提供可选的预训练权重加载与参数冻结逻辑（供子类在 __init__ 末尾调用）。
    """

    def __init__(self, embedding_size: int = 192, **kwargs):
        super().__init__()
        self._embedding_size = int(embedding_size)

    def output_size(self) -> int:
        """返回说话人 embedding 的维度 D。

        Returns:
            int: embedding 维度。
        """
        return self._embedding_size

    def forward(self, feats: torch.Tensor, feats_lengths: torch.Tensor = None) -> torch.Tensor:
        """由说话人特征提取 embedding（子类实现）。

        Args:
            feats: (B, T, F) 说话人特征。
            feats_lengths: (B,) 每段有效帧数，可空。

        Returns:
            torch.Tensor: (B, D) 说话人 embedding。

        Raises:
            NotImplementedError: 基类未实现，需由子类覆盖。
        """
        raise NotImplementedError

    # -------- 通用工具：加载预训练权重 & 冻结参数 --------
    def load_pretrained(self, init_param: str = None, strict: bool = False):
        """从 ckpt 加载权重。ckpt 可为纯 state_dict 或含 'state_dict'/'model' 的 dict。"""
        if not init_param:
            logging.warning(
                "[SpeakerEncoder] init_param 未设置：说话人编码器为随机初始化。"
                "冻结随机权重通常无意义，请提供预训练 ckpt 或改用非冻结模式。"
            )
            return
        ckpt = torch.load(init_param, map_location="cpu")
        if isinstance(ckpt, dict):
            for k in ("state_dict", "model", "module"):
                if k in ckpt and isinstance(ckpt[k], dict):
                    ckpt = ckpt[k]
                    break
        missing, unexpected = self.load_state_dict(ckpt, strict=strict)
        logging.info(
            f"[SpeakerEncoder] loaded {init_param}; "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

    def freeze(self):
        """冻结全部参数并切到 eval 模式（供冻结的预训练编码器使用）。

        注意：冻结的编码器保持 fp32，deepspeed bf16 下不会下转权重 dtype。
        """
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    # 冻结的说话人编码器保持 fp32：deepspeed bf16 会对整模型 .bfloat16()，但
    # 预训练说话人网络（如 SpeechBrain ECAPA）内部会用 fp32 mask/统计量做乘法，
    # bf16 下会触发 dtype 不一致。这里重写 _apply：冻结时忽略浮点 dtype 转换，
    # 仅跟随 device 变化（浮点张量搬设备但保持 fp32，非浮点缓冲照常应用）。
    def _apply(self, fn, recurse: bool = True):
        if getattr(self, "frozen", False):
            probe = torch.zeros(1, dtype=torch.float32)
            try:
                out = fn(probe)
            except Exception:
                out = probe
            if out.is_floating_point() and out.dtype != probe.dtype:
                # 这是一次浮点 dtype 转换（.half()/.bfloat16()/.to(dtype=...)）。
                # 判断是否同时改变 device：仅当 device 变化时才搬运浮点张量，
                # 且始终保持 fp32（绝不让权重被下转，避免精度损失与 dtype 冲突）。
                device_changed = out.device != probe.device

                def dev_only(t):
                    if t.is_floating_point():
                        return t.to(device=out.device) if device_changed else t
                    return fn(t)

                return super()._apply(dev_only, recurse=recurse)
        return super()._apply(fn, recurse=recurse)


@tables.register("spk_encoder_classes", "ECAPATDNN")
class ECAPATDNNSpeakerEncoder(SpeakerEncoder):
    """ECAPA-TDNN 说话人编码器（复用 funasr.models.sond 内的 ECAPA_TDNN 实现）。

    Args:
        input_size: 输入特征维度（fbank 维度，默认 80）
        embedding_size: 输出 embedding 维度（lin_neurons，默认 192）
        init_param: 预训练 ckpt 路径（可选）
        freeze: 是否冻结参数（配合预训练权重使用）
        ecapa_conf: 透传给底层 ECAPA_TDNN 的其余参数（channels/kernel_sizes/... ）
    """

    def __init__(
        self,
        input_size: int = 80,
        embedding_size: int = 192,
        init_param: str = None,
        freeze: bool = True,
        ecapa_conf: dict = None,
        **kwargs,
    ):
        super().__init__(embedding_size=embedding_size)
        from funasr.models.sond.encoder.ecapa_tdnn_encoder import ECAPA_TDNN

        ecapa_conf = dict(ecapa_conf or {})
        # window_size=None -> 整段做 Attentive Statistics Pooling，输出单条 (B, D) embedding
        ecapa_conf.setdefault("window_size", None)
        self.ecapa = ECAPA_TDNN(
            input_size,
            lin_neurons=embedding_size,
            **ecapa_conf,
        )
        self.input_size = input_size
        self.frozen = bool(freeze)

        self.load_pretrained(init_param)
        if self.frozen:
            self.freeze()

    def train(self, mode: bool = True):
        """切换训练/评估模式；冻结时底层模块始终保持 eval。

        Args:
            mode: 目标模式（默认 True）。

        Returns:
            self。
        """
        # 冻结时始终保持 eval，避免 BN/统计量被训练数据更新
        super().train(mode)
        if self.frozen:
            for m in self.ecapa.modules():
                m.eval()
        return self

    def forward(self, feats: torch.Tensor, feats_lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            feats: (B, T, input_size) 说话人特征
            feats_lengths: (B,) 有效帧数
        Returns:
            spk_emb: (B, embedding_size)
        """
        rel_lengths = None
        if feats_lengths is not None:
            t = feats.shape[1]
            rel_lengths = (feats_lengths.float().view(-1) / max(t, 1)).clamp(0.0, 1.0)

        # 对齐到 ecapa 参数 dtype（deepspeed bf16 下冻结参数可能被转 bf16）
        try:
            p_dtype = next(self.ecapa.parameters()).dtype
            feats = feats.to(p_dtype)
        except StopIteration:
            pass

        ctx = torch.no_grad() if self.frozen else _nullcontext()
        with ctx:
            emb = self.ecapa(feats, lengths=rel_lengths)  # (B, D)
        if emb.dim() == 3:  # 兜底：若底层返回 (B, 1, D)
            emb = emb.squeeze(1)
        return emb


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


@tables.register("spk_encoder_classes", "CAMPPlus")
class CAMPPlusSpeakerEncoder(SpeakerEncoder):
    """CAM++ 说话人编码器（中文场景：3D-Speaker / FunASR 的 speech_campplus_sv_zh-cn_16k-common）。

    相对 VoxCeleb 上训的 ECAPA-TDNN，这个模型用中文数据训练，对中文与短音频更匹配。

    Args:
        init_param: `campplus_cn_common.bin`（纯 state_dict）路径
        embedding_size: 192
        feat_dim: 输入 fbank 维度（80）
        freeze: 是否冻结（预训练 + 冻结）
        campplus_conf: 透传给底层 CAMPPlus 的其余参数

    输入约定：**80 维 kaldi-fbank + 逐句 mean 归一**，与 FunASR 官方
    `funasr/models/campplus/utils.py:extract_feature` 完全一致
    （`Kaldi.fbank(wav.unsqueeze(0), num_mel_bins=80)` 再减逐句均值）。
    因此 dataset 侧 ref_feature 需设为 kaldi_fbank。

    注意 `CAMPPlus.forward` 只接受 (B, T, 80)、**不接受长度**，末端是 statistics pooling，
    补零帧会污染统计量。本类在 batch 内长度不一致时逐条前向（模型仅 27 MB，代价可忽略），
    保证与单条推理一致。
    """

    def __init__(
        self,
        init_param: str = None,
        embedding_size: int = 192,
        feat_dim: int = 80,
        freeze: bool = True,
        campplus_conf: dict = None,
        **kwargs,
    ):
        super().__init__(embedding_size=embedding_size)
        from funasr.models.campplus.model import CAMPPlus

        conf = dict(campplus_conf or {})
        conf.setdefault("growth_rate", 32)
        conf.setdefault("bn_size", 4)
        conf.setdefault("init_channels", 128)
        conf.setdefault("config_str", "batchnorm-relu")
        conf.setdefault("memory_efficient", True)
        conf.setdefault("output_level", "segment")
        self.campplus = CAMPPlus(feat_dim=feat_dim, embedding_size=embedding_size, **conf)
        self.feat_dim = int(feat_dim)
        self.frozen = bool(freeze)

        self._load_campplus(init_param)
        if self.frozen:
            self.freeze()

    def _load_campplus(self, init_param: str):
        """把 campplus_cn_common.bin 载入 self.campplus。

        ckpt 的 key 是 `head.* / xvector.*`（没有前缀），而本类把模型包在 `self.campplus`
        下，所以**不能**用基类的 load_pretrained（它对 self 加载、strict=False 时会一个权重
        都不载入却静默通过 —— 实测那样会得到坍缩的 embedding：同/异说话人余弦都是 0.98、
        EER 73%）。这里直接对 self.campplus 用 strict=True，并显式校验。
        """
        if not init_param:
            raise ValueError(
                "[CAMPPlus] 必须提供 init_param（campplus_cn_common.bin）；"
                "随机初始化的冻结说话人编码器没有意义。"
            )
        ckpt = torch.load(init_param, map_location="cpu")
        if isinstance(ckpt, dict):
            for k in ("state_dict", "model", "module"):
                if k in ckpt and isinstance(ckpt[k], dict):
                    ckpt = ckpt[k]
                    break
        ckpt = {k[len("campplus.") :] if k.startswith("campplus.") else k: v
                for k, v in ckpt.items()}
        self.campplus.load_state_dict(ckpt, strict=True)
        logging.info(f"[CAMPPlus] 已加载 {len(ckpt)} 个权重: {init_param}")

    def train(self, mode: bool = True):
        """切换训练/评估模式；冻结时底层模块始终保持 eval。

        Args:
            mode: 目标模式（默认 True）。

        Returns:
            self。
        """
        # 冻结时始终保持 eval，避免 BN 统计量被训练数据更新
        super().train(mode)
        if self.frozen:
            for m in self.campplus.modules():
                m.eval()
        return self

    def forward(self, feats: torch.Tensor, feats_lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            feats: (B, T, feat_dim) 80 维 kaldi-fbank（已逐句 mean 归一，右侧可补零）
            feats_lengths: (B,) 有效帧数；长度不一致时逐条前向，避开补零帧
        Returns:
            spk_emb: (B, embedding_size)
        """
        try:
            p_dtype = next(self.campplus.parameters()).dtype
            feats = feats.to(p_dtype)
        except StopIteration:
            pass

        ctx = torch.no_grad() if self.frozen else _nullcontext()
        with ctx:
            if feats_lengths is None:
                emb = self.campplus(feats)
            else:
                lens = [max(int(n), 1) for n in feats_lengths.view(-1).tolist()]
                if len(set(lens)) == 1:
                    emb = self.campplus(feats[:, : lens[0]])
                else:
                    emb = torch.cat(
                        [self.campplus(feats[i : i + 1, :n]) for i, n in enumerate(lens)],
                        dim=0,
                    )
        if emb.dim() == 3:  # 兜底：若底层返回 (B, 1, D)
            emb = emb.squeeze(1)
        return emb


@tables.register("spk_encoder_classes", "ECAPASpeechBrain")
class ECAPASpeechBrainSpeakerEncoder(SpeakerEncoder):
    """SpeechBrain 官方 ECAPA-TDNN 说话人编码器（如 spkrec-ecapa-voxceleb）。

    直接复用 SpeechBrain 的特征管线（Fbank + 逐句 CMN + embedding 归一）与预训练权重，
    保证特征与权重完全一致（最忠实）。输入为原始波形，特征在编码器内部计算。

    Args:
        source: SpeechBrain 模型目录（本地路径或 HF id），含 hyperparams.yaml / *.ckpt
        savedir: from_hparams 的缓存目录（默认 source）
        embedding_size: 输出维度（voxceleb ECAPA=192）
        freeze: 是否冻结（预训练+冻结）
        use_mean_var_norm_emb: 是否对输出 embedding 应用 mean_var_norm_emb（与官方推理一致，默认 True）

    依赖：需安装 speechbrain（pip install speechbrain）。
    输入约定：dataset 侧 ref_feature: waveform，传入原始 16k 波形（[-1,1]）。
    """

    def __init__(
        self,
        source: str,
        savedir: str = None,
        embedding_size: int = 192,
        freeze: bool = True,
        use_mean_var_norm_emb: bool = True,
        **kwargs,
    ):
        super().__init__(embedding_size=embedding_size)
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except Exception:
            try:
                # 兼容旧版本 speechbrain
                from speechbrain.pretrained import EncoderClassifier
            except Exception as e:
                raise ImportError(
                    "ECAPASpeechBrain 需要 speechbrain：pip install speechbrain。"
                    f"（import 失败：{e}）"
                )

        run_opts = {"device": "cpu"}  # 先在 CPU 构建，随后随父模型 .to(device)
        # hyperparams.yaml 里 pretrainer.paths 用 pretrained_path(默认 HF repo id)拼 ckpt 路径，
        # 直接用会去 HuggingFace 下载。source 为本地目录时，覆盖 pretrained_path 指向本地，纯离线加载。
        overrides = {}
        if os.path.isdir(source):
            overrides["pretrained_path"] = source
        sb = EncoderClassifier.from_hparams(
            source=source,
            savedir=savedir or source,
            run_opts=run_opts,
            overrides=overrides,
        )
        # 把 SpeechBrain 的模块挂成本模块的子模块，使 .to(device)/state_dict 正常传播
        self.mods = sb.mods  # ModuleDict: compute_features, mean_var_norm, embedding_model, ...
        self.use_mean_var_norm_emb = bool(use_mean_var_norm_emb) and hasattr(
            self.mods, "mean_var_norm_emb"
        )
        self.frozen = bool(freeze)
        if self.frozen:
            self.freeze()

    def train(self, mode: bool = True):
        """切换训练/评估模式；冻结时底层模块始终保持 eval。

        Args:
            mode: 目标模式（默认 True）。

        Returns:
            self。
        """
        super().train(mode)
        if self.frozen:
            for m in self.mods.modules():
                m.eval()
        return self

    def forward(self, wavs: torch.Tensor, wav_lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            wavs: (B, T_samples) 原始波形（已 padding）
            wav_lengths: (B,) 每条有效采样点数
        Returns:
            spk_emb: (B, embedding_size)
        """
        if wavs.dim() == 3:  # (B, T, 1) -> (B, T)
            wavs = wavs.squeeze(-1)
        # 特征提取（含 STFT）在 fp32 下做，避免 bf16/fp16 波形 FFT 报错；
        # 随后把特征对齐到 embedding_model 的参数 dtype（deepspeed 可能把冻结参数转 bf16）。
        wavs = wavs.to(torch.float32)
        # SpeechBrain 约定 wav_lens 为相对长度（0~1）
        if wav_lengths is not None:
            t = wavs.shape[1]
            wav_lens = (wav_lengths.float().view(-1) / max(t, 1)).clamp(0.0, 1.0)
        else:
            wav_lens = torch.ones(wavs.shape[0], device=wavs.device)

        try:
            emb_dtype = next(self.mods.embedding_model.parameters()).dtype
        except StopIteration:
            emb_dtype = torch.float32

        ctx = torch.no_grad() if self.frozen else _nullcontext()
        with ctx:
            # 复现 EncoderClassifier.encode_batch 的步骤（用已注册的 mods，设备可控）
            # SpeechBrain 的 InputNormalization 把 glob_mean/glob_std 存成普通属性而非
            # registered buffer，.to(device) 不会搬运，需在前向里手动对齐设备/精度。
            self._sync_norm_stats(self.mods.mean_var_norm, wavs.device, torch.float32)
            feats = self.mods.compute_features(wavs)
            feats = self.mods.mean_var_norm(feats, wav_lens)
            emb = self.mods.embedding_model(feats.to(emb_dtype), wav_lens)  # (B, 1, D)
            if self.use_mean_var_norm_emb:
                self._sync_norm_stats(self.mods.mean_var_norm_emb, emb.device, emb.dtype)
                emb = self.mods.mean_var_norm_emb(
                    emb, torch.ones(emb.shape[0], device=emb.device)
                )
        if emb.dim() == 3:
            emb = emb.squeeze(1)
        return emb

    @staticmethod
    def _sync_norm_stats(norm_module, device, dtype):
        """把 SpeechBrain InputNormalization 的运行统计量搬到目标 device/dtype。"""
        for name in ("glob_mean", "glob_std"):
            val = getattr(norm_module, name, None)
            if torch.is_tensor(val):
                setattr(norm_module, name, val.to(device=device, dtype=dtype))
