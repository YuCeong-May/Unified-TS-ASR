"""Target-Speaker Paraformer with pluggable speaker-embedding injection.

在 Paraformer 基础上，于「fbank 特征进入编码器之前」注入目标说话人 embedding：

    speech(fbank) --specaug/normalize--> feats
                                          |
        ref_speech --spk_encoder--> spk_emb
                                          |
                         feats = fusion(feats, feats_lengths, spk_emb)
                                          |
                                       encoder --> ...(与父类完全相同)

设计要点：
    - 完全通过子类实现，不修改 Paraformer / ParaformerStreaming，现有训练零影响。
    - spk_encoder 与 fusion 均通过注册表按名字构建，可替换。
    - 若 config 未同时提供 spk_encoder 与 fusion，则退化为普通父类行为（不注入）。
    - 当前按用户需求「先不做流式」：默认基线 ParaformerSpkTS 基于离线 Paraformer；
      ParaformerStreamingSpkTS 保留作为后续流式扩展（训练前向共用同一注入逻辑）。
"""

import logging
import torch
from torch.cuda.amp import autocast

from funasr.register import tables
from funasr.models.paraformer.model import Paraformer
from funasr.models.paraformer_streaming.model import ParaformerStreaming


class _SpkTSInjectMixin:
    """说话人 embedding 注入逻辑（离线/流式基类共用）。

    通过多继承混入到具体的 Paraformer 子类，覆盖 forward/encode，
    在编码器之前把说话人 embedding 融合进声学特征。
    """

    def _build_spk_ts(self, **kwargs):
        """在子类 __init__ 里、super().__init__ 之后调用，构建 spk_encoder / fusion。

        两种用法：
        1) spk_encoder + fusion：dataset 给 `ref_speech`（波形/特征），模型内部现算 embedding。
        2) 只给 fusion + spk_emb_dim：dataset 直接给**离线预提好的** `spk_emb`（推荐）。
           spk_encoder 本来就是冻结的，离线提一遍存成 memmap 后训练不必重复前向，
           也免掉了 deepspeed bf16 下对说话人网络的 dtype 保护。
        """
        spk_encoder_name = kwargs.get("spk_encoder", None)
        spk_encoder_conf = dict(kwargs.get("spk_encoder_conf", {}) or {})
        fusion_name = kwargs.get("fusion", None)
        fusion_conf = dict(kwargs.get("fusion_conf", {}) or {})
        input_size = kwargs.get("input_size", 80)
        spk_emb_dim = kwargs.get("spk_emb_dim", None)

        self.spk_encoder = None
        self.fusion = None
        # forward -> encode 之间传递参考语音/预提 embedding 的临时缓存（每个 forward 覆盖一次）
        self._ref_speech = None
        self._ref_speech_lengths = None
        self._spk_emb = None
        self._warned_no_ref = False

        cls_name = type(self).__name__
        if fusion_name is None:
            logging.warning(
                f"[{cls_name}] 未配置 fusion，退化为普通父类行为（不注入说话人 embedding）。"
            )
            return

        if spk_encoder_name is not None:
            spk_encoder_cls = tables.spk_encoder_classes.get(spk_encoder_name)
            if spk_encoder_cls is None:
                raise ValueError(
                    f"Unknown spk_encoder '{spk_encoder_name}'. "
                    f"registered: {list(tables.spk_encoder_classes.keys())}"
                )
            self.spk_encoder = spk_encoder_cls(**spk_encoder_conf)
            spk_dim = self.spk_encoder.output_size()
        elif spk_emb_dim is not None:
            spk_dim = int(spk_emb_dim)
        else:
            raise ValueError(
                f"[{cls_name}] 配置了 fusion 但既没有 spk_encoder 也没有 spk_emb_dim；"
                f"用离线预提 embedding 时请设置 spk_emb_dim（如 192）。"
            )

        fusion_cls = tables.fusion_classes.get(fusion_name)
        if fusion_cls is None:
            raise ValueError(
                f"Unknown fusion '{fusion_name}'. "
                f"registered: {list(tables.fusion_classes.keys())}"
            )
        self.fusion = fusion_cls(feats_dim=input_size, spk_dim=spk_dim, **fusion_conf)
        logging.info(
            f"[{cls_name}] spk_encoder={spk_encoder_name or '(离线预提 spk_emb)'}"
            f"(dim={spk_dim}), fusion={fusion_name}, input_size={input_size}"
        )

    # ------------------------------------------------------------------ #
    def forward(self, speech, speech_lengths, text, text_lengths, **kwargs):
        """前向入口：先缓存参考语音/预提 embedding，再调用父类前向。

        Args:
            speech: (B, T) 输入音频或 (B, T, F) 特征。
            speech_lengths: (B,) 每段有效长度。
            text: (B, L) 文本 token id。
            text_lengths: (B,) 每句有效长度。
            **kwargs: 可含 ref_speech / ref_speech_lengths / spk_emb，供 encode 读取。

        Returns:
            与父类 forward 相同的输出（logits / sample_ids 等）。
        """
        # 缓存参考语音 / 预提 embedding，供被 super().forward 调用的 self.encode 读取
        self._ref_speech = kwargs.get("ref_speech", None)
        self._ref_speech_lengths = kwargs.get("ref_speech_lengths", None)
        self._spk_emb = kwargs.get("spk_emb", None)
        try:
            return super().forward(speech, speech_lengths, text, text_lengths, **kwargs)
        finally:
            self._ref_speech = None
            self._ref_speech_lengths = None
            self._spk_emb = None

    # ------------------------------------------------------------------ #
    def inference(self, *args, **kwargs):
        """推理入口：父类 inference 只调 self.encode(speech, lengths)、不透传 kwargs，
        所以这里先把参考音频缓存到实例上（与 forward 同一条通路）。

        kwargs["ref_speech"] 接受：
            - list/tuple of 1D np.ndarray / torch.Tensor（每条样本各自的注册音频，
              长度须等于 batch；内部按最长补零并给出真实长度）
            - torch.Tensor (T,) / (B, T)
        kwargs["spk_emb"] 接受 (B, D) / (D,) 的离线预提 embedding（此时不需要 ref_speech）。
        两者都不传则退化为无注入前向（等价于普通 Paraformer）。

        注意：AutoModel 会把 kwargs 整批透传给每个 batch，所以调用方需要**按 batch**
        调用 generate，让 ref_speech / spk_emb 与该 batch 一一对应。
        """
        ref_speech = kwargs.pop("ref_speech", None)
        ref_speech_lengths = kwargs.pop("ref_speech_lengths", None)
        spk_emb = kwargs.pop("spk_emb", None)
        if ref_speech is not None:
            ref_speech, ref_speech_lengths = self._pad_ref_speech(
                ref_speech, ref_speech_lengths
            )
        if spk_emb is not None and not isinstance(spk_emb, torch.Tensor):
            spk_emb = torch.as_tensor(spk_emb, dtype=torch.float32)
        if isinstance(spk_emb, torch.Tensor) and spk_emb.dim() == 1:
            spk_emb = spk_emb.unsqueeze(0)
        self._ref_speech = ref_speech
        self._ref_speech_lengths = ref_speech_lengths
        self._spk_emb = spk_emb
        try:
            return super().inference(*args, **kwargs)
        finally:
            self._ref_speech = None
            self._ref_speech_lengths = None
            self._spk_emb = None

    @staticmethod
    def _pad_ref_speech(ref_speech, ref_speech_lengths=None):
        """把变长参考音频整理成 (B, T) 波形张量 + (B,) 真实长度。"""
        if isinstance(ref_speech, torch.Tensor):
            if ref_speech.dim() == 1:
                ref_speech = ref_speech.unsqueeze(0)
            if ref_speech_lengths is None:
                ref_speech_lengths = torch.full(
                    (ref_speech.size(0),), ref_speech.size(1), dtype=torch.int32
                )
        else:
            waves = []
            for w in ref_speech:
                if not isinstance(w, torch.Tensor):
                    w = torch.as_tensor(w)
                waves.append(w.reshape(-1).float())
            lens = [w.numel() for w in waves]
            max_len = max(lens) if lens else 1
            padded = torch.zeros(len(waves), max(max_len, 1), dtype=torch.float32)
            for i, w in enumerate(waves):
                if w.numel() > 0:
                    padded[i, : w.numel()] = w
            ref_speech = padded
            if ref_speech_lengths is None:
                ref_speech_lengths = torch.tensor(lens, dtype=torch.int32)
        if not isinstance(ref_speech_lengths, torch.Tensor):
            ref_speech_lengths = torch.tensor(ref_speech_lengths, dtype=torch.int32)
        return ref_speech, ref_speech_lengths.view(-1)

    # ------------------------------------------------------------------ #
    def _inject_speaker(self, feats, feats_lengths):
        """在编码器之前，把说话人 embedding 融合进语音特征。

        优先用离线预提的 `spk_emb`；没有时才退回「ref_speech -> spk_encoder」现算。
        两者都没有则跳过注入（一次性告警），行为等价于普通 Paraformer。
        """
        if self.fusion is None:
            return feats

        spk_emb = self._spk_emb
        if spk_emb is not None:
            spk_emb = spk_emb.to(device=feats.device)
            if spk_emb.dim() == 1:
                spk_emb = spk_emb.unsqueeze(0)
        elif self.spk_encoder is not None and self._ref_speech is not None:
            # 只搬设备、不强制 dtype：说话人编码器（尤其含 FFT 特征提取的 SpeechBrain）
            # 内部自行选择计算精度，避免在 bf16/fp16 下对波形做 FFT 出错。
            ref_speech = self._ref_speech.to(device=feats.device)
            ref_speech_lengths = self._ref_speech_lengths
            if ref_speech_lengths is not None:
                ref_speech_lengths = ref_speech_lengths.to(feats.device).view(-1)
            spk_emb = self.spk_encoder(ref_speech, ref_speech_lengths)  # (B, D)
        else:
            if not self._warned_no_ref:
                logging.warning(
                    f"[{type(self).__name__}] 未提供 spk_emb / ref_speech，"
                    f"跳过说话人 embedding 注入。"
                )
                self._warned_no_ref = True
            return feats

        # 推理时若整批共用一条 embedding（B=1），按声学 batch 展开
        if spk_emb.size(0) == 1 and feats.size(0) > 1:
            spk_emb = spk_emb.expand(feats.size(0), -1)

        spk_emb = spk_emb.to(feats.dtype)  # 融合前对齐到 ASR 特征 dtype
        return self.fusion(feats, feats_lengths, spk_emb)

    # ------------------------------------------------------------------ #
    def encode(self, speech, speech_lengths, **kwargs):
        """与 Paraformer.encode 一致，但在编码器前插入说话人 embedding 融合。"""
        with autocast(False):
            if self.specaug is not None and self.training:
                speech, speech_lengths = self.specaug(speech, speech_lengths)
            if self.normalize is not None:
                speech, speech_lengths = self.normalize(speech, speech_lengths)

        # === 说话人 embedding 注入（编码器之前）===
        speech = self._inject_speaker(speech, speech_lengths)

        # Forward encoder（与父类保持一致：不额外传 ind，行为不变）
        encoder_out, encoder_out_lens, _ = self.encoder(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        return encoder_out, encoder_out_lens


@tables.register("model_classes", "ParaformerSpkTS")
class ParaformerSpkTS(_SpkTSInjectMixin, Paraformer):
    """带说话人 embedding 注入的离线 Paraformer（target-speaker ASR，默认基线）。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_spk_ts(**kwargs)


@tables.register("model_classes", "ParaformerStreamingSpkTS")
class ParaformerStreamingSpkTS(_SpkTSInjectMixin, ParaformerStreaming):
    """带说话人 embedding 注入的流式 Paraformer（后续流式扩展用；训练前向与离线一致）。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_spk_ts(**kwargs)


# =========================================================================== #
# 实验1：唤醒词 phoneme 文本线索注入（text-cued TS-ASR）
#   与 speaker-embedding 路径独立：线索是「音素序列」而非单向量，交互复用
#   DAE-TSE 的 Phoneme Encoder + cross-attention（见 phoneme_encoder.py /
#   phoneme_fusion.py）。无 ref_phoneme 的样本自动退化为普通 encoder-decoder。
# =========================================================================== #
class _PhoneTSInjectMixin:
    """唤醒词音素线索注入逻辑（在 encoder 之前把 phoneme 融合进声学特征）。"""

    def _build_phone_ts(self, **kwargs):
        phoneme_encoder_name = kwargs.get("phoneme_encoder", None)
        phoneme_encoder_conf = dict(kwargs.get("phoneme_encoder_conf", {}) or {})
        fusion_name = kwargs.get("fusion", None)
        fusion_conf = dict(kwargs.get("fusion_conf", {}) or {})
        input_size = kwargs.get("input_size", 560)

        self.phoneme_encoder = None
        self.phone_fusion = None
        self._ref_phoneme = None
        self._ref_phoneme_lengths = None
        self._warned_no_phoneme = False

        cls_name = type(self).__name__
        if phoneme_encoder_name is not None and fusion_name is not None:
            enc_cls = tables.phoneme_encoder_classes.get(phoneme_encoder_name)
            if enc_cls is None:
                raise ValueError(
                    f"Unknown phoneme_encoder '{phoneme_encoder_name}'. "
                    f"registered: {list(tables.phoneme_encoder_classes.keys())}"
                )
            self.phoneme_encoder = enc_cls(**phoneme_encoder_conf)
            d_model = self.phoneme_encoder.output_size()

            fusion_cls = tables.fusion_classes.get(fusion_name)
            if fusion_cls is None:
                raise ValueError(
                    f"Unknown fusion '{fusion_name}'. "
                    f"registered: {list(tables.fusion_classes.keys())}"
                )
            # d_model 由 phoneme encoder 决定，若 config 未显式给则补齐
            fusion_conf.setdefault("feats_dim", input_size)
            fusion_conf.setdefault("d_model", d_model)
            self.phone_fusion = fusion_cls(**fusion_conf)
            logging.info(
                f"[{cls_name}] phoneme_encoder={phoneme_encoder_name}(dim={d_model}), "
                f"fusion={fusion_name}, input_size={input_size}"
            )
        else:
            logging.warning(
                f"[{cls_name}] phoneme_encoder/fusion 未同时配置，"
                f"退化为普通父类行为（不注入音素线索）。"
            )

    # ------------------------------------------------------------------ #
    def forward(self, speech, speech_lengths, text, text_lengths, **kwargs):
        """前向入口：先缓存唤醒词音素线索，再调用父类前向。

        Args:
            speech: (B, T) 输入音频或 (B, T, F) 特征。
            speech_lengths: (B,) 每段有效长度。
            text: (B, L) 文本 token id。
            text_lengths: (B,) 每句有效长度。
            **kwargs: 可含 ref_phoneme / ref_phoneme_lengths，供编码器读取。

        Returns:
            与父类 forward 相同的输出（logits / sample_ids 等）。
        """
        self._ref_phoneme = kwargs.get("ref_phoneme", None)
        self._ref_phoneme_lengths = kwargs.get("ref_phoneme_lengths", None)
        try:
            return super().forward(speech, speech_lengths, text, text_lengths, **kwargs)
        finally:
            self._ref_phoneme = None
            self._ref_phoneme_lengths = None

    # ------------------------------------------------------------------ #
    def inference(self, *args, **kwargs):
        """推理入口：父类 inference 只调 self.encode(speech, lengths)，不透传 kwargs，
        所以这里先把唤醒词音素线索缓存到实例上（与 forward 同一条通路）。

        kwargs["ref_phoneme"] 接受：
            - list[int]           单条线索（整个 batch 共用，在 _inject_phoneme 里展开）
            - list[list[int]]     每条样本各自的线索（长度须等于 batch）
            - torch.Tensor (L,) / (B, L)
        不传则退化为无线索前向（等价于普通 Paraformer）。
        """
        ref_phoneme = kwargs.pop("ref_phoneme", None)
        ref_phoneme_lengths = kwargs.pop("ref_phoneme_lengths", None)
        if ref_phoneme is not None:
            if not isinstance(ref_phoneme, torch.Tensor):
                if len(ref_phoneme) > 0 and isinstance(ref_phoneme[0], (list, tuple)):
                    max_len = max(len(x) for x in ref_phoneme)
                    lens = [len(x) for x in ref_phoneme]
                    ref_phoneme = [list(x) + [0] * (max_len - len(x)) for x in ref_phoneme]
                else:
                    lens = [len(ref_phoneme)]
                    ref_phoneme = [list(ref_phoneme)]
                ref_phoneme = torch.tensor(ref_phoneme, dtype=torch.int64)
                if ref_phoneme_lengths is None:
                    ref_phoneme_lengths = torch.tensor(lens, dtype=torch.int32)
            elif ref_phoneme.dim() == 1:
                ref_phoneme = ref_phoneme.unsqueeze(0)
            if ref_phoneme_lengths is None:
                ref_phoneme_lengths = torch.full(
                    (ref_phoneme.size(0),), ref_phoneme.size(1), dtype=torch.int32
                )
            elif not isinstance(ref_phoneme_lengths, torch.Tensor):
                ref_phoneme_lengths = torch.tensor(ref_phoneme_lengths, dtype=torch.int32)
        self._ref_phoneme = ref_phoneme
        self._ref_phoneme_lengths = ref_phoneme_lengths
        try:
            return super().inference(*args, **kwargs)
        finally:
            self._ref_phoneme = None
            self._ref_phoneme_lengths = None

    # ------------------------------------------------------------------ #
    def _inject_phoneme(self, feats, feats_lengths):
        if self.phoneme_encoder is None or self.phone_fusion is None:
            return feats
        ref_phoneme = self._ref_phoneme
        ref_phoneme_lengths = self._ref_phoneme_lengths
        if ref_phoneme is None or ref_phoneme_lengths is None:
            # 无唤醒词线索（传统 ASR 样本 / 推理）→ 跳过，正常前向
            if not self._warned_no_phoneme:
                logging.warning(
                    f"[{type(self).__name__}] 未提供 ref_phoneme，跳过音素线索注入。"
                )
                self._warned_no_phoneme = True
            return feats

        ref_phoneme = ref_phoneme.to(device=feats.device)
        ref_phoneme_lengths = ref_phoneme_lengths.to(feats.device).view(-1)
        # 推理时整个 batch 共用一条线索（B=1），按声学 batch 展开
        if ref_phoneme.size(0) == 1 and feats.size(0) > 1:
            ref_phoneme = ref_phoneme.expand(feats.size(0), -1)
            ref_phoneme_lengths = ref_phoneme_lengths.expand(feats.size(0))

        kw_feat, kw_valid = self.phoneme_encoder(ref_phoneme, ref_phoneme_lengths)
        kw_feat = kw_feat.to(feats.dtype)  # 对齐到声学特征 dtype
        feats = self.phone_fusion(feats, feats_lengths, kw_feat, kw_valid)
        return feats

    # ------------------------------------------------------------------ #
    def encode(self, speech, speech_lengths, **kwargs):
        """与 Paraformer.encode 一致，但在编码器前插入音素线索融合。"""
        with autocast(False):
            if self.specaug is not None and self.training:
                speech, speech_lengths = self.specaug(speech, speech_lengths)
            if self.normalize is not None:
                speech, speech_lengths = self.normalize(speech, speech_lengths)

        # === 唤醒词音素注入（编码器之前）===
        speech = self._inject_phoneme(speech, speech_lengths)

        encoder_out, encoder_out_lens, _ = self.encoder(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        return encoder_out, encoder_out_lens


@tables.register("model_classes", "ParaformerPhoneTS")
class ParaformerPhoneTS(_PhoneTSInjectMixin, Paraformer):
    """带唤醒词音素线索注入的离线 Paraformer（实验1，text-cued TS-ASR）。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_phone_ts(**kwargs)
