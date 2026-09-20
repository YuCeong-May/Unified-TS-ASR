# -*- coding: utf-8 -*-
"""实验4（V4）：DAE-TSE / KCE `AEDKWSASRPhone` 的忠实复刻，**从零训练**。

模型三块：
    1. `cue_encoder`  `KeywordTransformerEncoderDae`，关键词 **中文 char** 序列 ->
       上下文特征（4 层，256 维）。
    2. `encoder`      `ConformerKceEncoderDae`，Conv2d x2 前端 + 9 层 Conformer，
       **每一层**都 cross-attend 关键词特征。
    3. `ctc`          char 词表上的 CTC 头。

相对 DAE-TSE 的取舍（详见 `dae_kws_encoders.py` 与配置注释）：
- 音素词表（74）-> ASR char 词表（8404），这是唯一的实质改动。
- 说话人验证辅助 CE 头默认**关闭**（`spk_ce_conf: null`，此时 ckpt 里不多出任何张量、
  loss 与纯 CTC 逐位相同）。最初是整个去掉的，理由是「本项目全树无说话人标签」；
  aishell1 + aishell2 的 jsonl 带 `speaker` 字段，那个前提不再成立，于是把它加回来成
  一个开关：填了 `spk_ce_conf` 就在倒数第二层的池化表征上问「你锁定的是谁」。
  实现与设计理由见 `spk_ce_head.py`，数据侧的 `spk_id` 见 `datasets_kws_span_tsasr.py`。
- CTC 用 FunASR 的 `funasr.models.ctc.ctc.CTC`（数学等价：`log_softmax(2)` ->
  `nn.CTCLoss(blank=0)` -> `sum / B`）。

「定位」监督来自数据侧：关键词是转写里的**随机连续片段**（不再恒为前缀），且 CTC 目标
必须在该片段原位吐出 `psok … peok` 括号对（`with_trans`）。模型不知道括号该打在哪，
除非真的用 cross-attention 定位了线索——这替代了负样本的作用。送进 cue encoder 的线索
两端**也**加这对括号（`inject_special_token` 的行为），它们在 cross-attention 的线索维度
上充当 sink，吸收「这一帧不匹配任何线索字」的注意力质量。
`psok` / `peok` 是**专用** token（`conf/tokens_kws8408.json` 的 `<psok>` / `<peok>`
= 8404 / 8405，`vocab_size` 8408）。早期版本复用 `<s>` / `</s>`(1/2)，但那两个是正常
转写的句首句尾符，语义混在一起。

拒识也用**专用** token（2026-08-22 起，此前复用 `<unk>`(8403)）：`<no_kws/>`(8406)
= 线索没出现在这条音频里，`<no_tgt_spk/>`(8407) = 注册说话人不在混合语音里。分两个 id
而不是一个，是为了让漏检能按原因归因；不复用 `<unk>` 是因为 `<unk>` 已经表示「转写里
有词表外的字」，两种含义压在一行 CTC 输出上既没法在评测里分开统计，也让同一行既要建模
一个声学事件、又要建模一个声学事件的**缺席**。
"""
import logging
import random
import time
from typing import Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast

from funasr.models.ctc.ctc import CTC
from funasr.models.tsasr.spk_ce_head import SpeakerCEHead
from funasr.register import tables
from funasr.train_utils.device_funcs import force_gatherable
from funasr.utils import postprocess_utils
from funasr.utils.datadir_writer import DatadirWriter
from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video

@tables.register("model_classes", "KwsAsrConformerDae")
class KwsAsrConformerDae(nn.Module):
    """KCE 关键词分支 + 每层 cross-attention 的 Conformer + CTC 头。

    Args:
        encoder / encoder_conf: 声学分支，须支持 `cue_feat/cue_valid`
            （`ConformerKceEncoderDae`）。
        cue_encoder / cue_encoder_conf: 关键词分支，从 `phoneme_encoder_classes` 取
            （`KeywordTransformerEncoderDae`）。None 时退化为纯 CTC ASR。
        specaug / normalize: 与其他 FunASR 模型一致；本实验两者都是 null。
        ctc_conf: 传给 `funasr.models.ctc.ctc.CTC`。
        input_size: 输入特征维度，`build_model` 用 `frontend.output_size()` 覆盖。
        vocab_size: 词表大小，`build_model` 从 `len(tokenizer.token_list)` 注入（8408）。
        psok_id / peok_id: 关键词起止的专用 token id（`<psok>`/`<peok>` = 8404/8405）。
            既是 `with_trans` 括号对，也是 `inference` 里给线索加的两端标记；拼 ASR
            文本时跳过它们并单独返回位置。
        punk_id: 历史参数，只在数据侧的负样本**线索**兜底里还有用（见
            `datasets_kws_span_tsasr.py`）；模型侧不用它，保留是为了让老 config.yaml
            仍能加载。
        no_kws_id / no_tgt_spk_id: 两个拒识 token（`<no_kws/>` = 8406 / `<no_tgt_spk/>`
            = 8407）。模型侧只用于 `_strip_reject` 把它们从假设里摘出来单独计数，
            不让拒识符污染 CER 分子。
        report_cer: 非训练态用 CTC greedy 解码算 token 级 CER 写进 stats。
        spk_ce_conf: 说话人分类辅助头的配置（透传给 `SpeakerCEHead`，`size` 会自动填成
            encoder 输出维度）。**None / 空 dict = 完全关闭**：不建模块、encoder 不收集
            隐层、loss 与纯 CTC 逐位相同、ckpt 里不多出任何张量。开启需要数据侧发
            `spk_id`（`dataset_conf.spk2id` 指向 `script/data/make_spk2id.py` 生成的
            映射），且 `num_speakers` 必须等于那份映射的条目数。
        spk_ce_weight: 辅助 CE 在总 loss 里的权重（`loss = loss_ctc + w * loss_spk_ce`），
            对应 DAE-TSE 的 `loss_weight['sv_ce']`。仅在 `spk_ce_conf` 非空时有意义。

    另外从 `**kwargs` 里取两个只用于日志的字段（`build_model` 会把整份 config 透传过来）：
    `token_list` 用来把 id 还原成字，`train_conf.log_interval` 决定抽样打印的间隔。
    """

    def __init__(
        self,
        specaug: str = None,
        specaug_conf: dict = None,
        normalize: str = None,
        normalize_conf: dict = None,
        encoder: str = None,
        encoder_conf: dict = None,
        cue_encoder: str = None,
        cue_encoder_conf: dict = None,
        ctc_conf: dict = None,
        input_size: int = 80,
        vocab_size: int = -1,
        ignore_id: int = -1,
        blank_id: int = 0,
        psok_id: int = 8404,
        peok_id: int = 8405,
        punk_id: int = 8403,
        no_kws_id: int = 8406,
        no_tgt_spk_id: int = 8407,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        spk_ce_conf: dict = None,
        spk_ce_weight: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        if specaug is not None:
            specaug = tables.specaug_classes.get(specaug)(**specaug_conf)
        if normalize is not None:
            normalize = tables.normalize_classes.get(normalize)(**normalize_conf)

        encoder_class = tables.encoder_classes.get(encoder)
        encoder = encoder_class(input_size=input_size, **encoder_conf)
        encoder_output_size = encoder.output_size()

        if cue_encoder is not None:
            cue_cls = tables.phoneme_encoder_classes.get(cue_encoder)
            if cue_cls is None:
                raise ValueError(
                    f"Unknown cue_encoder '{cue_encoder}'. "
                    f"registered: {list(tables.phoneme_encoder_classes.keys())}"
                )
            cue_encoder = cue_cls(**(cue_encoder_conf or {}))

        self.ctc = CTC(
            odim=vocab_size, encoder_output_size=encoder_output_size, **(ctc_conf or {})
        )

        self.specaug = specaug
        self.normalize = normalize
        self.encoder = encoder
        self.cue_encoder = cue_encoder
        self.blank_id = blank_id
        self.psok_id = psok_id
        self.peok_id = peok_id
        self.punk_id = punk_id
        self.no_kws_id = no_kws_id
        self.no_tgt_spk_id = no_tgt_spk_id
        # 拒识符只在词表真的够长时才参与解码侧的摘除；旧的 8406 词表下这两个 id 越界，
        # 此时集合为空 = 行为与改动前完全一致，老 ckpt 的评测脚本不会因此报错。
        self.reject_ids = {
            i for i in (no_kws_id, no_tgt_spk_id)
            if vocab_size is None or vocab_size <= 0 or 0 <= i < vocab_size
        }
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.length_normalized_loss = length_normalized_loss
        self.report_cer = report_cer
        self._warned_no_cue = False

        # === 说话人分类辅助头（默认关闭）===
        self.spk_ce_weight = float(spk_ce_weight)
        if spk_ce_conf:
            conf = dict(spk_ce_conf)
            conf.setdefault("size", encoder_output_size)
            self.spk_ce = SpeakerCEHead(**conf)
            logging.info(
                "[%s] auxiliary speaker CE head enabled: %d speakers, layers %s, weight %.4g",
                type(self).__name__,
                self.spk_ce.num_speakers,
                self.spk_ce.layer_indices,
                self.spk_ce_weight,
            )
        else:
            self.spk_ce = None
        self._warned_no_spk_id = False

        # === 训练态抽样打印（线索 / CTC目标 / 解码）===
        # token_list 与 train_conf 都是 build_model 通过 **kwargs 透传进来的
        # （`auto_model.py` 里 `deep_update(model_conf, kwargs)` 之后整个 config 都在 kwargs 里），
        # 所以不用给 __init__ 增加显式参数，也不影响其他调用方。
        # 间隔复用 train_conf.log_interval：抽样行与 trainer 的 loss 行同频出现，便于对读；
        # 推理路径没有 train_conf -> 0 -> 永不打印。
        self.token_list = kwargs.get("token_list", None)
        self.sample_log_interval = int((kwargs.get("train_conf") or {}).get("log_interval", 0) or 0)
        # 本 rank 的**前向**计数，进程启动即归零。accum_grad=1 时等于 optimizer step 数，
        # resume 之后不会接着上次的步数走（只用于控制打印频率，无需精确）。
        self._fwd_count = 0

    # ------------------------------------------------------------------ #
    def _encode_cue(self, cue_token, cue_token_lengths):
        """把 char 线索编码成 (cue_feat, cue_valid)；无线索时返回 (None, None)。

        Args:
            cue_token: (B, L) 关键词 char id，0 为 padding。
            cue_token_lengths: (B,) 或 (B, 1) 有效长度。

        Returns:
            (cue_feat, cue_valid)。
        """
        if self.cue_encoder is None or cue_token is None:
            if self.cue_encoder is not None and not self._warned_no_cue:
                logging.warning(f"[{type(self).__name__}] 未提供 cue_token，退化为无线索前向。")
                self._warned_no_cue = True
            return None, None
        if cue_token_lengths is None:
            cue_token_lengths = torch.full(
                (cue_token.size(0),), cue_token.size(1), dtype=torch.int32
            )
        cue_token = cue_token.to(torch.long)
        cue_token_lengths = cue_token_lengths.to(cue_token.device).view(-1)
        return self.cue_encoder(cue_token, cue_token_lengths)

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        cue_token: torch.Tensor = None,
        cue_token_lengths: torch.Tensor = None,
        **kwargs,
    ):
        """specaug -> normalize -> 关键词分支 -> 每层注入的 Conformer。

        Args:
            speech: (B, T, 80) fbank。
            speech_lengths: (B,) 有效帧长。
            cue_token: (B, L) 关键词 char id。
            cue_token_lengths: (B,) 关键词有效长度。

        Returns:
            (encoder_out, encoder_out_lens, extra)，extra 含逐层 cross-attention，
            且在说话人头开启时含 `hidden_states`。
        """
        with autocast(False):
            if self.specaug is not None and self.training:
                speech, speech_lengths = self.specaug(speech, speech_lengths)
            if self.normalize is not None:
                speech, speech_lengths = self.normalize(speech, speech_lengths)

        cue_feat, cue_valid = self._encode_cue(cue_token, cue_token_lengths)
        if cue_feat is not None:
            cue_feat = cue_feat.to(speech.dtype)

        encoder_out, encoder_out_lens, extra = self.encoder(
            speech,
            speech_lengths,
            cue_feat=cue_feat,
            cue_valid=cue_valid,
            return_hidden=self.spk_ce is not None,
        )
        return encoder_out, encoder_out_lens, extra

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        cue_token: torch.Tensor = None,
        cue_token_lengths: torch.Tensor = None,
        spk_id: torch.Tensor = None,
        kws_target: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Encoder + CTC（+ 可选的说话人辅助 CE），返回 (loss, stats, weight)。

        Args:
            speech: (B, T, 80) fbank。
            speech_lengths: (B,) 或 (B, 1) 有效帧长。
            text: (B, L_t) CTC 目标（已含 psok/peok），pad 值 -1。
            text_lengths: (B,) 或 (B, 1) 目标长度。
            cue_token: (B, L_k) 关键词 char id，pad 值 0。
            cue_token_lengths: (B,) 或 (B, 1) 关键词长度。
            spk_id: (B,) 或 (B, 1) 目标说话人 id，负值 = 不在映射里（忽略）。
                只在 `spk_ce_conf` 开启时用；数据侧靠 `dataset_conf.spk2id` 提供。
            kws_target: (B,) 或 (B, 1) 线索是否真的出现在这条音频里（数据集恒发）。
                在辅助头里当 `valid` 掩码用——负样本的线索借自别的句子，混合语音里
                **没有**任何线索指定的目标说话人，「目标说话人是谁」这个问题无解，
                必须忽略。这就是 DAE-TSE `CE.forward` 里的 `is_target`。

        Returns:
            (loss, stats, weight)，经 `force_gatherable` 处理。
        """
        if text_lengths.dim() > 1:
            text_lengths = text_lengths[:, 0]
        if speech_lengths.dim() > 1:
            speech_lengths = speech_lengths[:, 0]
        if cue_token_lengths is not None and cue_token_lengths.dim() > 1:
            cue_token_lengths = cue_token_lengths[:, 0]

        batch_size = speech.shape[0]
        encoder_out, encoder_out_lens, extra = self.encode(
            speech, speech_lengths, cue_token=cue_token, cue_token_lengths=cue_token_lengths
        )

        loss_ctc = self.ctc(encoder_out, encoder_out_lens, text, text_lengths)
        loss = loss_ctc

        stats = {"loss_ctc": loss_ctc.detach()}
        if self.spk_ce is not None:
            spk_loss, spk_acc = self._spk_ce_loss(extra, encoder_out_lens, spk_id, kws_target)
            if spk_loss is not None:
                loss = loss + self.spk_ce_weight * spk_loss
                stats["loss_spk_ce"] = spk_loss.detach()
                stats["acc_spk"] = spk_acc.detach()
        stats["loss"] = torch.clone(loss.detach())

        cer_ctc = self._calc_cer_ctc(encoder_out, encoder_out_lens, text, text_lengths)
        if cer_ctc is not None:
            stats["cer_ctc"] = cer_ctc
            # trainer 的进度条与选 ckpt 都只看 acc，这里把 1-CER 塞进 acc 位。
            # 辅助头**不**动这一位：选 ckpt 的判据必须一直是转写质量。
            stats["acc"] = 1.0 - cer_ctc

        if self.length_normalized_loss:
            batch_size = int(text_lengths.sum())
        if self.training:
            self._maybe_log_sample(
                encoder_out,
                encoder_out_lens,
                text,
                text_lengths,
                cue_token,
                cue_token_lengths,
                kwargs.get("mix_info"),
            )
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def _spk_ce_loss(self, extra, encoder_out_lens, spk_id, kws_target):
        """算辅助 CE；数据侧没给 `spk_id` 时打一次 warning 并返回 (None, None)。

        Args:
            extra: `encode` 返回的第三项，须含 `hidden_states`。
            encoder_out_lens: (B,) encoder 输出有效帧数。
            spk_id: (B,) 或 (B, 1) 目标说话人 id；None = 无监督。
            kws_target: (B,) 或 (B, 1) 线索真假标记，当 `valid` 掩码用；None = 全部参与。

        Returns:
            `(loss, acc)`，或 `(None, None)`。
        """
        if spk_id is None:
            if not self._warned_no_spk_id:
                logging.warning(
                    "[%s] model_conf.spk_ce_conf 已配置，但 batch 里没有 spk_id，"
                    "说话人辅助头收不到任何监督。请给 dataset_conf 加上 spk2id"
                    "（用 script/data/make_spk2id.py 生成）。",
                    type(self).__name__,
                )
                self._warned_no_spk_id = True
            return None, None
        hidden_states = (extra or {}).get("hidden_states")
        if not hidden_states:
            raise RuntimeError(
                f"[{type(self).__name__}] 说话人辅助头需要 encoder 返回 "
                "extra['hidden_states']，但没拿到；当前 encoder "
                f"({type(self.encoder).__name__}) 的 forward 需要支持 return_hidden=True"
            )
        return self.spk_ce(hidden_states, encoder_out_lens, spk_id, valid=kws_target)

    @torch.no_grad()
    def _ids2text(self, ids) -> str:
        """把 token id 序列拼成可读字符串，越界 id 显示成 `<id>`。"""
        out = []
        for x in ids:
            x = int(x)
            out.append(self.token_list[x] if 0 <= x < len(self.token_list) else f"<{x}>")
        return "".join(out)

    @torch.no_grad()
    def _maybe_log_sample(
        self,
        encoder_out,
        encoder_out_lens,
        text,
        text_lengths,
        cue_token,
        cue_token_lengths,
        mix_info=None,
    ):
        """每 `log_interval` 次前向，随机抽 batch 里的一条打印三行（+ 混音溯源）。

        三个字段互不冗余：
            cue  送进 cue encoder 的线索序列，两端是 `_wrap_cue` 加的 psok/peok；
            ref  CTC 目标，线索片段的**原位**带一对 psok/peok（`with_trans`）；
            hyp  同一条的 CTC greedy 解码（去 blank + 去重复），能直接看出括号打在哪。
        `mix_conf.prob: 1.0` 下每条音频里都有第二个说话人，所以再打一段 `mix`：
        本条与各干扰源的**音频路径 + 转写**（由 dataset 的 `collator` 放进
        `batch["mix_info"]`，见 `AudioDatasetKwsSpan._describe_mix`）。有它才能判定
        `hyp` 里多出来的字是「抄了干扰人的话」还是模型自己幻觉——这是 100% 混音训练下
        最常见的错误形态，只看 cue/ref/hyp 三行区分不了。
        `mix_info` 缺失（旧 dataset / 其他数据集类）时这段自动不打，不报错。

        只在 rank 0 打：四个 rank 都打会把日志刷成 4 倍且内容无关（各 rank 数据不同）。
        额外开销 = 一条样本的 `ctc.argmax`（一次 linear + argmax，no_grad），可忽略。

        Args:
            encoder_out: (B, T', D) 编码器输出。
            encoder_out_lens: (B,) 有效帧长。
            text / text_lengths: CTC 目标及长度，pad 值 `ignore_id`。
            cue_token / cue_token_lengths: 线索 id 及长度，pad 值 0；可为 None。
            mix_info: list[str]，长度 B 的混音溯源串；None 表示不可用。
        """
        self._fwd_count += 1
        if self.sample_log_interval <= 0 or self._fwd_count % self.sample_log_interval != 0:
            return
        if not self.token_list:
            return
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return

        i = random.randrange(encoder_out.size(0))
        n = int(encoder_out_lens[i])
        hyp = torch.unique_consecutive(self.ctc.argmax(encoder_out[i : i + 1, :n])[0])
        hyp = [int(x) for x in hyp.tolist() if x != self.blank_id]
        ref = [int(x) for x in text[i, : int(text_lengths[i])].tolist() if x != self.ignore_id]
        cue = []
        if cue_token is not None:
            L = int(cue_token_lengths[i]) if cue_token_lengths is not None else cue_token.size(1)
            cue = [int(x) for x in cue_token[i, :L].tolist()]
        logging.info(
            f"sample[{i}/{encoder_out.size(0)}] T={n}\n"
            f"  cue: {self._ids2text(cue)}\n"
            f"  ref: {self._ids2text(ref)}\n"
            f"  hyp: {self._ids2text(hyp)}"
            f"{self._fmt_mix_info(mix_info, i)}"
        )

    @staticmethod
    def _fmt_mix_info(mix_info, i) -> str:
        """取第 i 条的混音溯源串并加上 `  mix: ` 前缀；取不到就返回空串。

        用 try 兜住而不是逐个判类型：这纯粹是日志，任何形态不符都不该影响训练。
        """
        try:
            if mix_info is None:
                return ""
            return f"\n  mix: {mix_info[i]}"
        except Exception:
            return ""

    @torch.no_grad()
    def _calc_cer_ctc(self, encoder_out, encoder_out_lens, ys_pad, ys_pad_lens):
        """非训练态用 CTC greedy 解码算 token 级 CER（编辑距离 / 参考长度）。

        Args:
            encoder_out: (B, T', D) 编码器输出。
            encoder_out_lens: (B,) 有效帧长。
            ys_pad: (B, L) 目标 id。
            ys_pad_lens: (B,) 目标长度。

        Returns:
            float 或 None（训练态 / 关闭 report_cer 时）。
        """
        if self.training or not self.report_cer:
            return None
        import editdistance

        ys_hat = self.ctc.argmax(encoder_out)
        n_err, n_ref = 0, 0
        for i in range(ys_hat.size(0)):
            hyp = ys_hat[i, : encoder_out_lens[i]]
            hyp = torch.unique_consecutive(hyp)
            hyp = [int(x) for x in hyp.tolist() if x != self.blank_id]
            ref = [int(x) for x in ys_pad[i, : ys_pad_lens[i]].tolist() if x != self.ignore_id]
            n_err += editdistance.eval(hyp, ref)
            n_ref += len(ref)
        return n_err / max(n_ref, 1)

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    def _prepare_speech(self, data_in, data_lengths, tokenizer, frontend, meta_data, **kwargs):
        """把推理入参统一成 (speech, speech_lengths)，顺带把耗时写进 `meta_data`。

        `data_type=="fbank"` 且已是 Tensor 时直接用，否则读音频再算 fbank。
        （2026-08-11 从 `inference` 里原样抽出，供子类复用，行为未变。）

        Args:
            data_in: 音频路径 / 波形 / fbank。
            data_lengths: `data_type=="fbank"` 时的长度。
            tokenizer / frontend: 透传给 `load_audio_text_image_video` / `extract_fbank`。
            meta_data: 就地写入 `load_data` / `extract_feat` / `batch_data_time`。

        Returns:
            (speech, speech_lengths)，均在 CPU 上，调用方自行 `.to(device)`。
        """
        if isinstance(data_in, torch.Tensor) and kwargs.get("data_type", "sound") == "fbank":
            speech, speech_lengths = data_in, data_lengths
            if speech.dim() < 3:
                speech = speech[None, :, :]
            if speech_lengths is None:
                speech_lengths = speech.shape[1]
            return speech, speech_lengths

        time1 = time.perf_counter()
        audio_sample_list = load_audio_text_image_video(
            data_in,
            fs=frontend.fs,
            audio_fs=kwargs.get("fs", 16000),
            data_type=kwargs.get("data_type", "sound"),
            tokenizer=tokenizer,
        )
        time2 = time.perf_counter()
        meta_data["load_data"] = f"{time2 - time1:0.3f}"
        speech, speech_lengths = extract_fbank(
            audio_sample_list,
            data_type=kwargs.get("data_type", "sound"),
            frontend=frontend,
        )
        time3 = time.perf_counter()
        meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
        meta_data["batch_data_time"] = (
            speech_lengths.sum().item() * frontend.frame_shift * frontend.lfr_n / 1000
        )
        return speech, speech_lengths

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        """CTC greedy 解码。

        线索通过 `kwargs["cue_token"]`（list[int] / list[list[int]] / Tensor，**须自带
        psok/peok**）或 `kwargs["cue_text"]`（str / list[str]，由 `tokenizer` 现编并自动
        补括号）传入；B=1 的线索会按声学 batch 自动展开。

        每条结果除 `key/token/text` 外还带：
            `kws_hit`   CTC 假设里是否出现了 `psok … peok`；
            `kws_span`  括号对之间的 token 下标区间（相对去 blank 后的假设序列）；
            `frame_span` 括号对对应的**帧**区间（相对 encoder 输出，40ms/帧）；
            `cross_att` 仅在 `kwargs["return_att"]` 为真时给出，最后一层 cross-attention
                        的 (T', L) numpy 数组（多头取均值）；
            `cue_token` 与 `cross_att` 同时给出，是实际送进 cue encoder 的 L 个 id
                        （含两端括号），下游按它切列 / 标 y 轴。
        `text` 里已剔除 psok/peok，不会污染 CER。

        Args:
            data_in: 音频路径 / 波形 / fbank。
            data_lengths: `data_type=="fbank"` 时的长度。
            key: 每条样本的 key 列表。
            tokenizer: 用于 ids2tokens 与 cue_text 编码。
            frontend: WavFrontend。

        Returns:
            (results, meta_data)。
        """
        meta_data = {}
        speech, speech_lengths = self._prepare_speech(
            data_in, data_lengths, tokenizer, frontend, meta_data, **kwargs
        )

        speech = speech.to(device=kwargs["device"])

        speech_lengths = speech_lengths.to(device=kwargs["device"])

        cue_token, cue_token_lengths = self._prepare_cue_for_inference(
            kwargs, tokenizer, speech.size(0), speech.device
        )
        encoder_out, encoder_out_lens, extra = self.encode(
            speech, speech_lengths, cue_token=cue_token, cue_token_lengths=cue_token_lengths
        )
        ctc_logits = self.ctc.log_softmax(encoder_out)

        return_att = bool(kwargs.get("return_att", False))
        att_last = None
        if return_att and extra is not None and extra.get("cross_att"):
            att_last = extra["cross_att"][-1].float().mean(dim=1).cpu().numpy()

        results = []
        b = encoder_out.size(0)
        if isinstance(key[0], (list, tuple)):
            key = key[0]
        if len(key) < b:
            key = key * b
        for i in range(b):
            n_frame = int(encoder_out_lens[i])
            frame_ids = ctc_logits[i, :n_frame, :].argmax(dim=-1)
            # 保留帧下标，才能把 psok/peok 映射回帧区间
            keep = torch.ones_like(frame_ids, dtype=torch.bool)
            keep[1:] = frame_ids[1:] != frame_ids[:-1]
            keep &= frame_ids != self.blank_id
            frame_idx = torch.nonzero(keep, as_tuple=False).view(-1)
            token_int = [int(x) for x in frame_ids[keep].tolist()]
            frame_of_token = [int(x) for x in frame_idx.tolist()]

            kws_hit, kws_span, frame_span, asr_ids = self._split_brackets(
                token_int, frame_of_token
            )

            ibest_writer = None
            if kwargs.get("output_dir") is not None:
                if not hasattr(self, "writer"):
                    self.writer = DatadirWriter(kwargs.get("output_dir"))
                ibest_writer = self.writer["1best_recog"]

            token = tokenizer.ids2tokens(asr_ids)
            clean_ids, reject = self._strip_reject(asr_ids)
            text_postprocessed, _ = postprocess_utils.sentence_postprocess(
                tokenizer.ids2tokens(clean_ids)
            )
            item = {
                "key": key[i],
                "token": token,
                "text": text_postprocessed,
                "kws_hit": kws_hit,
                "kws_span": kws_span,
                "frame_span": frame_span,
                "no_kws": reject["no_kws"],
                "no_tgt_spk": reject["no_tgt_spk"],
            }
            if att_last is not None:
                kw_len = int(cue_token_lengths[i]) if cue_token_lengths is not None else 0
                item["cross_att"] = att_last[i, :n_frame, :kw_len]
                # 线索含 psok/peok，L 不等于 len(tokenizer.encode(cue_text))，
                # 下游要按这份 id 去切 att / 标 y 轴，不要自己重算
                item["cue_token"] = [int(x) for x in cue_token[i, :kw_len].tolist()]
            results.append(item)
            if ibest_writer is not None:
                ibest_writer["token"][key[i]] = " ".join(token)
                ibest_writer["text"][key[i]] = text_postprocessed

        return results, meta_data

    def _split_brackets(self, token_int, frame_of_token):
        """从 CTC 假设里摘出 `psok … peok`，返回命中标记、token 区间、帧区间与纯 ASR id。

        Args:
            token_int: 去 blank/去重复后的 token id 列表。
            frame_of_token: 与 `token_int` 等长的帧下标列表。

        Returns:
            (kws_hit, kws_span, frame_span, asr_ids)：未命中时两个 span 为 None。
        """
        asr_ids, asr_frames = [], []
        s_at = e_at = None
        for tid, fr in zip(token_int, frame_of_token):
            if tid == self.psok_id:
                if s_at is None:
                    s_at = len(asr_ids)
            elif tid == self.peok_id:
                if s_at is not None and e_at is None:
                    e_at = len(asr_ids)
            else:
                asr_ids.append(tid)
                asr_frames.append(fr)
        kws_hit = s_at is not None and e_at is not None and e_at > s_at
        if not kws_hit:
            return False, None, None, asr_ids
        frame_span = (asr_frames[s_at], asr_frames[e_at - 1] + 1)
        return True, (s_at, e_at), frame_span, asr_ids

    def _strip_reject(self, asr_ids):
        """把拒识 token 从假设里摘出来，返回纯 ASR id 与两类拒识的计数。

        必须在**算 text 之前**摘：`postprocess_utils.sentence_postprocess` 只洗
        `<s>/</s>/<unk>/<OOV>`，`<no_kws/>` / `<no_tgt_spk/>` 会原样留在 `hyp` 里，
        进而全额计入 CER 的插入错误。`token` 字段仍返回**未摘**的原始序列，拒识判据
        既可以读这里的计数、也可以自己数 token，两条路一致。

        Args:
            asr_ids: `_split_brackets` 摘掉括号后的 id 列表。

        Returns:
            (clean_ids, {"no_kws": int, "no_tgt_spk": int})
        """
        counts = {"no_kws": 0, "no_tgt_spk": 0}
        if not self.reject_ids:
            return asr_ids, counts
        clean = []
        for tid in asr_ids:
            if tid == self.no_kws_id and tid in self.reject_ids:
                counts["no_kws"] += 1
            elif tid == self.no_tgt_spk_id and tid in self.reject_ids:
                counts["no_tgt_spk"] += 1
            else:
                clean.append(tid)
        return clean, counts

    def _prepare_cue_for_inference(self, kwargs, tokenizer, batch_size, device):
        """把推理入参里的 cue_text / cue_token 整理成 (B, L) long 与 (B,) 长度。

        `cue_text` 路径会自动在两端补 `psok`/`peok`，与训练时 `_wrap_cue` 一致；
        直接传 `cue_token` 则**原样使用**，调用方需自己带好括号（避免重复包裹）。

        Args:
            kwargs: 推理关键字参数，读取 `cue_token` / `cue_text` / `cue_token_lengths`。
            tokenizer: 用于把 `cue_text` 编码成 id。
            batch_size: 声学 batch 大小，B=1 的线索按它展开。
            device: 目标设备。

        Returns:
            (cue_token, cue_token_lengths)，无线索时 (None, None)。
        """
        cue_token = kwargs.get("cue_token", None)
        cue_text = kwargs.get("cue_text", None)
        if cue_token is None and cue_text is not None:
            if isinstance(cue_text, str):
                cue_text = [cue_text]
            cue_token = [
                [self.psok_id] + list(tokenizer.encode(t)) + [self.peok_id] for t in cue_text
            ]
        if cue_token is None:
            return None, None

        if not isinstance(cue_token, torch.Tensor):
            if len(cue_token) > 0 and isinstance(cue_token[0], (list, tuple)):
                seqs = [list(x) for x in cue_token]
            else:
                seqs = [list(cue_token)]
            max_len = max(len(x) for x in seqs)
            lens = [len(x) for x in seqs]
            cue_token = torch.tensor(
                [x + [0] * (max_len - len(x)) for x in seqs], dtype=torch.int64
            )
            cue_token_lengths = torch.tensor(lens, dtype=torch.int32)
        else:
            if cue_token.dim() == 1:
                cue_token = cue_token.unsqueeze(0)
            cue_token_lengths = kwargs.get("cue_token_lengths", None)
            if cue_token_lengths is None:
                cue_token_lengths = torch.full(
                    (cue_token.size(0),), cue_token.size(1), dtype=torch.int32
                )
            elif not isinstance(cue_token_lengths, torch.Tensor):
                cue_token_lengths = torch.tensor(cue_token_lengths, dtype=torch.int32)

        if cue_token.size(0) == 1 and batch_size > 1:
            cue_token = cue_token.expand(batch_size, -1)
            cue_token_lengths = cue_token_lengths.expand(batch_size)
        return cue_token.to(device), cue_token_lengths.to(device)
