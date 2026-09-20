# -*- coding: utf-8 -*-
"""实验3（V3）：KCE + CTC 的 target-speaker ASR，去掉 Paraformer 的 decoder 与 CIF。

模型只剩三块：
    1. `cue_encoder`  唤醒词**中文 char** 序列 -> 上下文特征（复用实验1 的
       `PhonemeTransformerEncoder`，只是词表换成 ASR 的 char 词表）。
    2. `encoder`      `SANMEncoderKCE`，预训练 Paraformer 的 50 层 SANMEncoder，
       在指定层间插入 cross-attention adapter 接收线索。
    3. `ctc`          char 词表上的 CTC 头（帧同步，输出与 ASR 词表一致）。

去掉 decoder/CIF 的直接后果：训练目标只有 CTC，**不存在 V2 那条「长度被 teacher
forcing 缩放导致 CE 对字数零梯度」的路径**——CTC 自己就要在每一帧上决定吐字还是吐
blank，字数是解码结果而非被喂进去的条件。
"""
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from funasr.models.ctc.ctc import CTC
from funasr.register import tables
from funasr.train_utils.device_funcs import force_gatherable
from funasr.utils import postprocess_utils
from funasr.utils.datadir_writer import DatadirWriter
from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video


@tables.register("model_classes", "ParaformerKceCtc")
class ParaformerKceCtc(nn.Module):
    """Paraformer encoder + KCE 层内注入 + CTC 头（无 decoder / 无 CIF predictor）。

    Args:
        cue_encoder / cue_encoder_conf: 线索编码器，从 `phoneme_encoder_classes`
            取（实验1 注册的 `PhonemeTransformerEncoder` 对 char 词表同样适用，
            把 `num_phn_token` 设成 char 词表大小即可）。None 时退化为纯 CTC ASR。
        encoder / encoder_conf: 必须是支持 `cue_feat/cue_valid` 的编码器
            （`SANMEncoderKCE`）；线索为 None 时它与 `SANMEncoder` 行为一致。
        ctc_conf: 传给 `funasr.models.ctc.ctc.CTC`。
        report_cer: 非训练态用 CTC greedy 解码算 token 级 CER，写进 stats
            供日志观察（训练态跳过，避免拖慢）。
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
        sos: int = 1,
        eos: int = 2,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
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
        self.sos = sos if sos is not None else vocab_size - 1
        self.eos = eos if eos is not None else vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.length_normalized_loss = length_normalized_loss
        self.report_cer = report_cer
        self.error_calculator = None
        self._warned_no_cue = False

    # ------------------------------------------------------------------ #
    def _encode_cue(self, cue_token, cue_token_lengths):
        """把 char 线索编码成 (cue_feat, cue_valid)；无线索时返回 (None, None)。"""
        if self.cue_encoder is None or cue_token is None:
            if self.cue_encoder is not None and not self._warned_no_cue:
                import logging

                logging.warning(
                    f"[{type(self).__name__}] 未提供 cue_token，退化为无线索前向。"
                )
                self._warned_no_cue = True
            return None, None
        if cue_token_lengths is None:
            cue_token_lengths = torch.full(
                (cue_token.size(0),), cue_token.size(1), dtype=torch.int32
            )
        cue_token = cue_token.to(torch.long)
        cue_token_lengths = cue_token_lengths.to(cue_token.device).view(-1)
        cue_feat, cue_valid = self.cue_encoder(cue_token, cue_token_lengths)
        return cue_feat, cue_valid

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        cue_token: torch.Tensor = None,
        cue_token_lengths: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """frontend 之后：specaug -> CMVN -> KCE 层内注入的 SANMEncoder。"""
        with autocast(False):
            if self.specaug is not None and self.training:
                speech, speech_lengths = self.specaug(speech, speech_lengths)
            if self.normalize is not None:
                speech, speech_lengths = self.normalize(speech, speech_lengths)

        cue_feat, cue_valid = self._encode_cue(cue_token, cue_token_lengths)
        if cue_feat is not None:
            cue_feat = cue_feat.to(speech.dtype)

        encoder_out, encoder_out_lens, _ = self.encoder(
            speech, speech_lengths, cue_feat=cue_feat, cue_valid=cue_valid
        )
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]
        return encoder_out, encoder_out_lens

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        cue_token: torch.Tensor = None,
        cue_token_lengths: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Encoder + CTC，返回 (loss, stats, weight)。"""
        if text_lengths.dim() > 1:
            text_lengths = text_lengths[:, 0]
        if speech_lengths.dim() > 1:
            speech_lengths = speech_lengths[:, 0]
        if cue_token_lengths is not None and cue_token_lengths.dim() > 1:
            cue_token_lengths = cue_token_lengths[:, 0]

        batch_size = speech.shape[0]
        encoder_out, encoder_out_lens = self.encode(
            speech, speech_lengths, cue_token=cue_token, cue_token_lengths=cue_token_lengths
        )

        loss_ctc = self.ctc(encoder_out, encoder_out_lens, text, text_lengths)
        loss = loss_ctc

        stats = {"loss_ctc": loss_ctc.detach(), "loss": torch.clone(loss.detach())}
        cer_ctc = self._calc_cer_ctc(encoder_out, encoder_out_lens, text, text_lengths)
        if cer_ctc is not None:
            stats["cer_ctc"] = cer_ctc
            # trainer 的进度条只固定打印 acc，这里把 1-CER 塞进 acc 位便于观察
            stats["acc"] = 1.0 - cer_ctc

        if self.length_normalized_loss:
            batch_size = int(text_lengths.sum())
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    @torch.no_grad()
    def _calc_cer_ctc(self, encoder_out, encoder_out_lens, ys_pad, ys_pad_lens):
        """非训练态用 CTC greedy 解码算 token 级 CER（编辑距离/参考长度）。"""
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

        线索通过 `kwargs["cue_token"]` 传入，接受 list[int] / list[list[int]] /
        Tensor；B=1 的线索会按声学 batch 自动展开。也可直接给 `kwargs["cue_text"]`
        （字符串或字符串列表），由 `tokenizer` 现编码。
        """
        meta_data = {}
        if isinstance(data_in, torch.Tensor) and kwargs.get("data_type", "sound") == "fbank":
            speech, speech_lengths = data_in, data_lengths
            if speech.dim() < 3:
                speech = speech[None, :, :]
            if speech_lengths is None:
                speech_lengths = speech.shape[1]
        else:
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

        speech = speech.to(device=kwargs["device"])
        speech_lengths = speech_lengths.to(device=kwargs["device"])

        cue_token, cue_token_lengths = self._prepare_cue_for_inference(
            kwargs, tokenizer, speech.size(0), speech.device
        )
        encoder_out, encoder_out_lens = self.encode(
            speech, speech_lengths, cue_token=cue_token, cue_token_lengths=cue_token_lengths
        )
        ctc_logits = self.ctc.log_softmax(encoder_out)

        results = []
        b = encoder_out.size(0)
        if isinstance(key[0], (list, tuple)):
            key = key[0]
        if len(key) < b:
            key = key * b
        for i in range(b):
            yseq = ctc_logits[i, : encoder_out_lens[i], :].argmax(dim=-1)
            yseq = torch.unique_consecutive(yseq, dim=-1)
            token_int = [int(x) for x in yseq.tolist() if x != self.blank_id]

            ibest_writer = None
            if kwargs.get("output_dir") is not None:
                if not hasattr(self, "writer"):
                    self.writer = DatadirWriter(kwargs.get("output_dir"))
                ibest_writer = self.writer["1best_recog"]

            token = tokenizer.ids2tokens(token_int)
            text_postprocessed, _ = postprocess_utils.sentence_postprocess(token)
            results.append({"key": key[i], "token": token, "text": text_postprocessed})
            if ibest_writer is not None:
                ibest_writer["token"][key[i]] = " ".join(token)
                ibest_writer["text"][key[i]] = text_postprocessed

        return results, meta_data

    def _prepare_cue_for_inference(self, kwargs, tokenizer, batch_size, device):
        """把推理入参里的 cue_text / cue_token 整理成 (B,L) long 与 (B,) 长度。"""
        cue_token = kwargs.get("cue_token", None)
        cue_text = kwargs.get("cue_text", None)
        if cue_token is None and cue_text is not None:
            if isinstance(cue_text, str):
                cue_text = [cue_text]
            cue_token = [tokenizer.encode(t) for t in cue_text]
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
