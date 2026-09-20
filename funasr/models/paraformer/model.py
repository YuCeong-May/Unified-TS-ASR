#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import time
import copy
import torch
import logging
from torch.cuda.amp import autocast
from typing import Union, Dict, List, Tuple, Optional

from funasr.register import tables
from funasr.models.ctc.ctc import CTC
from funasr.utils import postprocess_utils
from funasr.metrics.compute_acc import th_accuracy
from funasr.train_utils.device_funcs import to_device
from funasr.utils.datadir_writer import DatadirWriter
from funasr.models.paraformer.search import Hypothesis
from funasr.models.paraformer.cif_predictor import mae_loss
from funasr.train_utils.device_funcs import force_gatherable
from funasr.losses.label_smoothing_loss import LabelSmoothingLoss
from funasr.models.transformer.utils.add_sos_eos import add_sos_eos
from funasr.models.transformer.utils.nets_utils import make_pad_mask
from funasr.utils.timestamp_tools import ts_prediction_lfr6_standard
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank


def ctc_forced_align_viterbi(logp, tokens, blank_id: int = 0):
    """CTC 强制对齐（Viterbi 动态规划）。

    给定 CTC 每帧 log 概率与一条 token 序列（如 Paraformer greedy 解码结果），
    用 CTC 拓扑（token 间插 blank，禁止相同相邻 token 直接跳转）做 Viterbi 搜索，
    找出概率最高的对齐路径，并返回：
      - path_logp: 最优路径的对数概率（含 blank 帧，所有对齐帧 log 概率之和）
      - token_logp: 每个 token 的得分（对齐到该 token 的帧的 log 概率之和）
      - token_frames: 每个 token 对齐到的帧数
      - frame_ids: 逐帧对齐路径（长度 T，每帧发射的 token id，blank 帧为 blank_id）
    路径不可达（例如帧数 < token 数）时退化为逐 token 取单帧最优，frame_ids 返回 []。

    Args:
        logp: numpy.ndarray (T, V)，每帧 log_softmax 后的对数概率。
        tokens: List[int]，目标 token id 序列（已去 sos/eos/blank）。
        blank_id: CTC blank 的 id（默认 0）。
    """
    import numpy as np

    T = logp.shape[0]
    L = len(tokens)
    if L == 0:
        return 0.0, [], []

    # 帧数不足以放下所有 token（每个 token 至少占 1 帧）：退化为逐 token 取单帧最优
    if T < L:
        token_logp = [float(logp[:, tk].max()) for tk in tokens]
        token_frames = [1] * L
        return float(sum(token_logp)), token_logp, token_frames, []

    # 扩展序列：blank, y1, blank, y2, ..., yL, blank  (长度 S = 2L+1)
    ext = np.empty(2 * L + 1, dtype=np.int64)
    ext[0::2] = blank_id
    ext[1::2] = np.asarray(tokens, dtype=np.int64)
    S = ext.shape[0]

    NEG = -1e30
    em = logp[:, ext]  # (T, S) 每个状态在每帧的发射 log 概率

    dp = np.full((T, S), NEG, dtype=np.float64)
    bp = np.zeros((T, S), dtype=np.int64)
    dp[0, 0] = em[0, 0]
    if S > 1:
        dp[0, 1] = em[0, 1]

    # 允许 s-2 -> s 跳转（跨过一个 blank）：当前非 blank 且与 s-2 的 token 不同
    skip_ok = np.zeros(S, dtype=bool)
    for s in range(2, S):
        if ext[s] != blank_id and ext[s] != ext[s - 2]:
            skip_ok[s] = True

    idx = np.arange(S)
    for t in range(1, T):
        prev = dp[t - 1]
        cand0 = prev  # 停留在 s
        cand1 = np.full(S, NEG)
        cand1[1:] = prev[:-1]  # 从 s-1
        cand2 = np.full(S, NEG)
        cand2[2:] = np.where(skip_ok[2:], prev[:-2], NEG)  # 从 s-2
        stack = np.stack([cand0, cand1, cand2], axis=0)  # (3, S)
        argmx = np.argmax(stack, axis=0)
        best = stack[argmx, idx]
        dp[t] = best + em[t]
        bp[t] = idx - argmx  # 回溯指针 = s - (0/1/2)

    # 终点必须停在最后一个 blank (S-1) 或最后一个 token (S-2)
    if S == 1:
        end_s = 0
    elif dp[T - 1, S - 1] >= dp[T - 1, S - 2]:
        end_s = S - 1
    else:
        end_s = S - 2
    path_logp = float(dp[T - 1, end_s])

    # 回溯得到逐帧状态
    frame_state = np.zeros(T, dtype=np.int64)
    s = end_s
    for t in range(T - 1, -1, -1):
        frame_state[t] = s
        s = int(bp[t, s])

    # 逐 token 累计得分 + 逐帧对齐路径
    token_logp = [0.0] * L
    token_frames = [0] * L
    frame_ids = [0] * T
    for t in range(T):
        s = int(frame_state[t])
        tid = int(ext[s])
        frame_ids[t] = tid
        if tid != blank_id:
            j = (s - 1) // 2
            token_logp[j] += float(em[t, s])
            token_frames[j] += 1

    # 极少数 token 未分到帧（对齐退化）：用单帧最优兜底
    for j in range(L):
        if token_frames[j] == 0:
            token_logp[j] = float(logp[:, tokens[j]].max())
            token_frames[j] = 1

    return path_logp, token_logp, token_frames, frame_ids


@tables.register("model_classes", "Paraformer")
class Paraformer(torch.nn.Module):
    """Paraformer: Non-autoregressive End-to-End ASR Model.

    High-accuracy speech recognition for Chinese/English. The production workhorse.

    Features:
    - Non-autoregressive (parallel decoding, fast inference)
    - Character-level timestamps via CIF predictor
    - Streaming and offline modes
    - Hotword customization
    - Speaker diarization (with spk_model)
    - ONNX export support

    Output: {"key": "...", "text": "recognized text", "timestamp": [[start_ms, end_ms], ...]}

    Note: Requires punc_model="ct-punc" for punctuation (unlike Fun-ASR-Nano/SenseVoice
    which output punctuation natively).

    Author: Speech Lab of DAMO Academy, Alibaba Group
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2206.08317
    """

    def __init__(
        self,
        specaug: Optional[str] = None,
        specaug_conf: Optional[Dict] = None,
        normalize: str = None,
        normalize_conf: Optional[Dict] = None,
        encoder: str = None,
        encoder_conf: Optional[Dict] = None,
        decoder: str = None,
        decoder_conf: Optional[Dict] = None,
        ctc: str = None,
        ctc_conf: Optional[Dict] = None,
        predictor: str = None,
        predictor_conf: Optional[Dict] = None,
        ctc_weight: float = 0.5,
        input_size: int = 80,
        vocab_size: int = -1,
        ignore_id: int = -1,
        blank_id: int = 0,
        sos: int = 1,
        eos: int = 2,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        # report_cer: bool = True,
        # report_wer: bool = True,
        # sym_space: str = "<space>",
        # sym_blank: str = "<blank>",
        # extract_feats_in_collect_stats: bool = True,
        # predictor=None,
        predictor_weight: float = 0.0,
        predictor_bias: int = 0,
        sampling_ratio: float = 0.2,
        share_embedding: bool = False,
        # preencoder: Optional[AbsPreEncoder] = None,
        # postencoder: Optional[AbsPostEncoder] = None,
        use_1st_decoder_loss: bool = False,
        **kwargs,
    ):

        """Initialize Paraformer.
        
            Args:
                specaug: TODO.
                specaug_conf: Configuration dict for specaug.
                normalize: TODO.
                normalize_conf: Configuration dict for normalize.
                encoder: TODO.
                encoder_conf: Configuration dict for encoder.
                decoder: TODO.
                decoder_conf: Configuration dict for decoder.
                ctc: TODO.
                ctc_conf: Configuration dict for ctc.
                predictor: TODO.
                predictor_conf: Configuration dict for predictor.
                ctc_weight: TODO.
                input_size: Size/dimension parameter.
                vocab_size: Size/dimension parameter.
                ignore_id: TODO.
                blank_id: TODO.
                sos: TODO.
                eos: TODO.
                lsm_weight: TODO.
                length_normalized_loss: TODO.
                predictor_weight: TODO.
                predictor_bias: TODO.
                sampling_ratio: TODO.
                share_embedding: TODO.
                use_1st_decoder_loss: TODO.
                **kwargs: Additional keyword arguments.
            """
        super().__init__()

        if specaug is not None:
            specaug_class = tables.specaug_classes.get(specaug)
            specaug = specaug_class(**specaug_conf)
        if normalize is not None:
            normalize_class = tables.normalize_classes.get(normalize)
            normalize = normalize_class(**normalize_conf)
        encoder_class = tables.encoder_classes.get(encoder)
        encoder = encoder_class(input_size=input_size, **encoder_conf)
        encoder_output_size = encoder.output_size()

        if decoder is not None:
            decoder_class = tables.decoder_classes.get(decoder)
            decoder = decoder_class(
                vocab_size=vocab_size,
                encoder_output_size=encoder_output_size,
                **decoder_conf,
            )
        if ctc_weight > 0.0:

            if ctc_conf is None:
                ctc_conf = {}

            ctc = CTC(odim=vocab_size, encoder_output_size=encoder_output_size, **ctc_conf)
        if predictor is not None:
            predictor_class = tables.predictor_classes.get(predictor)
            predictor = predictor_class(**predictor_conf)

        # note that eos is the same as sos (equivalent ID)
        self.blank_id = blank_id
        self.sos = sos if sos is not None else vocab_size - 1
        self.eos = eos if eos is not None else vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        # self.token_list = token_list.copy()
        #
        # self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        # self.preencoder = preencoder
        # self.postencoder = postencoder
        self.encoder = encoder
        #
        # if not hasattr(self.encoder, "interctc_use_conditioning"):
        #     self.encoder.interctc_use_conditioning = False
        # if self.encoder.interctc_use_conditioning:
        #     self.encoder.conditioning_layer = torch.nn.Linear(
        #         vocab_size, self.encoder.output_size()
        #     )
        #
        # self.error_calculator = None
        #
        if ctc_weight == 1.0:
            self.decoder = None
        else:
            self.decoder = decoder

        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )
        #
        # if report_cer or report_wer:
        #     self.error_calculator = ErrorCalculator(
        #         token_list, sym_space, sym_blank, report_cer, report_wer
        #     )
        #
        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc
        #
        # self.extract_feats_in_collect_stats = extract_feats_in_collect_stats
        self.predictor = predictor
        self.predictor_weight = predictor_weight
        self.predictor_bias = predictor_bias
        self.sampling_ratio = sampling_ratio
        self.criterion_pre = mae_loss(normalize_length=length_normalized_loss)

        self.share_embedding = share_embedding
        if self.share_embedding:
            self.decoder.embed = None

        self.use_1st_decoder_loss = use_1st_decoder_loss
        self.length_normalized_loss = length_normalized_loss
        self.beam_search = None
        self.error_calculator = None

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Encoder + Decoder + Calc loss
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch,)
        """
        if len(text_lengths.size()) > 1:
            text_lengths = text_lengths[:, 0]
        if len(speech_lengths.size()) > 1:
            speech_lengths = speech_lengths[:, 0]

        batch_size = speech.shape[0]

        # Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        loss_ctc, cer_ctc = None, None
        loss_pre = None
        stats = dict()

        # decoder: CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        # decoder: Attention decoder branch
        loss_att, acc_att, cer_att, wer_att, loss_pre, pre_loss_att = self._calc_att_loss(
            encoder_out, encoder_out_lens, text, text_lengths
        )

        # 3. CTC-Att loss definition
        if self.ctc_weight == 0.0:
            loss = loss_att + loss_pre * self.predictor_weight
        else:
            loss = (
                self.ctc_weight * loss_ctc
                + (1 - self.ctc_weight) * loss_att
                + loss_pre * self.predictor_weight
            )

        # Collect Attn branch stats
        stats["loss_att"] = loss_att.detach() if loss_att is not None else None
        stats["pre_loss_att"] = pre_loss_att.detach() if pre_loss_att is not None else None
        stats["acc"] = acc_att
        stats["cer"] = cer_att
        stats["wer"] = wer_att
        stats["loss_pre"] = loss_pre.detach().cpu() if loss_pre is not None else None

        stats["loss"] = torch.clone(loss.detach())
        stats["batch_size"] = batch_size

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = (text_lengths + self.predictor_bias).sum()
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encoder. Note that this method is used by asr_inference.py
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                ind: int
        """
        with autocast(False):

            # Data augmentation
            if self.specaug is not None and self.training:
                speech, speech_lengths = self.specaug(speech, speech_lengths)

            # Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                speech, speech_lengths = self.normalize(speech, speech_lengths)

        # Forward encoder
        encoder_out, encoder_out_lens, _ = self.encoder(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        return encoder_out, encoder_out_lens

    def calc_predictor(self, encoder_out, encoder_out_lens):

        """Calc predictor.
        
            Args:
                encoder_out: Encoder output tensor.
                encoder_out_lens: Encoder output lengths.
            """
        encoder_out_mask = (
            ~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]
        ).to(encoder_out.device)
        pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = self.predictor(
            encoder_out, None, encoder_out_mask, ignore_id=self.ignore_id
        )
        return pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index

    def cal_decoder_with_predictor(
        self, encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens
    ):

        """Cal decoder with predictor.
        
            Args:
                encoder_out: Encoder output tensor.
                encoder_out_lens: Encoder output lengths.
                sematic_embeds: TODO.
                ys_pad_lens: Lengths of ys_pad.
            """
        decoder_outs = self.decoder(encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens)
        decoder_out = decoder_outs[0]
        decoder_out = torch.log_softmax(decoder_out, dim=-1)
        return decoder_out, ys_pad_lens

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        """Internal: calc att loss.
        
            Args:
                encoder_out: Encoder output tensor.
                encoder_out_lens: Encoder output lengths.
                ys_pad: TODO.
                ys_pad_lens: Lengths of ys_pad.
            """
        encoder_out_mask = (
            ~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]
        ).to(encoder_out.device)
        if self.predictor_bias == 1:
            _, ys_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
            ys_pad_lens = ys_pad_lens + self.predictor_bias
        pre_acoustic_embeds, pre_token_length, _, pre_peak_index = self.predictor(
            encoder_out, ys_pad, encoder_out_mask, ignore_id=self.ignore_id
        )

        # 0. sampler
        decoder_out_1st = None
        pre_loss_att = None
        if self.sampling_ratio > 0.0:

            sematic_embeds, decoder_out_1st = self.sampler(
                encoder_out, encoder_out_lens, ys_pad, ys_pad_lens, pre_acoustic_embeds
            )
        else:
            sematic_embeds = pre_acoustic_embeds

        # 1. Forward decoder
        decoder_outs = self.decoder(encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens)
        decoder_out, _ = decoder_outs[0], decoder_outs[1]

        if decoder_out_1st is None:
            decoder_out_1st = decoder_out
        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_pad)
        acc_att = th_accuracy(
            decoder_out_1st.view(-1, self.vocab_size),
            ys_pad,
            ignore_label=self.ignore_id,
        )
        loss_pre = self.criterion_pre(ys_pad_lens.type_as(pre_token_length), pre_token_length)

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out_1st.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att, loss_pre, pre_loss_att

    def sampler(self, encoder_out, encoder_out_lens, ys_pad, ys_pad_lens, pre_acoustic_embeds):

        """Sampler.
        
            Args:
                encoder_out: Encoder output tensor.
                encoder_out_lens: Encoder output lengths.
                ys_pad: TODO.
                ys_pad_lens: Lengths of ys_pad.
                pre_acoustic_embeds: TODO.
            """
        tgt_mask = (~make_pad_mask(ys_pad_lens, maxlen=ys_pad_lens.max())[:, :, None]).to(
            ys_pad.device
        )
        ys_pad_masked = ys_pad * tgt_mask[:, :, 0]
        if self.share_embedding:
            ys_pad_embed = self.decoder.output_layer.weight[ys_pad_masked]
        else:
            ys_pad_embed = self.decoder.embed(ys_pad_masked)
        with torch.no_grad():
            decoder_outs = self.decoder(
                encoder_out, encoder_out_lens, pre_acoustic_embeds, ys_pad_lens
            )
            decoder_out, _ = decoder_outs[0], decoder_outs[1]
            pred_tokens = decoder_out.argmax(-1)
            nonpad_positions = ys_pad.ne(self.ignore_id)
            seq_lens = (nonpad_positions).sum(1)
            same_num = ((pred_tokens == ys_pad) & nonpad_positions).sum(1)
            input_mask = torch.ones_like(nonpad_positions)
            bsz, seq_len = ys_pad.size()
            for li in range(bsz):
                target_num = (
                    ((seq_lens[li] - same_num[li].sum()).float()) * self.sampling_ratio
                ).long()
                if target_num > 0:
                    input_mask[li].scatter_(
                        dim=0,
                        index=torch.randperm(seq_lens[li])[:target_num].to(input_mask.device),
                        value=0,
                    )
            input_mask = input_mask.eq(1)
            input_mask = input_mask.masked_fill(~nonpad_positions, False)
            input_mask_expand_dim = input_mask.unsqueeze(2).to(pre_acoustic_embeds.device)

        sematic_embeds = pre_acoustic_embeds.masked_fill(
            ~input_mask_expand_dim, 0
        ) + ys_pad_embed.masked_fill(input_mask_expand_dim, 0)
        return sematic_embeds * tgt_mask, decoder_out * tgt_mask

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        """Internal: calc ctc loss.
        
            Args:
                encoder_out: Encoder output tensor.
                encoder_out_lens: Encoder output lengths.
                ys_pad: TODO.
                ys_pad_lens: Lengths of ys_pad.
            """
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def init_beam_search(
        self,
        **kwargs,
    ):
        """Init beam search.
        
            Args:
                **kwargs: Additional keyword arguments.
            """
        from funasr.models.paraformer.search import BeamSearchPara
        from funasr.models.transformer.scorers.ctc import CTCPrefixScorer
        from funasr.models.transformer.scorers.length_bonus import LengthBonus

        # 1. Build ASR model
        scorers = {}

        if self.ctc != None:
            ctc = CTCPrefixScorer(ctc=self.ctc, eos=self.eos)
            scorers.update(ctc=ctc)
        token_list = kwargs.get("token_list")
        scorers.update(
            length_bonus=LengthBonus(len(token_list)),
        )

        # 3. Build ngram model
        # ngram is not supported now
        ngram = None
        scorers["ngram"] = ngram

        weights = dict(
            decoder=1.0 - kwargs.get("decoding_ctc_weight"),
            ctc=kwargs.get("decoding_ctc_weight", 0.0),
            lm=kwargs.get("lm_weight", 0.0),
            ngram=kwargs.get("ngram_weight", 0.0),
            length_bonus=kwargs.get("penalty", 0.0),
        )
        beam_search = BeamSearchPara(
            beam_size=kwargs.get("beam_size", 2),
            weights=weights,
            scorers=scorers,
            sos=self.sos,
            eos=self.eos,
            vocab_size=len(token_list),
            token_list=token_list,
            pre_beam_score_key=None if self.ctc_weight == 1.0 else "full",
        )
        # beam_search.to(device=kwargs.get("device", "cpu"), dtype=getattr(torch, kwargs.get("dtype", "float32"))).eval()
        # for scorer in scorers.values():
        #     if isinstance(scorer, torch.nn.Module):
        #         scorer.to(device=kwargs.get("device", "cpu"), dtype=getattr(torch, kwargs.get("dtype", "float32"))).eval()
        self.beam_search = beam_search

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        # init beamsearch
        """Run inference on input data.
        
            Args:
                data_in: Input data (audio samples, file paths, or text).
                data_lengths: Lengths of each input sample in the batch.
                key: Sample identifiers.
                tokenizer: Tokenizer instance for text encoding/decoding.
                frontend: Audio frontend for feature extraction.
                **kwargs: Additional keyword arguments.
            """
        is_use_ctc = kwargs.get("decoding_ctc_weight", 0.0) > 0.00001 and self.ctc != None
        is_use_lm = (
            kwargs.get("lm_weight", 0.0) > 0.00001 and kwargs.get("lm_file", None) is not None
        )
        pred_timestamp = kwargs.get("pred_timestamp", False)
        if self.beam_search is None and (is_use_lm or is_use_ctc):
            logging.info("enable beam_search")
            self.init_beam_search(**kwargs)
            self.nbest = kwargs.get("nbest", 1)

        meta_data = {}
        if (
            isinstance(data_in, torch.Tensor) and kwargs.get("data_type", "sound") == "fbank"
        ):  # fbank
            speech, speech_lengths = data_in, data_lengths
            if len(speech.shape) < 3:
                speech = speech[None, :, :]
            if speech_lengths is not None:
                speech_lengths = speech_lengths.squeeze(-1)
            else:
                speech_lengths = speech.shape[1]
        else:
            # extract fbank feats
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
                audio_sample_list, data_type=kwargs.get("data_type", "sound"), frontend=frontend
            )
            time3 = time.perf_counter()
            meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
            meta_data["batch_data_time"] = (
                speech_lengths.sum().item() * frontend.frame_shift * frontend.lfr_n / 1000
            )

        speech = speech.to(device=kwargs["device"])
        speech_lengths = speech_lengths.to(device=kwargs["device"])
        # Encoder
        if kwargs.get("fp16", False):
            speech = speech.half()
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        # predictor
        predictor_outs = self.calc_predictor(encoder_out, encoder_out_lens)
        pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = (
            predictor_outs[0],
            predictor_outs[1],
            predictor_outs[2],
            predictor_outs[3],
        )
        
        pre_token_length = pre_token_length.round().long()
        if torch.max(pre_token_length) < 1:
            return []
        decoder_outs = self.cal_decoder_with_predictor(
            encoder_out, encoder_out_lens, pre_acoustic_embeds, pre_token_length
        )
        decoder_out, ys_pad_lens = decoder_outs[0], decoder_outs[1]

        results = []
        b, n, d = decoder_out.size()
        if isinstance(key[0], (list, tuple)):
            key = key[0]
        if len(key) < b:
            key = key * b
        for i in range(b):
            x = encoder_out[i, : encoder_out_lens[i], :]
            am_scores = decoder_out[i, : pre_token_length[i], :]
            if self.beam_search is not None:
                nbest_hyps = self.beam_search(
                    x=x,
                    am_scores=am_scores,
                    maxlenratio=kwargs.get("maxlenratio", 0.0),
                    minlenratio=kwargs.get("minlenratio", 0.0),
                )

                nbest_hyps = nbest_hyps[: self.nbest]
            else:

                yseq = am_scores.argmax(dim=-1)
                score = am_scores.max(dim=-1)[0]
                score = torch.sum(score, dim=-1)
                # pad with mask tokens to ensure compatibility with sos/eos tokens
                yseq = torch.tensor([self.sos] + yseq.tolist() + [self.eos], device=yseq.device)
                nbest_hyps = [Hypothesis(yseq=yseq, score=score)]
            for nbest_idx, hyp in enumerate(nbest_hyps):
                ibest_writer = None
                if kwargs.get("output_dir") is not None:
                    if not hasattr(self, "writer"):
                        self.writer = DatadirWriter(kwargs.get("output_dir"))
                    ibest_writer = self.writer[f"{nbest_idx+1}best_recog"]
                # remove sos/eos and get results
                last_pos = -1
                if isinstance(hyp.yseq, list):
                    token_int = hyp.yseq[1:last_pos]
                else:
                    token_int = hyp.yseq[1:last_pos].tolist()

                # remove blank symbol id, which is assumed to be 0
                token_int = list(
                    filter(
                        lambda x: x != self.eos and x != self.sos and x != self.blank_id, token_int
                    )
                )

                if tokenizer is not None:
                    # Change integer-ids to tokens
                    token = tokenizer.ids2tokens(token_int)
                    text_postprocessed = tokenizer.tokens2text(token)
                    
                    if pred_timestamp:
                        timestamp_str, timestamp = ts_prediction_lfr6_standard(
                            pre_peak_index[i],
                            alphas[i],
                            copy.copy(token),
                            vad_offset=kwargs.get("begin_time", 0),
                            upsample_rate=1,
                        )
                        if not hasattr(tokenizer, "bpemodel"):
                            text_postprocessed, time_stamp_postprocessed, _ = postprocess_utils.sentence_postprocess(token, timestamp)
                        result_i = {"key": key[i], "text": text_postprocessed, "timestamp": time_stamp_postprocessed,}
                    else:
                        if not hasattr(tokenizer, "bpemodel"):
                            text_postprocessed, _ = postprocess_utils.sentence_postprocess(token)
                        result_i = {"key": key[i], "text": text_postprocessed}

                    if ibest_writer is not None:
                        ibest_writer["token"][key[i]] = " ".join(token)
                        # ibest_writer["text"][key[i]] = text
                        ibest_writer["text"][key[i]] = text_postprocessed
                else:
                    result_i = {"key": key[i], "token_int": token_int}

                # CTC 强制对齐：用 CTC 每帧分布对当前解码 token 序列做 Viterbi 动态规划，
                # 输出【逐帧】对齐结果（含 blank 帧、以及同一 token 跨多帧的"合并词"帧），
                # 以及 CTC 自身的最佳路径（逐帧 argmax）。
                if kwargs.get("ctc", False) and self.ctc is not None and len(token_int) > 0:
                    try:
                        import numpy as np

                        x_ctc = encoder_out[i, : encoder_out_lens[i], :]
                        ctc_logp = self.ctc.log_softmax(x_ctc.unsqueeze(0))[0]  # (T, V)
                        logp_np = ctc_logp.detach().float().cpu().numpy()
                        path_logp, _, _, frame_ids = ctc_forced_align_viterbi(
                            logp_np, token_int, blank_id=self.blank_id
                        )

                        # 最佳 CTC 路径（逐帧 argmax）及其折叠解码（greedy CTC 结果）
                        argmax_ids = logp_np.argmax(axis=1).tolist()
                        # CTC 最佳路径对数概率 = 逐帧 argmax 的 log 概率之和
                        best_logp = float(logp_np.max(axis=1).sum())
                        best_collapsed, prev = [], None
                        for tid in argmax_ids:
                            if tid != prev:
                                best_collapsed.append(tid)
                                prev = tid
                        best_ids = [t for t in best_collapsed if t != self.blank_id]

                        # id -> token 串缓存（blank 显示为 "_"）
                        _str_cache = {}

                        def _id2str(tid):
                            if tid == self.blank_id:
                                return "_"
                            if tid not in _str_cache:
                                if tokenizer is not None:
                                    _str_cache[tid] = tokenizer.ids2tokens([tid])[0]
                                else:
                                    _str_cache[tid] = str(tid)
                            return _str_cache[tid]

                        # 逐帧 [tok_str, prob]，prob = exp(该帧该 token 的 log 概率)
                        def _frames(ids):
                            out = []
                            for t, tid in enumerate(ids):
                                p = float(np.exp(logp_np[t, tid]))
                                out.append([_id2str(tid), round(p, 4)])
                            return out

                        result_i["ctc_tokens"] = (
                            token if tokenizer is not None else [str(t) for t in token_int]
                        )
                        result_i["ctc_path_logp"] = path_logp
                        result_i["ctc_best_logp"] = best_logp
                        result_i["ctc_num_frames"] = int(x_ctc.size(0))
                        # 最佳 CTC 路径（逐帧）+ greedy 解码文本
                        result_i["ctc_best_frames"] = _frames(argmax_ids)
                        result_i["ctc_best_tokens"] = [_id2str(t) for t in best_ids]
                        # 对齐到 Paraformer token 序列的逐帧路径（degenerate 时为空）
                        result_i["ctc_align_frames"] = _frames(frame_ids) if frame_ids else []
                    except Exception as e:
                        logging.warning(f"ctc forced align failed for key={key[i]}: {e}")

                results.append(result_i)

        return results, meta_data

    def export(self, **kwargs):
        """Export.
        
            Args:
                **kwargs: Additional keyword arguments.
            """
        from .export_meta import export_rebuild_model

        if "max_seq_len" not in kwargs:
            kwargs["max_seq_len"] = 512
        models = export_rebuild_model(model=self, **kwargs)
        return models
