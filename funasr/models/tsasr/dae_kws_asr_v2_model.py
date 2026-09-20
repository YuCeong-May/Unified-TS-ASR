# -*- coding: utf-8 -*-
"""实验5（V5）：在 V4-TSE 的 `Conformer + cue + CTC` 之后接一个**非自回归** decoder，
按 Paraformer-V2 的方式把 CTC 输出压成 token 级序列再解码。

与 V4-TSE（`KwsAsrConformerDae`）的关系：encoder / cue_encoder / ctc / 数据管线**逐字不变**，
只多一个 decoder 分支和一项 CE loss，所以能直接用 4-4 的 ckpt 热启动（decoder 随机初始化，
`load_pretrained_model` 对缺失键只打 `Warning, miss key in ckpt` 不报错）。

Paraformer-V2 的核心是**抛掉 CIF predictor，用 CTC 自己给 token 边界**：
    训练：`torchaudio.functional.forced_align` 把 CTC 后验强制对齐到**真实标签**，压缩后长度
          恒等于 `text_lengths`，decoder 输出与标签逐位置对齐 -> 普通 CE，不需要 sampler。
    推理：CTC greedy path 去 blank + 合并连续重复，压缩长度 U' 由 CTC 自己决定。
    两条路径的 decoder 输入都是**该段内 CTC 后验概率的均值**——是 (U, V) 的概率向量而不是
    token embedding，由 `decoder_conf.input_layer: linear` 的 `Linear(V, D)` 投影进 decoder。
    decoder 的 self-att 只有 padding mask（无因果 mask），一次前向出全序列。

由此而来的一个**结构性上限**：推理时 U' 完全由 CTC 决定，decoder 只能改 token、不能改长度，
所以它修得了替换错误，修不了 CTC 的插入/删除。这也是 `cer_ctc` 必须继续看的理由。

相对参考实现（`funasr/models/paraformer_v2_community/model.py`）的偏离，都有理由：
    1. 强制对齐在**归约词表**上做：只取 blank + 标签里出现过的列（<= L+1 列，而非 8406 列），
       相邻重复共用同一列以保住 CTC「重复 token 之间必须有 blank」的规则，因此与整表对齐
       数学等价。实测 B=100 / T=150：本实现 53 ms/batch，参考实现的「整表 .cpu()」465 ms，
       CUDA 版 121 ms（forced_align 的 kernel 启动开销在这种小问题上反而占主导）。
    2. 段内聚合用整 batch 一次 `index_add` 完成，不再逐条 `scatter_add_` + `pad_sequence`。
    3. `forced_align` 只支持 batch=1（版本限制，已实测），且要求 T >= L + 相邻重复数；
       对齐失败的样本退化成均匀切分，条数记进 stats 的 `align_fb` 供监控。
    4. att loss 默认**不**按 token 数归一（`att_length_normalized: false`）。本项目的 CTC loss
       是 `ctc.py:121` 的 `sum/B`，即**每条句子**的 -logP（实测收敛在 10 附近）；而按 token
       归一的 CE 是每 token 均值（0.x 量级），两者差约 L_avg≈22 倍，`ctc_weight` 就失去了
       字面意义。置 False 后 CE 也是每条句子的和，`ctc_weight` 才是真正的两支权重。
    5. 推理保留 V4 的 `psok/peok` 定位输出：decoder 第 i 个输出位置对应 CTC 第 i 个合并段，
       段起始帧就是该 token 的帧下标，于是 `kws_hit` / `kws_span` / `frame_span` 照旧可算。
    6. 参考实现 `inference` 把 `results.append` 写在 `else` 分支里，全 blank 的样本会**整条
       丢失** key（下游对不齐）；本实现空输出也返回一条空文本。
"""
import logging
import random

import torch
import torch.distributed as dist
import torchaudio

from funasr.losses.label_smoothing_loss import LabelSmoothingLoss
from funasr.metrics.compute_acc import th_accuracy
from funasr.models.tsasr.dae_kws_asr_model import KwsAsrConformerDae
from funasr.register import tables
from funasr.train_utils.device_funcs import force_gatherable
from funasr.utils import postprocess_utils
from funasr.utils.datadir_writer import DatadirWriter

_LOG_EPS = 1e-12


@tables.register("model_classes", "KwsAsrConformerDaeV2")
class KwsAsrConformerDaeV2(KwsAsrConformerDae):
    """V4-TSE 的 encoder/cue/CTC + Paraformer-V2 式非自回归 decoder。

    Args:
        decoder: `decoder_classes` 里的名字，默认 `ParaformerSANMDecoder_v2_community`
            （SANM 自注意力 + cross-att 到 encoder_out，`input_layer` 须为 `linear`）。
        decoder_conf: 传给 decoder 的其余参数；`vocab_size` 与 `encoder_output_size`
            由本类注入，不要在 yaml 里写。
        ctc_weight: `loss = ctc_weight * loss_ctc + (1 - ctc_weight) * loss_att`。
        lsm_weight: CE 的 label smoothing。
        att_length_normalized: CE 是否按 token 数归一。默认 False，理由见模块 docstring 第 4 条。
        其余参数与 `KwsAsrConformerDae` 完全一致，经 `**kwargs` 透传给父类。
    """

    def __init__(
        self,
        decoder: str = "ParaformerSANMDecoder_v2_community",
        decoder_conf: dict = None,
        ctc_weight: float = 0.3,
        lsm_weight: float = 0.1,
        att_length_normalized: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        decoder_class = tables.decoder_classes.get(decoder)
        if decoder_class is None:
            raise ValueError(
                f"Unknown decoder '{decoder}'. "
                f"registered: {list(tables.decoder_classes.keys())}"
            )
        self.decoder = decoder_class(
            vocab_size=self.vocab_size,
            encoder_output_size=self.encoder.output_size(),
            **(decoder_conf or {}),
        )
        self.criterion_att = LabelSmoothingLoss(
            size=self.vocab_size,
            padding_idx=self.ignore_id,
            smoothing=lsm_weight,
            normalize_length=att_length_normalized,
        )
        self.ctc_weight = float(ctc_weight)

    # =================================================== CTC -> token 级压缩
    @staticmethod
    def _align_to_target_index(align: torch.Tensor) -> torch.Tensor:
        """把 CTC 强制对齐路径映射成「每帧属于第几个目标位置」，blank 帧为 -1。

        段边界判据：当前帧非 blank 且与上一帧不同（上一帧是 blank 或是别的 token）。
        CTC 拓扑保证相邻的同一 token 之间必有 blank，所以连续相同的非 blank 帧一定属于
        同一个目标位置。例：标签 [A, B]、路径 [A, A, _, B, B] -> [0, 0, -1, 1, 1]。

        Args:
            align: (T,) 对齐路径，元素是**列 id**（0 = blank）。

        Returns:
            (T,) int64，值域 [-1, U-1]。
        """
        is_tok = align != 0
        prev = torch.roll(align, 1)
        prev[0] = 0  # 首帧强制与「上一帧」不同
        seg = (is_tok & (align != prev)).long().cumsum(0) - 1
        return torch.where(is_tok, seg, torch.full_like(seg, -1))

    def _compress_by_forced_align(self, probs, enc_lens, ys_pad, ys_lens):
        """训练路径：CTC 后验 -> 强制对齐 -> 段内均值，长度恒等于 `ys_lens`。

        Args:
            probs: (B, T, V) CTC softmax 后验，**带梯度**（CE 的梯度要经它回到 encoder）。
            enc_lens: (B,) encoder 有效帧数。
            ys_pad: (B, L) 目标 id，pad 为 `ignore_id`(-1)。
            ys_lens: (B,) 目标长度。

        Returns:
            (comp, n_fallback)：comp 为 (B, L, V)，第 i 行是第 i 个目标位置对应帧的后验均值。
        """
        B, T, V = probs.shape
        dev = probs.device
        L = int(ys_lens.max())
        ys = ys_pad[:, :L].clamp_min(0).long()

        # 归约词表：列 0 = blank，列 j+1 承载 ys[:, j] 的后验
        gcol = torch.cat([ys.new_zeros(B, 1), ys], dim=1)                    # (B, L+1)
        red = probs.detach().gather(2, gcol[:, None, :].expand(B, T, L + 1))
        red = red.clamp_min(_LOG_EPS).log()

        # 相邻重复必须共用同一列，否则 forced_align 会当成两个不同 token 而允许跳过中间 blank
        pos = torch.arange(1, L + 1, device=dev).expand(B, L)
        same = torch.zeros_like(ys, dtype=torch.bool)
        same[:, 1:] = ys[:, 1:] == ys[:, :-1]
        col = torch.cummax(torch.where(same, torch.zeros_like(pos), pos), dim=1).values

        # 一次 D2H，之后全在 CPU 上循环（forced_align 只支持 batch=1）
        red_cpu = red.cpu()
        col_cpu = col.to(torch.int32).cpu()
        t_list = enc_lens.to(torch.int64).cpu().tolist()
        l_list = ys_lens.to(torch.int64).cpu().tolist()

        idx_pad = torch.full((B, T), -1, dtype=torch.long)
        n_fb = 0
        for b in range(B):
            tb, lb = min(int(t_list[b]), T), min(int(l_list[b]), L)
            if lb <= 0 or tb <= 0:
                continue
            path = None
            if tb >= lb:
                try:
                    align, _ = torchaudio.functional.forced_align(
                        red_cpu[b : b + 1, :tb, : lb + 1].contiguous(),
                        col_cpu[b : b + 1, :lb].contiguous(),
                        blank=0,
                    )
                    path = self._align_to_target_index(align[0].long())
                except Exception as e:  # T 不够放下所有重复、或后验里出现 nan
                    logging.warning(f"forced_align 失败(b={b}, T={tb}, L={lb}): {e}")
            if path is None:
                # 退化：均匀切分。保证每个目标位置至少拿到一帧，CE 不会看到全零输入。
                n_fb += 1
                path = torch.arange(tb, dtype=torch.long) * lb // tb
            idx_pad[b, :tb] = path
        idx_pad = idx_pad.to(dev)

        # 段内求均值：flat = b * L + 目标位置，一次 index_add 覆盖整个 batch
        valid = idx_pad >= 0
        flat = (torch.arange(B, device=dev)[:, None] * L + idx_pad).clamp_min(0)[valid]
        comp = probs.new_zeros(B * L, V).index_add(0, flat, probs[valid])
        cnt = probs.new_zeros(B * L, 1).index_add(
            0, flat, probs.new_ones(flat.numel(), 1)
        )
        return (comp / cnt.clamp_min(1.0)).view(B, L, V), n_fb

    @staticmethod
    def _compress_by_greedy(probs, path, blank_id):
        """推理路径：CTC greedy path 去 blank + 合并连续重复，段内取后验均值。

        Args:
            probs: (T, V) 单条样本的 CTC 后验。
            path: (T,) greedy argmax 路径。
            blank_id: blank 的 id。

        Returns:
            (comp, frames)：comp 为 (U', V)；frames 为长度 U' 的段**起始帧**下标列表，
            用来把 decoder 的输出位置映射回帧区间（`frame_span` 依赖它）。
        """
        if path.numel() == 0:
            return probs.new_zeros(0, probs.size(1)), []
        toks, counts = torch.unique_consecutive(path, return_counts=True)
        ends = torch.cumsum(counts, dim=0)
        starts = torch.cat([ends.new_zeros(1), ends[:-1]])
        rows, frames = [], []
        for i in range(toks.numel()):
            if int(toks[i]) == blank_id:
                continue
            s, e = int(starts[i]), int(ends[i])
            rows.append(probs[s:e].mean(dim=0))
            frames.append(s)
        if not rows:
            return probs.new_zeros(0, probs.size(1)), []
        return torch.stack(rows), frames

    # ============================================================== 训练前向
    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        cue_token: torch.Tensor = None,
        cue_token_lengths: torch.Tensor = None,
        **kwargs,
    ):
        """Encoder + CTC + 非自回归 decoder，返回 (loss, stats, weight)。

        stats 里三个数各有分工：
            `loss_ctc` / `cer_ctc`  与 V4-TSE 同定义，**跨实验可比**（4-4 的曲线可直接对读）；
            `loss_att`              decoder 的 CE；
            `acc`                  decoder 的 token 准确率，也是选 ckpt 的准则。注意它是在
                                   **强制对齐**（长度 = 真实标签长度）下算的，属于 oracle 长度，
                                   比真实 NAR 解码乐观，只能纵向比不能当最终指标；
            `align_fb`             本 batch 里强制对齐失败、退化成均匀切分的样本数，应恒为 0。

        Args / Returns 与 `KwsAsrConformerDae.forward` 一致。
        """
        if text_lengths.dim() > 1:
            text_lengths = text_lengths[:, 0]
        if speech_lengths.dim() > 1:
            speech_lengths = speech_lengths[:, 0]
        if cue_token_lengths is not None and cue_token_lengths.dim() > 1:
            cue_token_lengths = cue_token_lengths[:, 0]

        batch_size = speech.shape[0]
        encoder_out, encoder_out_lens, _ = self.encode(
            speech, speech_lengths, cue_token=cue_token, cue_token_lengths=cue_token_lengths
        )

        loss_ctc = self.ctc(encoder_out, encoder_out_lens, text, text_lengths)

        probs = self.ctc.softmax(encoder_out)
        comp, n_fb = self._compress_by_forced_align(
            probs, encoder_out_lens, text, text_lengths
        )
        decoder_out, _ = self.decoder(encoder_out, encoder_out_lens, comp, text_lengths)

        ys = text[:, : comp.size(1)]
        loss_att = self.criterion_att(decoder_out, ys)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size), ys, ignore_label=self.ignore_id
        )
        loss = self.ctc_weight * loss_ctc + (1.0 - self.ctc_weight) * loss_att

        stats = {
            "loss": torch.clone(loss.detach()),
            "loss_ctc": loss_ctc.detach(),
            "loss_att": loss_att.detach(),
            "acc": acc_att,
            "align_fb": float(n_fb),
        }
        cer_ctc = self._calc_cer_ctc(encoder_out, encoder_out_lens, text, text_lengths)
        if cer_ctc is not None:
            stats["cer_ctc"] = cer_ctc

        if self.length_normalized_loss:
            batch_size = int(text_lengths.sum())
        if self.training:
            self._maybe_log_sample_nar(
                encoder_out,
                encoder_out_lens,
                decoder_out,
                text,
                text_lengths,
                cue_token,
                cue_token_lengths,
                kwargs.get("mix_info"),
            )
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    @torch.no_grad()
    def _maybe_log_sample_nar(
        self,
        encoder_out,
        encoder_out_lens,
        decoder_out,
        text,
        text_lengths,
        cue_token,
        cue_token_lengths,
        mix_info=None,
    ):
        """每 `log_interval` 次前向、只在 rank0 抽一条打四行（+ 混音溯源）。

        比 V4-TSE 多一行 `nar`：decoder 在**强制对齐**输入下的 argmax。它与 `ref` 等长，
        所以逐字对读就能看出 decoder 到底在 CTC 的基础上改对了还是改错了；`hyp` 仍是 CTC
        greedy（长度自由），两者的差异正是 decoder 分支的作用。
        末尾的 `mix` 段与父类一致：本条与各干扰源的音频路径 + 转写（见
        `AudioDatasetKwsSpan._describe_mix`），用来判断 `hyp`/`nar` 里多出来的字是不是
        抄了干扰说话人的话。

        Args:
            encoder_out / encoder_out_lens: 编码器输出与有效帧长。
            decoder_out: (B, U, V) decoder logits。
            text / text_lengths: CTC 目标及长度。
            cue_token / cue_token_lengths: 线索 id 及长度，可为 None。
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
        nar = [int(x) for x in decoder_out[i, : int(text_lengths[i])].argmax(-1).tolist()]
        cue = []
        if cue_token is not None:
            L = int(cue_token_lengths[i]) if cue_token_lengths is not None else cue_token.size(1)
            cue = [int(x) for x in cue_token[i, :L].tolist()]
        logging.info(
            f"sample[{i}/{encoder_out.size(0)}] T={n}\n"
            f"  cue: {self._ids2text(cue)}\n"
            f"  ref: {self._ids2text(ref)}\n"
            f"  hyp: {self._ids2text(hyp)}\n"
            f"  nar: {self._ids2text(nar)}"
            f"{self._fmt_mix_info(mix_info, i)}"
        )

    # ================================================================== 推理
    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        """非自回归解码：CTC greedy 合并 -> decoder 一次前向 -> argmax。

        `kwargs["decode_mode"]`：
            `nar`（默认）走 decoder；`ctc` 完全退回父类的 CTC greedy，用来在同一个 ckpt 上
            做 A/B —— 两者共享 encoder，差异纯粹来自 decoder 分支。

        返回字段与父类完全一致（`key/token/text/kws_hit/kws_span/frame_span`，可选
        `cross_att`/`cue_token`），所以 `infer_eval_kws_asr_dae.py` 与 `eval_kws_detection.py`
        不用改。`frame_span` 的映射见 `_compress_by_greedy` 的 docstring。

        Args:
            data_in: 音频路径 / 波形 / fbank。
            data_lengths: `data_type=="fbank"` 时的长度。
            key: 每条样本的 key 列表。
            tokenizer: 用于 ids2tokens 与 cue_text 编码。
            frontend: WavFrontend。

        Returns:
            (results, meta_data)。
        """
        if kwargs.get("decode_mode", "nar") == "ctc":
            return super().inference(
                data_in, data_lengths, key=key, tokenizer=tokenizer, frontend=frontend, **kwargs
            )

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
        probs = self.ctc.softmax(encoder_out)
        paths = probs.argmax(dim=-1)

        return_att = bool(kwargs.get("return_att", False))
        att_last = None
        if return_att and extra is not None and extra.get("cross_att"):
            att_last = extra["cross_att"][-1].float().mean(dim=1).cpu().numpy()

        results = []
        b_total = encoder_out.size(0)
        if isinstance(key[0], (list, tuple)):
            key = key[0]
        if len(key) < b_total:
            key = key * b_total
        for i in range(b_total):
            n_frame = int(encoder_out_lens[i])
            comp, frame_of_token = self._compress_by_greedy(
                probs[i, :n_frame], paths[i, :n_frame], self.blank_id
            )
            if comp.size(0) == 0:
                # 全 blank（静音 / 纯噪声）。参考实现在这里会把整条样本丢掉，导致下游 key
                # 对不齐；这里照常返回一条空结果。
                token_int, frame_of_token = [], []
            else:
                in_lens = torch.tensor([comp.size(0)], device=comp.device, dtype=torch.int32)
                decoder_out, _ = self.decoder(
                    encoder_out[i : i + 1, :n_frame],
                    encoder_out_lens[i : i + 1],
                    comp.unsqueeze(0),
                    in_lens,
                )
                token_int = [int(x) for x in decoder_out[0].argmax(dim=-1).tolist()]

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
                item["cue_token"] = [int(x) for x in cue_token[i, :kw_len].tolist()]
            results.append(item)
            if ibest_writer is not None:
                ibest_writer["token"][key[i]] = " ".join(token)
                ibest_writer["text"][key[i]] = text_postprocessed

        return results, meta_data
