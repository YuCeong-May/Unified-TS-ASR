# -*- coding: utf-8 -*-
"""No-LFR streaming target-speaker Conformer-CTC.

唤醒段整段送 offline 编码器（与唤醒词文本做 cross-attn）锁定目标说话人并提取
speaker embedding；指令段逐 chunk 流式编码，每个 chunk 用该 embedding 做条件化，
不与文本做 cross-attn。两段编码输出拼接后过同一个 CTC，对齐目标是完整的
「唤醒词 + 指令词」。

流式路径复用 offline 前端的 4 倍下采样，因此 ``chunk_size=[left, emit, right]``
的单位是 **40 ms 帧**。默认 ``[0, 4, 1]`` 表示每步提交 4 帧（160 ms）、前瞻 1 帧
（40 ms）。调用方喂 10 ms fbank 的口径见 :meth:`chunk_input_window`。
"""
import logging
import random
import torch
import torch.distributed as dist
import torch.nn as nn
from funasr.models.ctc.ctc import CTC
from funasr.models.tsasr.dae_modules import (DepthWiseConv, FNNBlock, MultiHeadAtt,
    MultiHeadCrossAtt, combine_mask, make_mask)
from funasr.models.tsasr.spk_ce_head import SpeakerCEHead
from funasr.models.tsasr.dae_kws_encoders import ConformerKceEncoderDae
from funasr.register import tables
from funasr.train_utils.device_funcs import force_gatherable

_SUBSAMPLE_RATE = 4      # au_conv 两层 kernel3/stride2/pad0 => 时间轴 4 倍下采样
_SUBSAMPLE_CONTEXT = 7   # 输出帧 j 只依赖输入帧 [4j, 4j+6]，纯 FIR、无 padding

class _Conditioner(nn.Module):
    def __init__(self, dim, mode="film"):
        super().__init__()
        if mode not in ("film", "repeat_add"):
            raise ValueError("condition_type must be 'film' or 'repeat_add'")
        self.mode = mode
        self.proj = nn.Linear(dim, 2 * dim if mode == "film" else dim)
        self.gate = nn.Parameter(torch.zeros(2 * dim if mode == "film" else dim))
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    def forward(self, x, speaker):
        p = self.proj(speaker).unsqueeze(1) * torch.tanh(self.gate).view(1, 1, -1)
        if self.mode == "film":
            gamma, beta = p.chunk(2, dim=-1)
            return x * (1.0 + gamma) + beta
        return x + p

class _StreamingLayer(nn.Module):
    def __init__(self, dim, heads, ffn_dim, kernel, norm, mode, cross_heads=1):
        super().__init__()
        self.mac = FNNBlock(list(ffn_dim)); self.mac_norm = nn.LayerNorm(dim)
        self.att = MultiHeadAtt(heads, dim); self.att_norm = nn.LayerNorm(dim)
        self.cross = MultiHeadCrossAtt(cross_heads, dim, norm="LayerNorm")
        self.condition = _Conditioner(dim, mode)
        self.conv_norm = nn.LayerNorm(dim)
        self.conv = DepthWiseConv(dim, kernel, norm=norm, causal=True)
        self.conv_cache_len = kernel - 1
        self.ffn_norm = nn.LayerNorm(dim); self.ffn = FNNBlock(list(ffn_dim))
    def forward_chunk(self, x, cache, speaker, look_back, cue=None, cue_valid=None,
                      cache_frames=None, att_mask=None):
        history = x.new_zeros(x.size(0), 0, x.size(-1)) if cache is None else cache
        if look_back >= 0:
            history = history[:, -look_back:] if look_back else history[:, :0]
        kv = torch.cat([history, x], dim=1)
        # att_mask 形状 (B, T_q, T_k)，True=可见。训练时用它复刻流式推理的可见范围
        # （左侧 cache + 右侧前瞻），不传则退化成「全部可见」。
        valid = att_mask if att_mask is not None else torch.ones(
            kv.size(0), 1, kv.size(1), dtype=torch.bool, device=x.device)
        y = x + self.att(self.att_norm(x), self.att_norm(kv), self.att_norm(kv), valid)[0]
        y = y + 0.5 * self.mac(self.mac_norm(y))
        # cue is intentionally optional: command chunks pass None, because the
        # wake-word text has no semantic relation to later commands.
        if cue is not None:
            q_valid = torch.ones(y.size(0), y.size(1), dtype=torch.bool, device=y.device)
            cross_mask = ~combine_mask(q_valid, cue_valid, 1)
            y = y + self.cross(y, cue, cue, cross_mask)[0]
        y = self.condition(y, speaker)
        conv_hist = history[:, -self.conv_cache_len:] if self.conv_cache_len else history[:, :0]
        if conv_hist.size(1) < self.conv_cache_len:
            conv_hist = torch.cat([
                x.new_zeros(x.size(0), self.conv_cache_len - conv_hist.size(1), x.size(-1)),
                conv_hist,
            ], dim=1)
        y = y + self.conv(self.conv_norm(torch.cat([conv_hist, y], dim=1)))[:, -x.size(1):]
        y = y + self.ffn(self.ffn_norm(y))
        cache_y = y if cache_frames is None else y[:, :cache_frames]
        new_cache = torch.cat([history, cache_y], dim=1)
        if look_back >= 0:
            new_cache = new_cache[:, -look_back:] if look_back else new_cache[:, :0]
        return y, new_cache

@tables.register("model_classes", "StreamingTextCueTSConformerCTC")
class StreamingTextCueTSConformerCTC(nn.Module):
    """Streaming text-cue TS-ASR; chunk_size is [left, emit, right]."""
    def __init__(self, encoder=None, encoder_conf=None, cue_encoder=None, cue_encoder_conf=None,
                 ctc_conf=None, input_size=80, vocab_size=-1, ignore_id=-1, blank_id=0,
                 spk_ce_conf=None, spk_ce_weight=0.1, condition_type="film",
                 speaker_condition_type=None, chunk_size=(0, 16, 4),
                 encoder_chunk_look_back=4, decoder_chunk_look_back=0, ctc_commit_delay=1, **kwargs):
        super().__init__()
        if int(input_size) != 80:
            raise ValueError("Streaming model requires no-LFR 80-dim fbank")
        conf = dict(encoder_conf or {})
        self.dim = int(conf.get("au_hidden_dim", 256)); self.num_blocks = int(conf.get("num_blocks", 9))
        heads = int(conf.get("self_att_heads", 4)); ffn = conf.get("ffn_dim") or [self.dim, 4*self.dim, self.dim]
        kernel = int(conf.get("conv_kernel_size", 15)); norm = conf.get("conv_norm", "batch_norm")
        mode = speaker_condition_type or condition_type
        encoder_name = encoder or "ConformerKceEncoderDae"
        encoder_cls = tables.encoder_classes.get(encoder_name) or ConformerKceEncoderDae
        self.encoder = encoder_cls(input_size=80, **conf)
        # 流式路径复用 offline 编码器的前端（4 倍下采样 + 投影 + PE），共享参数而非复制：
        # 两条路径落在同一特征空间，唤醒/指令两个 loss 都能训练前端。
        for name in ("au_conv", "au_conv_trans", "speech_input_projection", "speech_pe_module"):
            if not hasattr(self.encoder, name):
                raise ValueError(f"encoder {encoder_name} lacks front-end module '{name}'")
        self.layers = nn.ModuleList([_StreamingLayer(self.dim, heads, ffn, kernel, norm, mode,
                                                      int(conf.get("cross_att_heads", 1)))
                                     for _ in range(self.num_blocks)])
        self._warm_start_streaming_layers()
        self.ctc = CTC(vocab_size, self.dim, **(ctc_conf or {})); self.blank_id = int(blank_id)
        self.vocab_size = int(vocab_size); self.ignore_id = int(ignore_id)
        self.cue_encoder = None
        if cue_encoder is not None:
            cls = tables.phoneme_encoder_classes.get(cue_encoder)
            if cls is None: raise ValueError("Unknown cue_encoder: " + str(cue_encoder))
            self.cue_encoder = cls(**(cue_encoder_conf or {}))
        self.spk_ce = None
        if spk_ce_conf:
            sc = dict(spk_ce_conf); sc.setdefault("size", self.dim); self.spk_ce = SpeakerCEHead(**sc)
        self.spk_ce_weight = float(spk_ce_weight)
        self.chunk_size = tuple(int(x) for x in chunk_size)
        if len(self.chunk_size) != 3 or self.chunk_size[1] <= 0 or self.chunk_size[2] < 0:
            raise ValueError("chunk_size must be [left, emit, right]")
        # 导出喂帧口径，避免推理脚本自己推算下采样算术。
        self.subsample_rate = _SUBSAMPLE_RATE
        self.subsample_context = _SUBSAMPLE_CONTEXT
        self.chunk_input_frames = _SUBSAMPLE_RATE * (self.chunk_size[1] + self.chunk_size[2] - 1) + _SUBSAMPLE_CONTEXT
        self.chunk_input_shift = _SUBSAMPLE_RATE * self.chunk_size[1]
        self.encoder_chunk_look_back = int(encoder_chunk_look_back)
        self.decoder_chunk_look_back = int(decoder_chunk_look_back); self.ctc_commit_delay = int(ctc_commit_delay)
        self._warned = False
        # 与离线模型（dae_kws_asr_model.py:_maybe_log_sample）同一个开关：复用
        # `train_conf.log_interval`，不额外加配置项，样本日志与 trainer 的 loss 行同频出现。
        # 置 0 关闭。`token_list` 由 AutoModel 注入（auto_model.py 里 kwargs["token_list"]），
        # 缺了就只跳过打印、不影响训练。
        self.token_list = kwargs.get("token_list", None)
        self.sample_log_interval = int((kwargs.get("train_conf") or {}).get("log_interval", 0) or 0)
        self._fwd_count = 0
    def _encode_cue(self, cue_token, cue_token_lengths):
        if self.cue_encoder is None or cue_token is None:
            return None, None
        if cue_token_lengths is None:
            cue_token_lengths = torch.full((cue_token.size(0),), cue_token.size(1),
                                            dtype=torch.long, device=cue_token.device)
        return self.cue_encoder(cue_token.long(), cue_token_lengths.view(-1))

    @torch.no_grad()
    def _warm_start_streaming_layers(self):
        """Copy compatible parameters from the text-cue Conformer blocks."""
        source = getattr(self.encoder, "speech_transformer", None)
        if source is None:
            return
        for dst, src in zip(self.layers, source):
            pairs = ((dst.mac, src.macaron_layer), (dst.mac_norm, src.macaron_norm),
                     (dst.att, src.self_att), (dst.att_norm, src.input_norm),
                     (dst.conv_norm, src.conv_norm), (dst.conv, src.conv_layer),
                     (dst.ffn_norm, src.fnn_norm), (dst.ffn, src.feed_forward))
            for target, original in pairs:
                target.load_state_dict(original.state_dict(), strict=False)

    def _offline(self, speech, lengths, cue_token=None, cue_token_lengths=None):
        cue, cue_valid = self._encode_cue(cue_token, cue_token_lengths)
        x, out_lens, extra = self.encoder(speech, lengths, cue_feat=cue, cue_valid=cue_valid,
                                          return_hidden=True)
        return x, out_lens, extra["hidden_states"]

    def _subsample(self, fbank):
        """10 ms fbank -> 40 ms/帧的编码器输入，与 `self.encoder` 共享前端参数。

        `au_conv` 是两层 kernel3/stride2/pad0 的纯 FIR 卷积（无 padding），输出帧 j
        只依赖输入帧 ``[4j, 4j+6]``。因此只要调用方按 :meth:`chunk_input_window`
        切窗，分 chunk 的结果与整段前向 **逐元素相等**，不需要跨 chunk 状态。
        """
        if fbank.size(1) < _SUBSAMPLE_CONTEXT:
            return fbank.new_zeros(fbank.size(0), 0, self.dim)
        x = self.encoder.au_conv(fbank.unsqueeze(1))
        b, c, t, d = x.size()
        x = self.encoder.au_conv_trans(x.transpose(1, 2).contiguous().view(b, t, c * d))
        return self.encoder.speech_input_projection(x)

    def _pe_at(self, x, offset=0):
        """叠加从 `offset` 开始的位置编码（`PositionalEncoding.forward` 总是从 0 开始）。"""
        pe = self.encoder.speech_pe_module
        pe.pe = pe.pe.to(x.device)
        pos = pe.position_encoding(offset + x.size(1), x.device)[offset:]
        return x * pe.xcale + pos

    def chunk_input_window(self, out_index):
        """产出第 `out_index` 帧起 ``emit + right`` 个 40 ms 帧所需的 fbank 区间 [start, stop)。

        推理侧循环：``out_index`` 从 0 起、每步 ``+= emit``；窗口之间按设计重叠
        ``right`` 帧对应的输入（右侧前瞻帧下一步要重算，只有 ``emit`` 帧进 cache）。
        """
        start = _SUBSAMPLE_RATE * int(out_index)
        return start, start + self.chunk_input_frames

    def _chunk_mask(self, lengths, max_len, device):
        """构造 (B, T_q, T_k) 自注意力掩码（True=可见），复刻流式的可见范围。

        query 帧 t 属于 chunk ``c = t // emit``，可见 key 落在
        ``[c*emit - look_back*emit, (c+1)*emit - 1 + right]``，再与 padding 掩码相交。
        对角线强制可见，避免 padding query 整行被屏蔽导致 softmax 出 NaN。
        """
        emit, right = self.chunk_size[1], self.chunk_size[2]
        look = self.encoder_chunk_look_back * emit
        pos = torch.arange(max_len, device=device)
        chunk = torch.div(pos, emit, rounding_mode="floor")
        start = (chunk * emit - look).clamp(min=0) if self.encoder_chunk_look_back >= 0 else torch.zeros_like(pos)
        end = (chunk + 1) * emit - 1 + right
        vis = (pos.unsqueeze(0) >= start.unsqueeze(1)) & (pos.unsqueeze(0) <= end.unsqueeze(1))
        vis = vis | torch.eye(max_len, dtype=torch.bool, device=device)
        valid = ~make_mask(lengths, max_len=max_len)  # True=有效
        return vis.unsqueeze(0) & valid.unsqueeze(1)

    def _stream_full(self, speech, lengths, speaker, cue_token=None, cue_token_lengths=None):
        """整段跑流式栈（训练用），可见范围按 `chunk_size` 掩码与推理对齐。"""
        cue, cue_valid = self._encode_cue(cue_token, cue_token_lengths)
        x = self._subsample(speech)
        olens = self.encoder.subsampled_lengths(lengths.view(-1)).clamp(min=0, max=x.size(1))
        x = self._pe_at(x, 0)
        valid = ~make_mask(olens, max_len=x.size(1))
        att = self._chunk_mask(olens, x.size(1), x.device)
        for layer in self.layers:
            x, _ = layer.forward_chunk(x, None, speaker, -1, cue, cue_valid, att_mask=att)
            x = x.masked_fill(~valid.unsqueeze(-1), 0.0)
        return x, olens

    @staticmethod
    def _concat_enc(wake, wake_lens, cmd, cmd_lens):
        """按有效帧拼接唤醒段与指令段编码（都是 40 ms/帧），供同一个 CTC 对齐完整文本。"""
        pieces = [torch.cat([wake[b, : int(wake_lens[b])], cmd[b, : int(cmd_lens[b])]], dim=0)
                  for b in range(wake.size(0))]
        out = nn.utils.rnn.pad_sequence(pieces, batch_first=True)
        return out, (wake_lens.view(-1) + cmd_lens.view(-1)).to(wake_lens.dtype)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load legacy text-cue checkpoints and mirror them into stream blocks."""
        result = super().load_state_dict(state_dict, strict=False, assign=assign)
        self._warm_start_streaming_layers()
        if strict and (result.missing_keys or result.unexpected_keys):
            raise RuntimeError("incompatible checkpoint: " + str(result))
        return result
    def forward(self, speech, speech_lengths, text, text_lengths, cue_token=None,
                cue_token_lengths=None, spk_id=None, kws_target=None, **kwargs):
        wake_speech = kwargs.get("wake_speech")
        wake_speech_lengths = kwargs.get("wake_speech_lengths")
        speaker_speech = wake_speech if wake_speech is not None else speech
        speaker_lengths = (wake_speech_lengths if wake_speech_lengths is not None else speech_lengths).view(-1)
        # 唤醒段：整段 offline 编码 + 唤醒词文本 cross-attn，锁定目标说话人。
        base_enc, base_lens, states = self._offline(
            speaker_speech, speaker_lengths, cue_token, cue_token_lengths
        )
        speaker = speech.new_zeros(speech.size(0), self.dim)
        stats = {}
        if self.spk_ce is not None:
            speaker = self.spk_ce.extract_embedding(states, base_lens, detach=False)
        # 指令段：流式栈 + 说话人条件化，显式关闭 cue（唤醒文本与后续语音语义无关）。
        cmd_enc, cmd_lens = self._stream_full(
            speech, speech_lengths.view(-1), speaker, None, None
        )
        if wake_speech is None:
            enc, lens = cmd_enc, cmd_lens
        else:
            # 两段拼接后过同一个 CTC，对齐目标是完整的「唤醒词 + 指令词」。
            enc, lens = self._concat_enc(base_enc, base_lens, cmd_enc, cmd_lens)
        loss = self.ctc(enc, lens, text, text_lengths.view(-1)); stats["loss_ctc"] = loss.detach()
        self._maybe_log_sample(enc, lens, text, text_lengths.view(-1), cue_token, cue_token_lengths,
                               None if wake_speech is None else base_lens, cmd_lens,
                               kwargs.get("mix_info"))
        if self.spk_ce is not None and spk_id is not None:
            sl, sa = self.spk_ce(states, base_lens, spk_id, valid=kws_target); loss = loss + self.spk_ce_weight * sl
            stats.update(loss_spk_ce=sl.detach(), acc_spk=sa.detach())
        stats["loss"] = loss.detach(); return force_gatherable((loss, stats, speech.size(0)), loss.device)

    @torch.no_grad()
    def _ids2text(self, ids):
        """把 token id 序列拼成可读字符串，越界 id 显示成 `<id>`。"""
        return "".join(self.token_list[int(x)] if 0 <= int(x) < len(self.token_list) else f"<{int(x)}>"
                       for x in ids)

    @torch.no_grad()
    def _maybe_log_sample(self, enc, lens, text, text_lengths, cue_token, cue_token_lengths,
                          wake_lens=None, cmd_lens=None, mix_info=None):
        """每 `train_conf.log_interval` 次前向随机抽一条打印，格式与离线模型一致。

        四行的分工同 `dae_kws_asr_model.py:_maybe_log_sample`：`cue` 是送进 cue encoder 的
        唤醒词线索（两端 psok/peok）、`ref` 是 CTC 目标（完整的「唤醒词 + 指令词」）、
        `hyp` 是同一条的贪心解码、`mix` 是混音溯源（哪条干扰音频 + 它的转写，用来判断
        `hyp` 里多出来的字是抄了干扰人还是幻觉）。

        比离线多两样，都是为了看流式那半边有没有在干活：表头给出
        ``T=总帧(唤醒 n + 指令 n)``（40 ms/帧），以及 `seg` 行把两段**分别**解码——
        唤醒段走 offline 栈、指令段走 chunk 掩码的流式栈，只看拼接后的 `hyp` 分不出
        是哪一段在错。分段解码与 `hyp` 在边界处可能差一个重复 token（贪心去重是按整条
        序列做的），不是 bug。

        只在 rank 0 打（四个 rank 各自的数据不同，全打就是 4 倍无关日志）。开销是
        一条样本的两三次 `ctc_lo` + argmax，no_grad，可忽略。
        """
        self._fwd_count += 1
        if self.sample_log_interval <= 0 or self._fwd_count % self.sample_log_interval != 0:
            return
        if not self.token_list or self.ctc.ctc_lo is None:
            return
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
        i = random.randrange(enc.size(0))
        n = int(lens[i])
        one = torch.tensor([n], device=enc.device)
        hyp = self._greedy_ctc(self.ctc.ctc_lo(enc[i : i + 1, :n]), one)[0]
        ref = [int(x) for x in text[i, : int(text_lengths[i])].tolist() if int(x) != self.ignore_id]
        cue = []
        if cue_token is not None:
            length = int(cue_token_lengths.view(-1)[i]) if cue_token_lengths is not None else cue_token.size(1)
            cue = [int(x) for x in cue_token[i, :length].tolist()]
        n_wake = 0 if wake_lens is None else int(wake_lens.view(-1)[i])
        n_cmd = n - n_wake if cmd_lens is None else int(cmd_lens.view(-1)[i])
        seg = ""
        if n_wake > 0:
            wake_hyp = self._greedy_ctc(self.ctc.ctc_lo(enc[i : i + 1, :n_wake]),
                                        torch.tensor([n_wake], device=enc.device))[0]
            cmd_hyp = self._greedy_ctc(self.ctc.ctc_lo(enc[i : i + 1, n_wake:n]),
                                       torch.tensor([n - n_wake], device=enc.device))[0]
            seg = f"\n  seg: 唤醒={self._ids2text(wake_hyp)} | 指令={self._ids2text(cmd_hyp)}"
        logging.info(
            f"sample[{i}/{enc.size(0)}] T={n}(唤醒 {n_wake} + 指令 {n_cmd}) ref_len={len(ref)}\n"
            f"  cue: {self._ids2text(cue)}\n"
            f"  ref: {self._ids2text(ref)}\n"
            f"  hyp: {self._ids2text(hyp)}"
            f"{seg}{self._fmt_mix_info(mix_info, i)}"
        )

    @staticmethod
    def _fmt_mix_info(mix_info, i):
        """取第 i 条的混音溯源串；这纯粹是日志，任何形态不符都不该影响训练，故 try 兜住。"""
        try:
            return "" if mix_info is None else f"\n  mix: {mix_info[i]}"
        except Exception:
            return ""

    def _greedy_ctc(self, logits, lengths):
        """整段贪心 CTC 解码（去 blank + 去重复），返回每条样本的 token list。"""
        path = logits.argmax(-1)
        out = []
        for b in range(path.size(0)):
            prev, item = self.blank_id, []
            for tok in path[b, : int(lengths[b])].tolist():
                if tok != self.blank_id and tok != prev:
                    item.append(int(tok))
                prev = tok
            out.append(item)
        return out

    @torch.no_grad()
    def init_target_speaker(self, wake_speech, wake_speech_lengths, **kwargs):
        if self.spk_ce is None: raise RuntimeError("init_target_speaker requires spk_ce_conf")
        enc, lens, states = self._offline(wake_speech, wake_speech_lengths.view(-1),
                                         kwargs.get("cue_token"), kwargs.get("cue_token_lengths"))
        # 唤醒段也参与 CTC，这里顺带解出唤醒段 token，供推理拼完整假设。
        wake_tokens = (self._greedy_ctc(self.ctc.ctc_lo(enc), lens)
                       if self.ctc.ctc_lo is not None else [[] for _ in range(enc.size(0))])
        return {"speaker_condition": self.spk_ce.extract_embedding(states, lens, detach=True),
                "wake_tokens": wake_tokens, "initialized": True}

    def init_cache(self, batch_size=1, device=None, dtype=None, speaker_condition=None):
        p = next(self.parameters()); device = device or p.device; dtype = dtype or p.dtype
        return {"layers": [torch.zeros(batch_size, 0, self.dim, device=device, dtype=dtype) for _ in self.layers],
                "speaker_condition": speaker_condition, "pe_offset": 0,
                "ctc": {"last_token": self.blank_id, "tokens": []}}

    def forward_chunk(self, speech_chunk, speech_lengths=None, cache=None, is_final=False, **kwargs):
        """喂一窗 fbank（区间由 :meth:`chunk_input_window` 给出），返回本步提交的 token。"""
        if cache is None: cache = self.init_cache(speech_chunk.size(0), speech_chunk.device, speech_chunk.dtype)
        speaker = cache.get("speaker_condition")
        if speaker is None:
            if not self._warned: logging.warning("No speaker condition; using zero condition"); self._warned = True
            speaker = speech_chunk.new_zeros(speech_chunk.size(0), self.dim)
        if speaker.dim() == 1: speaker = speaker.unsqueeze(0)
        if speaker.size(0) == 1 and speech_chunk.size(0) > 1: speaker = speaker.expand(speech_chunk.size(0), -1)
        n = speech_chunk.size(1) if speech_lengths is None else int(speech_lengths.max())
        x = self._subsample(speech_chunk[:, :n])
        if x.size(1) == 0:
            # 不足 7 帧（280 ms）出不了一帧 40 ms 输出，与整段前向丢尾的行为一致。
            if not cache["ctc"]["tokens"] or not isinstance(cache["ctc"]["tokens"][0], list):
                cache["ctc"]["tokens"] = [[] for _ in range(speech_chunk.size(0))]
            return {"logits": x.new_zeros(speech_chunk.size(0), 0, self.vocab_size),
                    "tokens": [[] for _ in range(speech_chunk.size(0))],
                    "all_tokens": [list(t) for t in cache["ctc"]["tokens"]],
                    "is_final": bool(is_final)}, cache
        x = self._pe_at(x, cache["pe_offset"])
        look = self.encoder_chunk_look_back * self.chunk_size[1]
        # 只提交 emit 帧：右侧前瞻帧下一步会连同新输入重算，因此不进 cache、不推进 PE。
        emit = x.size(1) if is_final else min(self.chunk_size[1], x.size(1))
        for i, layer in enumerate(self.layers):
            x, cache["layers"][i] = layer.forward_chunk(
                x, cache["layers"][i], speaker, look, cache_frames=emit
            )
        cache["pe_offset"] += emit
        logits = self.ctc.ctc_lo(x) if self.ctc.ctc_lo is not None else x
        logits = logits[:, :emit]
        path = logits.argmax(-1)
        if not isinstance(cache["ctc"]["last_token"], list):
            cache["ctc"]["last_token"] = [cache["ctc"]["last_token"]] * path.size(0)
        if not cache["ctc"]["tokens"] or not isinstance(cache["ctc"]["tokens"][0], list):
            cache["ctc"]["tokens"] = [[] for _ in range(path.size(0))]
        emitted = []
        for b in range(path.size(0)):
            item = []
            for tok in path[b].tolist():
                if tok != self.blank_id and tok != cache["ctc"]["last_token"][b]: item.append(int(tok))
                cache["ctc"]["last_token"][b] = int(tok)
            cache["ctc"]["tokens"][b].extend(item)
            emitted.append(item)
        return {"logits": logits, "tokens": emitted,
                "all_tokens": [list(x) for x in cache["ctc"]["tokens"]],
                "is_final": bool(is_final)}, cache
