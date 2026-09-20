#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一的 CER/SER 评测口径，供 infer_eval / infer_eval_onnx / eval_infer_json 共用。

打分流程（ref 与 hyp 同口径）：
  1. text-norm:
       OpenCC 繁转简 → 句中英文 Whisper EnglishTextNormalizer
       （let's → let us 等）→ cn_tn.TextNorm（written → spoken，如 "26度"→"二十六度"）
  2. tokenize:  去标签/标点、小写；中文逐字、英文按词（MER，对外仍叫 CER）
  3. 编辑距离:  computer_cer.Calculator（compute-wer 算法）统计 N/C/S/D/I

约定：
  - ref 和 hyp 都做 TN
  - TN 报错时直接抛出中断（保证全部走 TN 口径）
  - TextNorm 开关：to_banjiao(全转半) + remove_space；不去语气词、不去儿化音
"""

import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from computer_cer import Calculator, width

# 使用本仓库随 FunASR 一并维护的 cn_tn。
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CN_TN_DIRS = (_REPO_ROOT / "funasr" / "models" / "fun_asr_nano" / "tools",)
for _cn_tn_dir in reversed(_CN_TN_DIRS):
    if _cn_tn_dir.is_dir() and str(_cn_tn_dir) not in sys.path:
        sys.path.insert(0, str(_cn_tn_dir))

# 连续英文片段（含缩写 apostrophe），供 Whisper EnglishTextNormalizer 单独处理，
# 避免整句中文被 EnglishNumberNormalizer / spelling map 改写。
_EN_SPAN = re.compile(
    r"[A-Za-z]+(?:['’][A-Za-z]+)?(?:[\s\-]+[A-Za-z]+(?:['’][A-Za-z]+)?)*"
)

# CJKT 字符区间，与 infer_eval.py / add_space_between_chars_asr.py 保持一致
_CJK_PATTERN = re.compile(
    r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF\uFE30-\uFE4F"
    r"\uFF65-\uFFDC\U00020000-\U0002FFFF\u3000-\u303F\uff01-\uff60\u0E00-\u0E7F\u4e00-\u9fff])"
)


# ---------------------------------------------------------------------------
# Text Normalization
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_opencc():
    """懒加载 OpenCC t2s（繁体 → 简体）。"""
    from opencc import OpenCC

    try:
        return OpenCC("t2s")
    except Exception:
        return OpenCC("t2s.json")


@lru_cache(maxsize=1)
def _get_whisper_english_normalizer():
    """加载 Whisper EnglishTextNormalizer（transformers，与 script/local/Whisper_norm.py 同套规则）。"""
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    mapping_path = Path(__file__).resolve().parent.parent / "english.json"
    with open(mapping_path, encoding="utf-8") as f:
        mapping = json.load(f)
    return EnglishTextNormalizer(mapping)


def _normalize_english_spans(text: str) -> str:
    """对句中连续英文片段做 Whisper 英文归一化，其余原文保留。"""
    english_normalizer = _get_whisper_english_normalizer()
    pieces = []
    last = 0
    for match in _EN_SPAN.finditer(text):
        pieces.append(text[last : match.start()])
        normalized = english_normalizer(match.group()).strip()
        pieces.append(f" {normalized} " if normalized else " ")
        last = match.end()
    pieces.append(text[last:])
    return re.sub(r"\s+", " ", "".join(pieces)).strip()


@lru_cache(maxsize=1)
def _get_normalizer():
    """懒加载 cn_tn.TextNorm（纯 Python 正则，构建极快，进程内复用）。

    开关：全角转半角 + 去空白；不去语气词、不去儿化音。
    normalize_nsw（数字/日期/金额/分数/百分数等转中文）与去标点为内置必做。
    """
    from cn_tn import TextNorm

    kwargs = dict(
        to_banjiao=True,
        to_upper=False,
        to_lower=False,
        remove_fillers=False,
        remove_erhua=False,
        check_chars=False,
        remove_space=True,
    )
    # 旧版 chinese_text_normalization 有 cc_mode；仓库内兼容实现没有该参数。
    try:
        return TextNorm(**kwargs, cc_mode="")
    except TypeError:
        return TextNorm(**kwargs)


def text_norm(s: str) -> str:
    """对单条文本做 TN。空串直接返回；报错向上抛出（中断）。

    顺序：OpenCC 繁转简 → 句中英文 Whisper 归一化 → 中文 cn_tn.TextNorm。
    Whisper 必须在 cn_tn 之前：后者会把 apostrophe 换成空格，let's 会碎成 let s。
    """
    if s is None:
        return ""
    s = str(s).strip()
    if not s:
        return ""
    s = _get_opencc().convert(s)
    s = _normalize_english_spans(s)
    if not s:
        return ""
    return _get_normalizer()(s)


# ---------------------------------------------------------------------------
# Tokenization（MER：中文逐字、英文按词；对外仍记为 CER）
# ---------------------------------------------------------------------------
def normalize_text(s: str) -> str:
    """去标签/标点，只保留中文字符、英文字母、数字（保留空格用于英文分词）。"""
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"<[^>]*>", "", s)
    s = s.replace("|", "")
    s = re.sub(r"\[[^\]]*\]", "", s)
    s = re.sub(r"[^\u4e00-\u9fffa-zA-Z0-9 ]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def add_space_between_chars(text: str) -> str:
    """对 CJK 字符逐字加空格，英文单词保持整体。"""
    chars = _CJK_PATTERN.split(text)
    chars = [ch for ch in chars if ch.strip()]
    text = " ".join(chars)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_for_scoring(s: str) -> List[str]:
    """评测专用分词：中文按字符、英文按单词为计算单元（不含 TN）。"""
    s = normalize_text(s)
    s = s.lower()
    s = add_space_between_chars(s)
    return [tok for tok in s.split(" ") if tok]


def score_tokens(s: str) -> List[str]:
    """完整打分单元：先 TN，再 tokenize_for_scoring。"""
    return tokenize_for_scoring(text_norm(s))


# ---------------------------------------------------------------------------
# CER via computer_cer.Calculator (compute-wer 算法)
# ---------------------------------------------------------------------------
def calc_cer(ref_tokens: List[str], hyp_tokens: List[str]) -> Tuple[float, int, int, int]:
    """对已切分的 token 列表计算 (cer, ins, dels, subs)。

    computer_cer.Calculator.calculate 会原地 insert(0,'') 修改入参，故传入副本。
    cer = (ins + dels + subs) / max(1, N)，N 为参考 token 数（result['all']）。
    """
    calc = Calculator()
    res = calc.calculate(list(ref_tokens), list(hyp_tokens))
    ins, dels, subs = res["ins"], res["sub"], res["del"]
    n = res["all"]
    cer = (ins + dels + subs) / max(1, n)
    return cer, ins, dels, subs


def align_tokens(ref_tokens: List[str], hyp_tokens: List[str]) -> Dict[str, Any]:
    """返回 computer_cer.Calculator 的完整对齐结果（含 lab/rec 对齐列表、N/C/S/D/I）。

    用于生成 wenet (compute-wer) 风格的逐句对齐文本。传入副本避免原地修改。
    """
    calc = Calculator()
    return calc.calculate(list(ref_tokens), list(hyp_tokens))


def format_wer_block(utt: str, align: Dict[str, Any], source: str = "") -> str:
    """把单条对齐结果格式化成 wenet/compute-wer 风格文本块：

        utt: <key>
        src: <真实音频路径>
        WER: 20.00 % N=5 C=5 S=0 D=0 I=1
        lab: 海 外    行 って
        rec: 海 外 に 行 って

    lab/rec 按 east-asian 宽度对齐（复用 computer_cer.width），与 compute-wer.py 一致。
    """
    n = align["all"]
    if n != 0:
        wer = float(align["ins"] + align["sub"] + align["del"]) * 100.0 / n
    else:
        wer = 0.0

    lines = [f"utt: {utt}"]
    if source:
        lines.append(f"src: {source}")
    lines.append(
        "WER: %4.2f %% N=%d C=%d S=%d D=%d I=%d"
        % (wer, n, align["cor"], align["sub"], align["del"], align["ins"])
    )

    lab_pad, rec_pad = [], []
    for idx in range(len(align["lab"])):
        len_lab = width(align["lab"][idx])
        len_rec = width(align["rec"][idx])
        length = max(len_lab, len_rec)
        lab_pad.append(length - len_lab)
        rec_pad.append(length - len_rec)

    lab_line = "lab: " + " ".join(
        align["lab"][idx] + " " * lab_pad[idx] for idx in range(len(align["lab"]))
    )
    rec_line = "rec: " + " ".join(
        align["rec"][idx] + " " * rec_pad[idx] for idx in range(len(align["rec"]))
    )
    lines.append(lab_line)
    lines.append(rec_line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluate a list of results (统一入口)
# ---------------------------------------------------------------------------
def evaluate(
    results: List[Dict[str, Any]],
    wer_path: Optional[str] = None,
    print_wer: bool = False,
) -> Dict[str, Any]:
    """计算 CER / SER 等指标（CER 字段实为 MER：中文逐字、英文按词）。

    每条 result 需含 ref_raw / hyp_raw 字段。
    - 若提供 wer_path，则把【只含错误句(CER>0)、按 CER 从高到低排序】的逐句
      wenet/compute-wer 风格对齐文本写入该文件，末尾追加 Overall CER / Overall SER 汇总。
    - 若 print_wer=True，则把同样内容打印到终端。
    """
    total = 0
    sum_dist = 0
    sum_ref_len = 0
    ins_sum = dels_sum = subs_sum = 0
    cor_sum = 0
    err_count = 0
    collect_wer = wer_path is not None or print_wer
    err_blocks: List[Tuple[float, str]] = []  # (cer, block) 仅错误句

    for r in results:
        ref_tokens = score_tokens(r["ref_raw"])
        hyp_tokens = score_tokens(r["hyp_raw"])
        align = align_tokens(ref_tokens, hyp_tokens)
        ins, dels, subs = align["ins"], align["del"], align["sub"]
        n = align["all"]
        cer = (ins + dels + subs) / max(1, n)

        r["ref_norm"] = " ".join(ref_tokens)
        r["hyp_norm"] = " ".join(hyp_tokens)
        r["cer"] = cer
        r["ins"] = ins
        r["dels"] = dels
        r["subs"] = subs
        r["ref_len"] = len(ref_tokens)

        total += 1
        dist = ins + dels + subs
        sum_dist += dist
        sum_ref_len += max(1, len(ref_tokens))
        ins_sum += ins
        dels_sum += dels
        subs_sum += subs
        cor_sum += align["cor"]
        if cer > 0:
            err_count += 1
            if collect_wer:
                err_blocks.append((
                    cer,
                    # TS-ASR jsonl follows Kaldi convention and preserves the
                    # source utterance ID as ``utt``.  Older callers only have
                    # FunASR's ``key``, so retain it as a backward-compatible
                    # fallback.
                    format_wer_block(
                        str(r.get("utt", r.get("key", ""))), align,
                        str(r.get("source", "")),
                    ),
                ))

    CER = sum_dist / max(1, sum_ref_len)
    SER = err_count / max(1, total)

    if collect_wer:
        err_blocks.sort(key=lambda x: x[0], reverse=True)  # CER 从高到低
        n_all = cor_sum + subs_sum + dels_sum
        overall_wer = float(ins_sum + subs_sum + dels_sum) * 100.0 / max(1, n_all)
        body = "\n\n".join(b for _, b in err_blocks)
        if err_blocks:
            body += "\n"
        body += "\n" + "=" * 67 + "\n\n"
        body += "Overall CER -> %4.2f %% N=%d C=%d S=%d D=%d I=%d\n" % (
            overall_wer, n_all, cor_sum, subs_sum, dels_sum, ins_sum,
        )
        body += "Overall SER -> %4.2f %% (err_utts=%d / num_utts=%d)\n" % (
            SER * 100.0, err_count, total,
        )
        if wer_path is not None:
            with open(wer_path, "w", encoding="utf-8") as f:
                f.write(body)
        if print_wer:
            print(body)

    return {
        "num_utts": total,
        "CER": CER,
        "SER": SER,
        "ins": ins_sum,
        "dels": dels_sum,
        "subs": subs_sum,
    }


def save_metrics_md(all_metrics: Dict[str, Any], md_path: str) -> None:
    """将 {ds_name -> metrics_dict} 保存为 Markdown 表格。"""
    lines = []
    lines.append("# Eval Metrics\n")
    header = "| Dataset | Utts | CER | SER | Ins | Del | Sub |"
    sep = "|---|---:|---:|---:|---:|---:|---:|"
    lines.append(header)
    lines.append(sep)
    for ds_name, m in all_metrics.items():
        lines.append(
            f"| {ds_name} | {m['num_utts']} "
            f"| {m['CER']:.4%} | {m['SER']:.4%} "
            f"| {m['ins']} | {m['dels']} | {m['subs']} |"
        )
    lines.append("")
    text = "\n".join(lines)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(text)
