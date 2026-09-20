#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验4（V4, DAE-TSE/KCE 完全复刻）的离线评测：CTC greedy 解码 + 统一 CER 口径。

与 `infer_eval_kce_ctc.py`（V3）的差别：
  - 模型是 `KwsAsrConformerDae`（从零训练的 9 层 256 维 Conformer + CTC），
    `inference()` 除 `text` 外还返回 `kws_hit` / `kws_span` / `frame_span`
    （`psok…peok` 括号对的位置），`text` 里已剔除括号，不污染 CER。
  - 新增**错线索**对照 `--wrong_cue`：给每条样本换成测试集里另一个唤醒词
    （必然不在音频里）。这是 V1/V2/V3 都没做过的对照，也是唯一能证明线索真在
    起作用（而不是被忽略）的实验。
  - `--kws_score` 打开时顺带算 attention DP 打分（复刻 `AttentionAnalyzer`），
    写进每条结果的 `att_score`，供 `eval_kws_detection.py` 扫阈值。
    **偏离**：原实现的 `_KW_BORDER_TRIM=(2,-2)` 是按英文音素序列（长得多）设的，
    4 字中文唤醒词（含括号 L=6）会被裁成 2 行，故 `--kw_trim` 默认 0；想只去掉
    `psok`/`peok` 两个 sink 行就传 `--kw_trim 1`。
  - 每条结果带两个计数：`no_kws` = 假设里拒识符 `<no_kws/>`(8406) 的个数，
    `unk` = `<unk>`(8403) 的个数。**必须分开记**，也必须单独记：`inference()` 的
    `text` 走了 `postprocess_utils.sentence_postprocess`，那里第 185 行把
    `<s>/</s>/<unk>/<OOV>` 全部丢掉，所以 `hyp` 字段里永远看不到 `<unk>`；而
    `<no_kws/>` 不在那份名单里，是模型侧 `_strip_reject` 在算 `text` 之前摘掉的
    （不摘就会全额计入 CER 的插入错误）。`use_filler_label_for_neg` 训出来的模型用
    `<no_kws/>` 表示「该拒识」，拒识判据读 `no_kws`；`unk` 现在只表示真正的 OOV，
    2026-08-22 之前两者共用 `<unk>`，老结果文件里的 `unk` 是二者之和。

典型用法（三个 arm，共用同一份 ckpt）：
    python3 infer_eval_kws_asr_dae.py --config <exp>/config.yaml \
        --init_param <exp>/model.pt.best --cue_text 小吉同学 --kws_score \
        --test_sets test_A:/path/a.jsonl ... --output_dir <exp>/eval/cue1 --gpus 0,1
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from eval_metrics import evaluate, save_metrics_md  # noqa: E402


def log(message):
    """打印带时间戳与脚本前缀的日志。

    Args:
        message: 日志内容。
    """
    print(f"[infer_eval_kws_asr_dae {time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_jsonl(path, dataset):
    """读取评测 jsonl（{key, source, target, keyword_text}），附上数据集名。

    Args:
        path: jsonl 文件路径。
        dataset: 数据集名，写入每条记录。

    Returns:
        记录列表。
    """
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            items.append({
                "key": obj["key"],
                "source": obj["source"],
                "target": obj.get("target", ""),
                "cue_text": obj.get("cue_text", ""),
                "dataset": dataset,
            })
    return items


def target_unit_length(target):
    """返回开放 CUE/训练 char tokenizer 共用的 target 长度口径（忽略空白）。"""
    return sum(not char.isspace() for char in target)


def load_state_dict(path):
    """加载权重：`path` 是 deepspeed ckpt 目录时取其中的完整 fp32 权重。

    Args:
        path: 普通 `.pt` 文件，或 deepspeed 保存的 ckpt 目录。

    Returns:
        state_dict（OrderedDict）。
    """
    from unified_ts_asr.runtime import load_checkpoint_state

    return load_checkpoint_state(path)


def build(config_path):
    """按训练 config 构建 `KwsAsrConformerDae` + tokenizer + frontend。

    Args:
        config_path: 训练用的 yaml（`exp/<run>/config.yaml` 或 `conf/*.yaml`）。

    Returns:
        (model, tokenizer, frontend)。
    """
    from omegaconf import OmegaConf

    from funasr.register import tables
    import funasr  # noqa: F401

    from unified_ts_asr.runtime import resolve_config_paths

    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    cfg = resolve_config_paths(dict(cfg), config_path)
    token_list = json.load(open(cfg["tokenizer_conf"]["token_list"], encoding="utf-8"))
    cfg["vocab_size"] = len(token_list)

    tokenizer = tables.tokenizer_classes.get(cfg["tokenizer"])(**cfg["tokenizer_conf"])
    fe_conf = dict(cfg["frontend_conf"])
    # V4 训练时不做 CMVN，顶层 cmvn_file 必须是 null；这里只在两者一致时才覆盖，
    # 避免评测悄悄做训练时没做的归一化。
    if cfg.get("cmvn_file") is not None:
        raise ValueError(
            f"config 顶层 cmvn_file={cfg['cmvn_file']}，但 V4 训练不做 CMVN，评测必须一致"
        )
    frontend = tables.frontend_classes.get(cfg["frontend"])(**fe_conf)

    kwargs = dict(cfg)
    model_conf = dict(kwargs.pop("model_conf") or {})
    model_conf.update(kwargs)
    model = tables.model_classes.get(cfg["model"])(**model_conf)
    return model, tokenizer, frontend


def att_dp_score(att, kw_len, trim):
    """复刻 `AttentionAnalyzer`：在 cross-attention 图上找最优单调路径并归一。

    Args:
        att: (T', L) numpy 数组，最后一层 cross-attention（多头已取均值）。
        kw_len: 线索有效长度 L。
        trim: 首尾各裁掉的线索行数（原实现是 2，见模块 docstring 的偏离说明）。

    Returns:
        (score, start, end)：score 为路径均值；`K < 2` 时退化为逐行时间最大值的均值，
        此时 start/end 为 None。
    """
    import numpy as np

    from funasr.models.tsasr.dae_extra import find_max_path

    mat = np.asarray(att, dtype=np.float64)[:, :kw_len].T   # (L, T')
    if trim > 0 and mat.shape[0] > 2 * trim:
        mat = mat[trim: mat.shape[0] - trim]
    if mat.shape[0] < 2 or mat.shape[1] < 2:
        # find_max_path 在 K<2 时会走空回溯直接崩（原实现的边界缺陷），退化打分
        return float(mat.max(axis=1).mean()) if mat.size else 0.0, None, None
    max_sum, (start, end) = find_max_path(mat)
    return float(max_sum) / mat.shape[0], int(start), int(end)


def infer_shard(gpu, config_path, init_param, items, batch_size, kws_score, kw_trim, queue):
    """单个 GPU 进程：每完成一个 batch 就经 queue 回传结果。

    Args:
        gpu: GPU 编号。
        config_path: 训练 config yaml。
        init_param: ckpt 路径（文件或 deepspeed 目录）。
        items: 待推理样本列表，每条的 `cue_text` 已由主进程决定（空串 = 不注入线索）。
        batch_size: 每批样本数。
        kws_score: 是否算 attention DP 打分。
        kw_trim: DP 打分裁掉的首尾线索行数。
        queue: 回传 ``("rows", gpu, rows)`` / ``("done", gpu, None)`` 的队列。
            主进程收到 rows 后立即追加写入 jsonl，避免把整个评测集攒在内存里。
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    try:
        import torch

        model, tokenizer, frontend = build(config_path)
        sd = load_state_dict(init_param)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            log(f"GPU{gpu} missing keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            log(f"GPU{gpu} unexpected keys ({len(unexpected)}): {unexpected[:5]}")
        model.to("cuda:0").eval()

        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            kwargs = {"device": "cuda:0"}
            cues = [item["cue_text"] for item in batch]
            if any(cues):
                kwargs["cue_text"] = cues
                if kws_score:
                    kwargs["return_att"] = True
            try:
                with torch.no_grad():
                    outputs, _ = model.inference(
                        data_in=[item["source"] for item in batch],
                        key=[item["key"] for item in batch],
                        tokenizer=tokenizer,
                        frontend=frontend,
                        **kwargs,
                    )
            except Exception as exc:
                log(f"GPU{gpu} batch at {start} failed: {exc}")
                outputs = [{} for _ in batch]
            rows = []
            for item, output in zip(batch, outputs):
                output = output if isinstance(output, dict) else {}
                row = {
                    "key": item["key"],
                    "source": item["source"],
                    "ref_raw": item["target"],
                    "hyp_raw": output.get("text", "") or "",
                    "dataset": item["dataset"],
                    "cue_text": item["cue_text"],
                    "kws_hit": bool(output.get("kws_hit", False)),
                    "frame_span": output.get("frame_span", None),
                    # 拒识判据。`no_kws` 是模型侧 `_strip_reject` 数好的（那里已把拒识符
                    # 从 `text` 里摘掉，否则会全额计入 CER 插入错误）；`unk` 仍单独记，
                    # 它现在只表示真正的 OOV，不再兼任拒识。
                    "no_kws": int(output.get("no_kws", 0)),
                    "unk": sum(1 for t in (output.get("token") or []) if t == "<unk>"),
                }
                att = output.get("cross_att", None)
                if att is not None and item["cue_text"]:
                    # 线索含 psok/peok，L 用模型回传的 cue_token，别拿 cue_text 重编码
                    kw_len = len(output.get("cue_token") or [])
                    score, s, e = att_dp_score(att, kw_len, kw_trim)
                    row["att_score"] = score
                    row["att_span"] = None if s is None else [s, e]
                rows.append(row)
            # 每个 batch 都立刻交给唯一的主进程 writer；即使后续推理中断，已完成
            # batch 的结果也已经安全落盘。
            queue.put(("rows", gpu, rows))
        queue.put(("done", gpu, None))
    except Exception:
        import traceback

        queue.put(("error", gpu, traceback.format_exc()))


def output_row(row):
    """将内部推理记录转换为对外持续写入的 jsonl 格式。"""
    out = {
        "key": row["key"],
        "source": row["source"],
        "target": row["ref_raw"],
        "hyp": row["hyp_raw"],
        "cue_text": row["cue_text"],
        "kws_hit": row["kws_hit"],
        "frame_span": row["frame_span"],
        "no_kws": row["no_kws"],
        "unk": row["unk"],
    }
    if "att_score" in row:
        out["att_score"] = row["att_score"]
        out["att_span"] = row["att_span"]
    return out


def read_output_rows(path, dataset):
    """读回流式写出的结果，转换为 ``evaluate`` 所需的内部字段。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            row = {
                "key": obj["key"],
                "source": obj["source"],
                "ref_raw": obj.get("target", ""),
                "hyp_raw": obj.get("hyp", ""),
                "dataset": dataset,
                "cue_text": obj.get("cue_text", ""),
                "kws_hit": bool(obj.get("kws_hit", False)),
                "frame_span": obj.get("frame_span"),
                # 旧结果文件没有 no_kws 字段，缺省 0 -> reject_utt_rate 退化成 0，
                # 不会把老实验的数字算错，只是那一列没有意义。
                "no_kws": int(obj.get("no_kws", 0)),
                "unk": int(obj.get("unk", 0)),
            }
            if "att_score" in obj:
                row["att_score"] = obj["att_score"]
                row["att_span"] = obj.get("att_span")
            rows.append(row)
    return rows


def dataset_name_from_spec(spec):
    """从 ``name:path`` 或纯路径 test-set 参数得到输出文件名。"""
    if ":" in spec and not os.path.exists(spec):
        return spec.split(":", 1)[0]
    return Path(spec).stem


def summarize_outputs(output_dir, datasets):
    """从已写出的 jsonl 生成 CER 与汇总指标，不重新执行模型推理。"""
    metrics = {}
    results = []
    for dataset in datasets:
        rows = read_output_rows(output_dir / f"{dataset}.jsonl", dataset)
        results.extend(rows)
        metrics[dataset] = evaluate(
            rows, wer_path=str(output_dir / "eval" / f"{dataset}.cer.txt"), print_wer=False
        )
        hit = sum(1 for row in rows if row["kws_hit"])
        metrics[dataset]["kws_hit_rate"] = hit / max(len(rows), 1)
        # 拒识率 = 假设里出现过 `<no_kws/>` 的句子占比。正线索集上它是**误拒**，
        # 错线索集（--wrong_cue）上它才是想要的行为，两个 arm 的同名指标含义相反。
        reject = sum(1 for row in rows if row["no_kws"])
        metrics[dataset]["no_kws_utt_rate"] = reject / max(len(rows), 1)
        unk = sum(1 for row in rows if row["unk"])
        metrics[dataset]["unk_utt_rate"] = unk / max(len(rows), 1)
        m = metrics[dataset]
        log(f"{dataset}: utts={m['num_utts']} CER={m['CER']:.4%} "
            f"ins={m['ins']} del={m['dels']} sub={m['subs']} "
            f"bracket_hit={m['kws_hit_rate']:.2%} "
            f"no_kws_utt={m['no_kws_utt_rate']:.2%} unk_utt={m['unk_utt_rate']:.2%}")

    overall = evaluate(results)
    overall["kws_hit_rate"] = sum(1 for row in results if row["kws_hit"]) / max(len(results), 1)
    overall["no_kws_utt_rate"] = sum(1 for row in results if row["no_kws"]) / max(len(results), 1)
    overall["unk_utt_rate"] = sum(1 for row in results if row["unk"]) / max(len(results), 1)
    metrics["__overall__"] = overall
    log(f"OVERALL: utts={overall['num_utts']} CER={overall['CER']:.4%} "
        f"ins={overall['ins']} del={overall['dels']} sub={overall['subs']} "
        f"bracket_hit={overall['kws_hit_rate']:.2%} "
        f"no_kws_utt={overall['no_kws_utt_rate']:.2%} unk_utt={overall['unk_utt_rate']:.2%}")
    with open(output_dir / "eval" / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    save_metrics_md(metrics, str(output_dir / "eval" / "metrics.md"))
    log(f"metrics written to {output_dir / 'eval'}")


def _random_target_cue(target, cue_len, seed, key):
    """从一条 target 中确定性地抽取连续的开放唤醒词。

    评测 jsonl 的中文转写可能含有为可读性加入的空白；cue encoder 使用 char
    tokenizer，故这里去掉空白后按字符截取。每条样本单独以 ``seed:key`` 初始化
    RNG，避免数据集顺序、GPU 数或分片方式改变时抽到不同的 cue。
    """
    units = [char for char in target if not char.isspace()]
    if not units:
        raise ValueError("target 为空，无法从中抽取开放唤醒词")
    # 与训练集 AudioDatasetKwsSpan._sample_span 完全同口径：
    # kw_len = min(random_len, n_unit)。短 target 不丢弃，直接整条作为 cue。
    actual_len = min(cue_len, len(units))
    rng = random.Random(f"{seed}:{key}")
    start = rng.randint(0, len(units) - actual_len)
    return "".join(units[start : start + actual_len])


def assign_cues(
    all_items,
    cue_text,
    mode,
    seed=0,
    random_from_target=False,
    cue_len=4,
):
    """按 arm 决定每条样本用什么线索。

    Args:
        all_items: 全部样本（原地写回 `cue_text`）。
        cue_text: jsonl 无 `keyword_text` 时的兜底线索文本。
        mode: `"true"` 真线索 / `"none"` 不注入 / `"wrong"` 换成另一个唤醒词。
        seed: 错线索抽样的随机种子。
        random_from_target: True 时忽略 jsonl 的 `keyword_text`，从每条 target
            随机截取连续字符作为开放唤醒词。
        cue_len: 本次评测固定使用的开放唤醒词字符长度。

    Returns:
        用到的线索集合（`mode="none"` 时为空集）。
    """
    if mode == "none":
        for item in all_items:
            item["cue_text"] = ""
        return set()

    if random_from_target:
        if not 2 <= cue_len <= 5:
            raise SystemExit(
                f"ERROR: cue 长度必须在 2--5 之间，当前为 {cue_len}。"
            )
        shortened = 0
        for item in all_items:
            n_unit = sum(not char.isspace() for char in item["target"])
            if n_unit == 0:
                raise SystemExit(f"ERROR: {item['key']} 的 target 为空，无法生成开放唤醒词")
            shortened += n_unit < cue_len
            item["cue_text"] = _random_target_cue(
                item["target"], cue_len, seed, item["key"]
            )
        if shortened:
            log(
                f"{shortened} 条 target 短于请求的 cue_len={cue_len}；"
                "按训练口径使用完整 target 作为较短 CUE。"
            )
    else:
        for item in all_items:
            item["cue_text"] = item["cue_text"] or cue_text
    if mode == "true":
        return {item["cue_text"] for item in all_items}

    pool = sorted({item["cue_text"] for item in all_items if item["cue_text"]})
    if len(pool) < 2:
        raise SystemExit(
            f"ERROR: --wrong_cue 需要测试集里至少 2 种唤醒词，实际只有 {pool}。"
            f"可用 --cue_text 显式指定一个不存在的唤醒词代替。"
        )
    rng = random.Random(seed)
    for item in all_items:
        others = [w for w in pool if w != item["cue_text"]]
        item["cue_text"] = rng.choice(others)
    return set(pool)


def main():
    """命令行入口：解析参数、分片推理、按数据集汇总 CER 与括号命中率。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="训练 config yaml")
    parser.add_argument("--init_param", required=True, help="ckpt 文件或 deepspeed 目录")
    parser.add_argument("--test_sets", nargs="+", required=True, help="name:jsonl 或 jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--cue_text", default="小吉同学",
                        help="jsonl 无 keyword_text 时使用的线索文本")
    parser.add_argument("--random_cue_from_target", action="store_true",
                        help="开放唤醒词：忽略 keyword_text，从每条 target 随机截取连续字符")
    parser.add_argument("--cue_len", type=int, default=4,
                        help="--random_cue_from_target 时本次评测固定使用的关键词长度（2--5）")
    parser.add_argument("--cue_seed", type=int, default=0,
                        help="开放/错线索抽样随机种子；相同 key 和 seed 可复现")
    parser.add_argument("--min_target_length", type=int, default=2,
                        help="评测前过滤 target 长度小于该值的样本；默认与训练一致")
    parser.add_argument("--score_only", action="store_true",
                        help="跳过模型推理，直接为 output_dir 中已流式写出的 jsonl 计算指标")
    parser.add_argument("--no_cue", action="store_true", help="不注入线索（消融对照）")
    parser.add_argument("--wrong_cue", action="store_true",
                        help="换成测试集里另一个唤醒词（负样本对照）")
    parser.add_argument("--kws_score", action="store_true",
                        help="顺带算 attention DP 打分，写进结果的 att_score")
    parser.add_argument("--kw_trim", type=int, default=0,
                        help="DP 打分首尾各裁掉的线索行数（1 = 只去掉 psok/peok 两个 sink 行）")
    parser.add_argument("--limit", type=int, default=0, help=">0 时每个测试集只取前 N 条")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    (output_dir / "eval").mkdir(parents=True, exist_ok=True)
    if args.score_only:
        datasets = [dataset_name_from_spec(spec) for spec in args.test_sets]
        missing = [name for name in datasets if not (output_dir / f"{name}.jsonl").is_file()]
        if missing:
            raise SystemExit(f"ERROR: --score_only 缺少结果文件: {missing}")
        log(f"score-only: 使用 {len(datasets)} 个已写出结果文件，不执行模型推理")
        summarize_outputs(output_dir, datasets)
        return

    all_items, datasets = [], []
    for spec in args.test_sets:
        name = dataset_name_from_spec(spec)
        path = spec.split(":", 1)[1] if ":" in spec and not os.path.exists(spec) else spec
        if not os.path.isfile(path):
            log(f"WARNING: dataset not found, skip: {path}")
            continue
        rows = read_jsonl(path, name)
        before = len(rows)
        rows = [
            row for row in rows
            if target_unit_length(row["target"]) >= args.min_target_length
        ]
        if len(rows) != before:
            log(
                f"{name}: 按 min_target_length={args.min_target_length} "
                f"过滤 {before - len(rows)}/{before} 条短 target 样本"
            )
        if args.limit > 0:
            rows = rows[: args.limit]
        if rows:
            all_items.extend(rows)
            datasets.append(name)
    if not all_items:
        raise SystemExit("ERROR: no valid test samples")

    mode = "none" if args.no_cue else ("wrong" if args.wrong_cue else "true")
    pool = assign_cues(
        all_items,
        args.cue_text,
        mode,
        seed=args.cue_seed,
        random_from_target=args.random_cue_from_target,
        cue_len=args.cue_len,
    )
    log(f"{len(datasets)} 个测试集，共 {len(all_items)} 条；cue_mode={mode}"
        + (f"；开放 cue 长度={args.cue_len}"
           if args.random_cue_from_target and mode != "none" else "")
        + (f"；唤醒词池={len(pool)} 个" if pool else ""))

    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    shards = [[] for _ in gpus]
    for i, item in enumerate(all_items):
        shards[i % len(gpus)].append(item)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    processes = []
    # 只有主进程写文件，避免多 GPU append 同一个 jsonl 时行交错。以 w 打开会清空
    # 同 RUN_TAG 的旧结果；随后每收到一个 batch 都 flush，结果可在推理尚未结束时查看。
    writers = {
        dataset: open(output_dir / f"{dataset}.jsonl", "w", encoding="utf-8")
        for dataset in datasets
    }
    for gpu, shard in zip(gpus, shards):
        if shard:
            process = ctx.Process(target=infer_shard, args=(
                gpu, args.config, args.init_param, shard, args.batch_size,
                args.kws_score, args.kw_trim, queue,
            ))
            process.start()
            processes.append(process)
    done = 0
    written = 0
    try:
        while done < len(processes):
            kind, gpu, payload = queue.get()
            if kind == "rows":
                touched = set()
                for row in payload:
                    writers[row["dataset"]].write(
                        json.dumps(output_row(row), ensure_ascii=False) + "\n"
                    )
                    touched.add(row["dataset"])
                for dataset in touched:
                    writers[dataset].flush()
                written += len(payload)
                log(f"streamed {written}/{len(all_items)} results (latest: GPU{gpu})")
            elif kind == "done":
                done += 1
                log(f"GPU{gpu} finished ({done}/{len(processes)})")
            elif kind == "error":
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                raise RuntimeError(f"GPU {gpu} failed:\n{payload}")
            else:
                raise RuntimeError(f"unknown worker message from GPU{gpu}: {kind}")
    finally:
        for writer in writers.values():
            writer.close()
    for process in processes:
        process.join()

    if written != len(all_items):
        raise RuntimeError(f"结果条数不完整：写出 {written}，预期 {len(all_items)}")
    summarize_outputs(output_dir, datasets)


if __name__ == "__main__":
    main()
