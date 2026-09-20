#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DualCueConformerCTC 离线评测 / 推理：四种线索条件用同一个 ckpt 跑。

`--cue` 选择这次解码给哪些线索：

    both    text-cue + enroll-cue（部署条件）
    text    只给唤醒词文本
    enroll  只给注册语音
    none    一个都不给（纯 ASR 对照）

同一个 checkpoint 在四种条件下的 CER 差值就是「每种线索各贡献了多少」，这是
单线索模型做不到的对比——它们缺线索就直接报错。

JSONL 每行需要 `source`（待识别音频）、`target`（参考转写）；按 `--cue` 还需要
`--cue_text_field`（默认 `kw_text`，唤醒词文本）与 `--enroll_field`（默认
`enroll_source`，注册语音路径）。

注册语音的特征用**训练时那套 ASR frontend**（`extract_fbank`），不是 CAM++ 的
`kaldi.fbank - mean`：`ConformerDualCueEncoder` 让注册语音过与混合语音**同一套**
下采样与投影，特征口径必须一致，否则注册那一路进的是模型没见过的分布。

示例：
  python infer_eval_dual_cue.py \
    --config exp/dual_cue_conformer_ctc/config.yaml \
    --init_param exp/dual_cue_conformer_ctc/model.pt.best \
    --test_sets dev:/data/dev_with_enroll.jsonl --output_dir exp/.../eval \
    --cue both --gpus 0,1 --batch_size 16
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from eval_metrics import evaluate, save_metrics_md  # noqa: E402

CUE_MODES = ("both", "text", "enroll", "none")


def log(message):
    print(f"[infer_eval_dual_cue {time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def make_progress(total):
    """Terminal progress bar without making tqdm a hard dependency."""
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc="Inference", unit="utt", dynamic_ncols=True)
    except ImportError:
        class _LogProgress:
            def __init__(self, total):
                self.total, self.current = total, 0

            def update(self, count):
                self.current += count
                if self.current == self.total or self.current % 1000 < count:
                    log(f"progress: {self.current}/{self.total} utt")

            def close(self):
                pass
        return _LogProgress(total)


def read_jsonl(path, dataset, cue_text_field, enroll_field, need_text, need_enroll):
    """Read inference records, requiring only the fields this cue mode uses."""
    rows = []
    with open(path, encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            utt = obj.get("utt", obj.get("key", f"{dataset}:{line_no}"))
            cue_text = str(obj.get(cue_text_field, "") or "")
            enroll_source = obj.get(enroll_field, obj.get("ref_source"))
            if need_text and not cue_text:
                raise ValueError(
                    f"{path}:{line_no} misses {cue_text_field!r}; --cue includes the text cue"
                )
            if need_enroll and not enroll_source:
                raise ValueError(
                    f"{path}:{line_no} misses {enroll_field!r}; --cue includes the enrollment cue"
                )
            rows.append({
                "key": obj.get("key", utt),
                "utt": utt,
                "source": obj["source"],
                "cue_text": cue_text,
                "enroll_source": enroll_source,
                "target": obj.get("target", ""),
                "dataset": dataset,
            })
    return rows


def load_state_dict(path):
    """Load a regular checkpoint or a DeepSpeed ZeRO-1 checkpoint directory."""
    from unified_ts_asr.runtime import load_checkpoint_state

    return load_checkpoint_state(path)


def build(config_path):
    """Build the exact model, tokenizer and frontend used for training."""
    import importlib

    from omegaconf import OmegaConf
    from funasr.register import tables

    # `import funasr` scans submodules best-effort and swallows import errors, so a
    # missing dependency would surface here as `NoneType is not callable`.  Import
    # the two modules this path needs explicitly.
    importlib.import_module("funasr.models.tsasr.dae_kws_encoders")
    importlib.import_module("funasr.models.tsasr.dual_cue_conformer_ctc")

    from unified_ts_asr.runtime import resolve_config_paths

    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    cfg = resolve_config_paths(dict(cfg), config_path)
    with open(cfg["tokenizer_conf"]["token_list"], encoding="utf-8") as source:
        cfg["vocab_size"] = len(json.load(source))
    tokenizer = tables.tokenizer_classes.get(cfg["tokenizer"])(**cfg["tokenizer_conf"])
    frontend_conf = dict(cfg["frontend_conf"])
    frontend_conf["cmvn_file"] = cfg.get("cmvn_file")
    frontend = tables.frontend_classes.get(cfg["frontend"])(**frontend_conf)
    model_kwargs = dict(cfg)
    model_conf = dict(model_kwargs.pop("model_conf") or {})
    model_conf.update(model_kwargs)
    model_cls = tables.model_classes.get(cfg["model"])
    if model_cls is None:
        raise KeyError(
            f"Model {cfg['model']!r} is not registered after importing "
            "funasr.models.tsasr.dual_cue_conformer_ctc; check that --config is the "
            "DualCueConformerCTC training config."
        )
    return model_cls(**model_conf), tokenizer, frontend


def load_enroll_waveform(path, max_seconds=0.0):
    """Load one enrollment waveform as a 1-D float32 tensor at 16 kHz."""
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    wav, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = torch.from_numpy(np.asarray(wav, dtype=np.float32)).reshape(-1)
    if sample_rate != 16000:
        wav = torchaudio.functional.resample(wav.unsqueeze(0), sample_rate, 16000)[0]
    if max_seconds > 0:
        wav = wav[: int(max_seconds * 16000)]
    if wav.numel() < 400:
        raise ValueError(f"enrollment is too short ({wav.numel()} samples): {path}")
    return wav


def make_ref_batch(items, frontend, max_seconds):
    """Padded enrollment fbank batch, computed with the training ASR frontend.

    Returns:
        `(ref (B, T_ref, 80), ref_lengths (B, 1), keep)` where `keep` holds the
        indices of the items whose enrollment could be read.
    """
    import torch
    from funasr.utils.load_utils import extract_fbank

    waveforms, keep = [], []
    for index, item in enumerate(items):
        try:
            waveforms.append(load_enroll_waveform(item["enroll_source"], max_seconds))
            keep.append(index)
        except Exception as exc:
            log(f"skip key={item['key']}: cannot load enroll {item['enroll_source']}: {exc}")
    if not waveforms:
        return None, None, keep
    ref, ref_lengths = extract_fbank(
        waveforms, data_type="sound", frontend=frontend, is_final=True
    )
    return ref, ref_lengths.reshape(-1, 1).to(torch.int32), keep


def infer_shard(gpu, config_path, init_param, items, batch_size, cue, enroll_max_seconds, queue):
    """Load one model on one GPU and return decoded rows through `queue`."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    try:
        import torch

        want_text = cue in ("both", "text")
        want_enroll = cue in ("both", "enroll")
        model, tokenizer, frontend = build(config_path)
        state_dict = load_state_dict(init_param)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            log(f"GPU{gpu} missing keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            log(f"GPU{gpu} unexpected keys ({len(unexpected)}): {unexpected[:5]}")
        model.to("cuda:0").eval()

        results = []
        profile = {"batches": 0, "utts": 0, "wall": 0.0, "load_data": 0.0, "extract_feat": 0.0}
        for start in range(0, len(items), batch_size):
            original = items[start: start + batch_size]
            batch, conditioning = original, {}
            if want_enroll:
                ref, ref_lengths, keep = make_ref_batch(
                    original, frontend, enroll_max_seconds
                )
                for index in set(range(len(original))) - set(keep):
                    results.append({**original[index], "hyp": "", "unk": 0,
                                    "no_kws": 0, "no_tgt_spk": 0,
                                    "error": "enroll_load_failed"})
                if ref is None:
                    queue.put(("progress", gpu, len(original)))
                    continue
                batch = [original[index] for index in keep]
                conditioning["ref_speech"] = ref
                conditioning["ref_speech_lengths"] = ref_lengths
            if want_text:
                # `inference` wraps each cue_text in psok/peok, matching `_wrap_cue`.
                conditioning["cue_text"] = [item["cue_text"] for item in batch]
            try:
                with torch.no_grad():
                    started = time.perf_counter()
                    outputs, batch_meta = model.inference(
                        data_in=[item["source"] for item in batch],
                        key=[item["key"] for item in batch],
                        tokenizer=tokenizer,
                        frontend=frontend,
                        device="cuda:0",
                        **conditioning,
                    )
                    profile["wall"] += time.perf_counter() - started
                    profile["load_data"] += float(batch_meta.get("load_data", 0.0))
                    profile["extract_feat"] += float(batch_meta.get("extract_feat", 0.0))
                    profile["batches"] += 1
                    profile["utts"] += len(batch)
            except Exception as exc:
                log(f"GPU{gpu} batch at {start} failed: {exc}")
                outputs = [{} for _ in batch]
            for item, output in zip(batch, outputs):
                output = output if isinstance(output, dict) else {}
                results.append({
                    **item,
                    "hyp": output.get("text", "") or "",
                    "kws_hit": bool(output.get("kws_hit", False)),
                    # Both rejection labels are stripped model-side before `text` is
                    # built; counting them here keeps them out of the CER numerator
                    # while still allowing miss attribution (keyword vs speaker).
                    "no_kws": int(output.get("no_kws", 0)),
                    "no_tgt_spk": int(output.get("no_tgt_spk", 0)),
                    "unk": sum(token == "<unk>" for token in (output.get("token") or [])),
                })
            queue.put(("progress", gpu, len(original)))
        queue.put(("done", gpu, True, results, profile))
    except Exception:
        import traceback
        queue.put(("done", gpu, False, traceback.format_exc(), None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="training config.yaml")
    parser.add_argument("--init_param", required=True,
                        help="checkpoint file or DeepSpeed directory")
    parser.add_argument("--test_sets", nargs="+", required=True, help="name:jsonl or jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cue", default="both", choices=CUE_MODES,
                        help="which cues to supply; 'none' is the plain-ASR control")
    parser.add_argument("--cue_text_field", default="kw_text",
                        help="JSONL field holding the text cue")
    parser.add_argument("--enroll_field", default="enroll_source",
                        help="JSONL field holding the enrollment waveform path")
    parser.add_argument("--enroll_max_seconds", type=float, default=5.0,
                        help="0 means use the entire enrollment audio")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    want_text = args.cue in ("both", "text")
    want_enroll = args.cue in ("both", "enroll")

    output_dir = Path(args.output_dir)
    (output_dir / "eval").mkdir(parents=True, exist_ok=True)
    items, datasets = [], []
    for spec in args.test_sets:
        if ":" in spec and not os.path.exists(spec):
            name, path = spec.split(":", 1)
        else:
            name, path = Path(spec).stem, spec
        if not os.path.isfile(path):
            log(f"WARNING: dataset not found, skip: {path}")
            continue
        rows = read_jsonl(
            path, name, args.cue_text_field, args.enroll_field, want_text, want_enroll
        )
        if args.limit > 0:
            rows = rows[: args.limit]
        if rows:
            items.extend(rows)
            datasets.append(name)
    if not items:
        raise SystemExit("ERROR: no valid test samples")

    gpus = [int(gpu) for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise SystemExit("ERROR: --gpus is empty")
    shards = [[] for _ in gpus]
    for index, item in enumerate(items):
        shards[index % len(gpus)].append(item)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    processes = []
    for gpu, shard in zip(gpus, shards):
        if not shard:
            continue
        process = ctx.Process(target=infer_shard, args=(
            gpu, args.config, args.init_param, shard, args.batch_size, args.cue,
            args.enroll_max_seconds, queue,
        ))
        process.start()
        processes.append(process)
    collected, finished = {}, 0
    progress = make_progress(len(items))
    try:
        while finished < len(processes):
            message = queue.get()
            kind, gpu = message[:2]
            if kind == "progress":
                progress.update(message[2])
                continue
            _, _, ok, payload, profile = message
            finished += 1
            if not ok:
                raise RuntimeError(f"GPU {gpu} failed:\n{payload}")
            collected[gpu] = payload
            if profile and profile["batches"]:
                io_time = profile["load_data"] + profile["extract_feat"]
                compute_time = max(0.0, profile["wall"] - io_time)
                log(
                    f"GPU{gpu} profile: {profile['utts']} utt / {profile['wall']:.1f}s, "
                    f"load_wav={profile['load_data']:.1f}s, fbank={profile['extract_feat']:.1f}s, "
                    f"model+CTC={compute_time:.1f}s"
                )
    finally:
        progress.close()
    for process in processes:
        process.join()
        if process.exitcode:
            raise RuntimeError(f"worker pid={process.pid} exited with {process.exitcode}")

    by_key = {row["key"]: row for rows in collected.values() for row in rows}
    results = [by_key[item["key"]] for item in items if item["key"] in by_key]
    metrics = {}
    for dataset in datasets:
        rows = [row for row in results if row["dataset"] == dataset]
        with open(output_dir / f"{dataset}.jsonl", "w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps({
                    "utt": row["utt"], "key": row["key"], "source": row["source"],
                    "cue_text": row["cue_text"], "enroll_source": row["enroll_source"],
                    "target": row["target"], "hyp": row["hyp"], "unk": row["unk"],
                    "kws_hit": row.get("kws_hit", False),
                    "no_kws": row.get("no_kws", 0),
                    "no_tgt_spk": row.get("no_tgt_spk", 0),
                    **({"error": row["error"]} if "error" in row else {}),
                }, ensure_ascii=False) + "\n")
        metrics[dataset] = evaluate(
            [{
                "utt": row["utt"], "key": row["key"], "source": row["source"],
                "ref_raw": row["target"], "hyp_raw": row["hyp"],
            } for row in rows],
            wer_path=str(output_dir / "eval" / f"{dataset}.cer.txt"), print_wer=False,
        )
        metrics[dataset]["cue"] = args.cue
        # Reported, never thresholded here: on a target-present set a rejection is a
        # false reject, on a target-absent set it is the desired behaviour.
        metrics[dataset]["no_kws_utt_rate"] = (
            sum(1 for row in rows if row.get("no_kws", 0)) / max(len(rows), 1)
        )
        metrics[dataset]["no_tgt_spk_utt_rate"] = (
            sum(1 for row in rows if row.get("no_tgt_spk", 0)) / max(len(rows), 1)
        )
        if want_text:
            metrics[dataset]["kws_hit_rate"] = (
                sum(1 for row in rows if row.get("kws_hit")) / max(len(rows), 1)
            )
    with open(output_dir / "eval" / "metrics.json", "w", encoding="utf-8") as output:
        json.dump(metrics, output, ensure_ascii=False, indent=2)
    save_metrics_md(metrics, str(output_dir / "eval" / "metrics.md"))
    log(f"done: {len(results)} samples (cue={args.cue}) -> {output_dir}")


if __name__ == "__main__":
    main()
