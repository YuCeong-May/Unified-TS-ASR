#!/usr/bin/env python3
"""Speaker-only TS-ASR：离线提取完整注册语音的 CAM++ embedding。

输入是带 ``key``、``source``、``speaker`` 的 train/dev JSONL。每条完整 source
音频由本地文件系统读取，按训练相同口径提取：16 kHz -> 80-d Kaldi FBank -> 逐句
mean norm -> 冻结 CAM++。不依赖旧 V2 的 ref_start/ref_end 或 BOS 读取。

输入 / 输出
-----------
输入：train.jsonl / dev.jsonl（取 `key == emb_key` 的正样本行，按 key 去重）
输出：<out>/emb.<i>of<n>.npy   float32 (M, 192)
      <out>/keys.<i>of<n>.txt  与 npy 逐行对应的 key
      merge 阶段合成 <out>/spk_emb.f32（memmap）+ spk_emb.keys + spk_emb.mean.npy
      + spk_emb.meta.json

用法
    python scripts/extract_embeddings.py --jsonl train.jsonl dev.jsonl
        --out-dir <work>/spk_emb --released-model <model.pt>
        --shard 0 --num-shards 8
    python scripts/extract_embeddings.py --merge --out-dir <work>/spk_emb
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def build_encoder(model_path, device):
    """Build the frozen CAM++ encoder used by the released training recipe."""
    from funasr.models.tsasr.spk_encoder import CAMPPlusSpeakerEncoder

    encoder = CAMPPlusSpeakerEncoder(init_param=model_path, freeze=True)
    return encoder.to(device).eval()


def to_fbank(wav):
    """80 维 kaldi-fbank + 逐句 mean 归一（与 FunASR campplus extract_feature 一致）。"""
    import torch
    import torchaudio.compliance.kaldi as Kaldi

    feat = Kaldi.fbank(torch.as_tensor(wav).reshape(1, -1), num_mel_bins=80)
    return feat - feat.mean(dim=0, keepdim=True)


def load_jobs(jsonl_paths):
    """从带 speaker 的 train/dev JSONL 收集完整注册语音（按 key 去重）。"""
    jobs, seen = [], set()
    for path in jsonl_paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                key = r["key"]
                if key in seen:
                    continue
                seen.add(key)
                jobs.append({"key": key, "source": r["source"]})
    return jobs


def read_enroll(job):
    """读取本地完整 source 音频，返回 16 kHz float32 1D 注册语音。"""
    try:
        from funasr.utils.load_utils import load_audio_text_image_video

        wav = load_audio_text_image_video(job["source"], fs=16000)
    except Exception:
        return None
    if isinstance(wav, (tuple, list)):
        wav = wav[0]
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    return wav if wav.size >= 160 else None


def do_merge(out_dir):
    """把各 shard 的 npy/keys 合成一份 memmap + key 索引。"""
    npys = sorted(f for f in os.listdir(out_dir) if f.startswith("emb.") and f.endswith(".npy"))
    if not npys:
        sys.exit(f"[emb] {out_dir} 下没有 emb.*.npy")
    keys_all, arrays, dim = [], [], None
    for name in npys:
        arr = np.load(os.path.join(out_dir, name))
        kpath = os.path.join(out_dir, name.replace("emb.", "keys.").replace(".npy", ".txt"))
        with open(kpath, encoding="utf-8") as f:
            ks = [line.rstrip("\n") for line in f if line.strip()]
        if len(ks) != arr.shape[0]:
            sys.exit(f"[emb] {name} 与 keys 行数不一致: {arr.shape[0]} vs {len(ks)}")
        dim = arr.shape[1] if dim is None else dim
        if arr.shape[1] != dim:
            sys.exit(f"[emb] {name} 维度不一致: {arr.shape[1]} vs {dim}")
        arrays.append(arr)
        keys_all.extend(ks)
        print(f"[emb] {name}: {arr.shape[0]} 条", file=sys.stderr)

    # 同一个 key 只保留一份（多 shard 理论上不会重复，这里做兜底）
    seen, rows, keys = set(), [], []
    idx = 0
    for arr in arrays:
        for i in range(arr.shape[0]):
            k = keys_all[idx]
            idx += 1
            if k in seen:
                continue
            seen.add(k)
            rows.append(arr[i])
            keys.append(k)
    mat = np.stack(rows).astype(np.float32)

    mm_path = os.path.join(out_dir, "spk_emb.f32")
    mm = np.memmap(mm_path, dtype=np.float32, mode="w+", shape=mat.shape)
    mm[:] = mat
    mm.flush()
    del mm
    with open(os.path.join(out_dir, "spk_emb.keys"), "w", encoding="utf-8") as f:
        f.write("\n".join(keys) + "\n")
    # 全局均值：中心化 + L2 归一（说话人验证的标准前处理）用得到。
    # 实测这一步把「同说话人 base↔aug」vs「异说话人」的近似 EER 从 20.0% 降到 13.5%，
    # 因为全局均值的范数占了 embedding 范数的 39%，是不含身份信息的共同分量。
    mean_vec = mat.mean(axis=0)
    np.save(os.path.join(out_dir, "spk_emb.mean.npy"), mean_vec)
    with open(os.path.join(out_dir, "spk_emb.meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"num": mat.shape[0], "dim": int(mat.shape[1]), "dtype": "float32",
             "mean_file": "spk_emb.mean.npy"},
            f,
        )
    norms = np.linalg.norm(mat, axis=1)
    print(
        f"[emb] merge 完成: {mat.shape[0]} x {mat.shape[1]} -> {mm_path} "
        f"({mat.nbytes/1024**3:.2f} GB)；L2 范数 min={norms.min():.3f} "
        f"mean={norms.mean():.3f} max={norms.max():.3f}；全零向量 {(norms==0).sum()} 条；"
        f"全局均值范数 {np.linalg.norm(mean_vec):.1f}",
        file=sys.stderr,
    )


def main():
    """命令行入口：提取说话人 embedding 或执行 merge 阶段。"""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--jsonl", nargs="*", default=[], help="train.jsonl dev.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--merge", action="store_true", help="只做合并阶段")
    ap.add_argument(
        "--campplus-ckpt",
        required=False,
        help="standalone campplus_cn_common.bin",
    )
    ap.add_argument(
        "--released-model",
        default=None,
        help="released speaker-cue-ts-asr-embedding/model.pt containing CAM++",
    )
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--read-threads", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    global np
    import numpy as np

    os.makedirs(args.out_dir, exist_ok=True)
    if args.merge:
        do_merge(args.out_dir)
        return
    if not args.jsonl:
        sys.exit("需要 --jsonl 或 --merge")
    if bool(args.campplus_ckpt) == bool(args.released_model):
        sys.exit("提取阶段必须且只能指定 --campplus-ckpt 或 --released-model 之一")

    jobs = load_jobs(args.jsonl)
    print(f"[emb] 待提取 {len(jobs)} 条（已按 key 去重）", file=sys.stderr, flush=True)
    if args.limit > 0:
        jobs = jobs[: args.limit]
    jobs = [j for i, j in enumerate(jobs) if i % args.num_shards == args.shard]
    print(
        f"[emb] shard {args.shard}/{args.num_shards}: {len(jobs)} 条",
        file=sys.stderr,
        flush=True,
    )
    if not jobs:
        return

    import torch

    temporary = None
    if args.released_model:
        import tempfile

        from unified_ts_asr.runtime import (
            _write_embedded_campplus,
            load_checkpoint_state,
        )

        released_model = os.path.abspath(args.released_model)
        if not os.path.isfile(released_model):
            sys.exit(f"Released model not found: {released_model}")
        temporary = tempfile.TemporaryDirectory(prefix="unified-ts-asr-campplus-")
        model_path = _write_embedded_campplus(
            load_checkpoint_state(released_model), temporary.name
        )
    else:
        model_path = os.path.abspath(args.campplus_ckpt)
        if not os.path.isfile(model_path):
            sys.exit(f"CAM++ checkpoint not found: {model_path}")
    encoder = build_encoder(model_path, args.device)
    if temporary is not None:
        temporary.cleanup()
    needs_fbank = True
    dim = encoder.output_size()
    print(
        f"[emb] CAMPPlus 就绪，输出维度 {dim}，"
        f"输入=80维kaldi-fbank，权重={model_path}",
        file=sys.stderr,
        flush=True,
    )

    pool = ThreadPoolExecutor(max_workers=args.read_threads)
    chunks = [jobs[i : i + args.batch_size] for i in range(0, len(jobs), args.batch_size)]

    def submit(idx):
        """异步预读第 idx 个 chunk：**逐条 submit**，让所有线程并行。
        （若整块丢进一个 task 里顺序读，就退化成单线程 ~12 req/s。）
        kaldi-fbank 也放在这些线程里算 —— 它是 CPU 上的重活，放主循环会成为新瓶颈。"""
        if idx >= len(chunks):
            return None

        def one(job):
            seg = read_enroll(job)
            if seg is None:
                return None
            return to_fbank(seg) if needs_fbank else torch.from_numpy(
                np.ascontiguousarray(seg)
            )

        return [pool.submit(one, j) for j in chunks[idx]]

    out_emb, out_keys, n_bad = [], [], 0
    t0 = time.perf_counter()
    pending = submit(0)
    for ci, chunk in enumerate(chunks):
        loaded = [f.result() for f in pending]
        pending = submit(ci + 1)
        waves, keys = [], []
        for job, feat in zip(chunk, loaded):
            if feat is None:
                n_bad += 1
                continue
            waves.append(feat)
            keys.append(job["key"])
        if not waves:
            continue
        lens = torch.tensor([w.shape[0] for w in waves], dtype=torch.int32)
        max_len = int(lens.max())
        if needs_fbank:
            batch = torch.zeros(len(waves), max_len, waves[0].shape[1], dtype=torch.float32)
            for i, w in enumerate(waves):
                batch[i, : w.shape[0]] = w
        else:
            batch = torch.zeros(len(waves), max_len, dtype=torch.float32)
            for i, w in enumerate(waves):
                batch[i, : w.numel()] = w
        with torch.no_grad():
            emb = encoder(batch.to(args.device), lens.to(args.device))
        out_emb.append(emb.float().cpu().numpy())
        out_keys.extend(keys)

        if ci % 20 == 0:
            el = time.perf_counter() - t0
            print(
                f"[emb] {len(out_keys)}/{len(jobs)} 条, {el:.0f}s, "
                f"{len(out_keys)/max(el,1e-6):.1f} utt/s, 坏样本 {n_bad}",
                file=sys.stderr,
                flush=True,
            )

    tag = f"{args.shard:02d}of{args.num_shards:02d}"
    mat = np.concatenate(out_emb, axis=0) if out_emb else np.zeros((0, dim), np.float32)
    np.save(os.path.join(args.out_dir, f"emb.{tag}.npy"), mat)
    with open(os.path.join(args.out_dir, f"keys.{tag}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out_keys) + ("\n" if out_keys else ""))
    el = time.perf_counter() - t0
    print(
        f"[emb] DONE shard {args.shard}: {mat.shape[0]} 条（坏样本 {n_bad}），"
        f"{el:.0f}s, {mat.shape[0]/max(el,1e-6):.1f} utt/s",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    main()
