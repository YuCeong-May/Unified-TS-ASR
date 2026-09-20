#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Speaker-Conformer-CTC 离线评测 / 推理。

默认每条 JSONL 给出待识别 ``source`` 与一条干净的注册语音 ``enroll_source``
（可由 ``--enroll_field`` 改名）。脚本会在线提取 CAM++ embedding。若传入
``--emb_store``，则改为从 JSONL 的 ``enroll_key``（可由 ``--enroll_key_field`` 改名）
在离线 embedding store 查表；该模式不会读取 enroll 音频或运行 CAM++。

示例：
  python infer_eval_speaker_ctc.py \
    --config exp/speaker_conformer_ctc_online_enroll/config.yaml \
    --init_param exp/speaker_conformer_ctc_online_enroll/model.pt.best \
    --test_sets dev:/data/dev_with_enroll.jsonl --output_dir exp/.../eval \
    --gpus 0,1 --batch_size 16
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from eval_metrics import evaluate, save_metrics_md  # noqa: E402


def log(message):
    print(f"[infer_eval_speaker_ctc {time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def make_progress(total):
    """Create a terminal progress bar without making tqdm a hard dependency."""
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


def read_jsonl(path, dataset, enroll_field, enroll_key_field, use_offline_emb):
    """Read inference records and retain both the enroll path and its key.

    ``utt`` is the conventional Kaldi-style utterance identifier used by many
    evaluation tools.  FunASR inference calls it ``key``.  Keep both names so
    neither the decoder nor downstream scoring scripts lose the original ID.
    """
    rows = []
    with open(path, encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            enroll_source = obj.get(enroll_field, obj.get("ref_source"))
            enroll_key = obj.get(enroll_key_field)
            if enroll_key is None:
                enroll_key = obj.get("ref_key", obj.get("enroll_utt"))
            # Older TS test JSONLs contain only enroll_source.  The CAMP++
            # utterance store convention uses the clean wav filename stem as
            # its key (e.g. BAC009S0904W0134.wav -> BAC009S0904W0134), so use
            # that deterministic fallback before declaring the sample invalid.
            if not enroll_key and enroll_source:
                enroll_key = Path(str(enroll_source)).stem
            if use_offline_emb and not enroll_key:
                raise ValueError(
                    f"{path}:{line_no} misses {enroll_key_field!r}; offline embedding "
                    "inference needs the key of the enrollment utterance."
                )
            if not use_offline_emb and not enroll_source:
                raise ValueError(
                    f"{path}:{line_no} misses {enroll_field!r}; "
                    "speaker CTC inference requires a clean enrollment waveform."
                )
            utt = obj.get("utt", obj.get("key", f"{dataset}:{line_no}"))
            rows.append({
                "key": obj.get("key", utt),
                "utt": utt,
                "source": obj["source"],
                "enroll_source": enroll_source,
                "enroll_key": str(enroll_key) if enroll_key is not None else None,
                "target": obj.get("target", ""),
                "dataset": dataset,
            })
    return rows


class EnrollEmbeddingStore:
    """Read-only utterance embedding lookup for both supported store formats.

    Supported formats:
      * ``embeddings.npy`` + ``utt2index.json`` (the CAMP++ extractor store);
      * ``embeddings.npy`` + ``keys.txt``;
      * ``spk_emb.f32`` + ``spk_emb.keys`` + ``spk_emb.meta.json``.
    """

    def __init__(self, store_dir):
        self.dir = Path(store_dir)
        if not self.dir.is_dir():
            raise FileNotFoundError(f"embedding store not found: {self.dir}")
        index_path = self.dir / "utt2index.json"
        if index_path.is_file():
            with open(index_path, encoding="utf-8") as source:
                self.key_to_index = {str(key): int(value) for key, value in json.load(source).items()}
        else:
            keys_path = self.dir / "keys.txt"
            if not keys_path.is_file():
                keys_path = self.dir / "spk_emb.keys"
            if not keys_path.is_file():
                raise FileNotFoundError(
                    f"{self.dir} needs utt2index.json, keys.txt, or spk_emb.keys"
                )
            with open(keys_path, encoding="utf-8") as source:
                self.key_to_index = {
                    line.rstrip("\n"): index for index, line in enumerate(source) if line.strip()
                }

        npy_path = self.dir / "embeddings.npy"
        if npy_path.is_file():
            self.embeddings = np.load(npy_path, mmap_mode="r")
        else:
            meta_path, raw_path = self.dir / "spk_emb.meta.json", self.dir / "spk_emb.f32"
            if not meta_path.is_file() or not raw_path.is_file():
                raise FileNotFoundError(
                    f"{self.dir} needs embeddings.npy or spk_emb.f32 + spk_emb.meta.json"
                )
            with open(meta_path, encoding="utf-8") as source:
                meta = json.load(source)
            self.embeddings = np.memmap(
                raw_path, dtype=np.float32, mode="r",
                shape=(int(meta["num"]), int(meta["dim"])),
            )
        if self.embeddings.ndim != 2:
            raise ValueError(f"embedding array must be 2-D, got {self.embeddings.shape}")
        if not self.key_to_index:
            raise ValueError(f"empty embedding index: {self.dir}")
        max_index = max(self.key_to_index.values())
        if max_index >= len(self.embeddings) or min(self.key_to_index.values()) < 0:
            raise ValueError("embedding index points outside embedding array")

    def get(self, key):
        index = self.key_to_index.get(str(key))
        if index is None:
            raise KeyError(f"enroll key {key!r} is absent from {self.dir}")
        return np.array(self.embeddings[index], dtype=np.float32, copy=True)


def load_state_dict(path):
    """Load a regular checkpoint or a DeepSpeed ZeRO-1 checkpoint directory."""
    from unified_ts_asr.runtime import load_checkpoint_state

    return load_checkpoint_state(path)


def build(config_path, state_dict=None, offline_spk_emb=False):
    """Build the exact model, tokenizer and ASR frontend used for training."""
    import importlib

    from omegaconf import OmegaConf
    from funasr.register import tables

    # ``import funasr`` scans submodules best-effort and intentionally suppresses
    # import errors.  That is convenient for the generic package, but leaves this
    # script with a cryptic ``NoneType is not callable`` when this experimental
    # model was not discovered.  Import the model owning this inference path
    # explicitly so its @tables.register decorator always runs (and so a genuine
    # dependency error is shown at the real import site).
    importlib.import_module("funasr.models.tsasr.speaker_conformer_ctc")

    from unified_ts_asr.runtime import resolve_config_paths, _write_embedded_campplus

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
    temporary = None
    if offline_spk_emb:
        # No enroll waveform is consumed in this mode, so avoid loading the
        # frozen CAM++ checkpoint solely to leave it unused.
        model_conf["skip_spk_encoder"] = True
    elif cfg.get("model") == "SpeakerConformerCTCOnlineEnroll":
        speaker_conf = dict(model_conf.get("spk_encoder_conf") or {})
        init_param = speaker_conf.get("init_param")
        if (not init_param or not os.path.isfile(str(init_param))) and state_dict is not None:
            import tempfile

            temporary = tempfile.TemporaryDirectory(prefix="unified-ts-asr-")
            speaker_conf["init_param"] = _write_embedded_campplus(
                state_dict, temporary.name
            )
            model_conf["spk_encoder_conf"] = speaker_conf
    model_cls = tables.model_classes.get(cfg["model"])
    if model_cls is None:
        raise KeyError(
            f"Model {cfg['model']!r} is not registered after importing "
            "funasr.models.tsasr.speaker_conformer_ctc. "
            "Check that --config is the SpeakerConformerCTCOnlineEnroll "
            "training config."
        )
    model = model_cls(**model_conf)
    if temporary is not None:
        temporary.cleanup()
    return model, tokenizer, frontend


def load_enroll_fbank(path, max_seconds=0.0):
    """Load a clean enroll waveform and reproduce training CAM++ features."""
    import soundfile as sf
    import torch
    import torchaudio
    import torchaudio.compliance.kaldi as kaldi

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
    feat = kaldi.fbank(wav.unsqueeze(0), num_mel_bins=80)
    if feat.size(0) == 0:
        raise ValueError(f"empty enrollment fbank: {path}")
    return feat - feat.mean(dim=0, keepdim=True)


def make_ref_batch(items, max_seconds):
    """Return padded CAM++ fbank batch plus the original valid item indices."""
    import torch

    feats, keep = [], []
    for index, item in enumerate(items):
        try:
            feats.append(load_enroll_fbank(item["enroll_source"], max_seconds))
            keep.append(index)
        except Exception as exc:
            log(f"skip key={item['key']}: cannot load enroll {item['enroll_source']}: {exc}")
    if not feats:
        return None, None, keep
    lengths = torch.tensor([feat.size(0) for feat in feats], dtype=torch.int32)
    ref = torch.zeros(len(feats), int(lengths.max()), 80, dtype=torch.float32)
    for index, feat in enumerate(feats):
        ref[index, : feat.size(0)] = feat
    return ref, lengths.view(-1, 1), keep


def infer_shard(gpu, config_path, init_param, items, batch_size, enroll_max_seconds,
                emb_store_dir, queue):
    """Load one model on one GPU and return decoded rows through ``queue``."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    try:
        import torch

        state_dict = load_state_dict(init_param)
        model, tokenizer, frontend = build(
            config_path,
            state_dict=state_dict,
            offline_spk_emb=bool(emb_store_dir),
        )
        if emb_store_dir:
            # The model intentionally omitted frozen CAM++ in offline mode.
            # Its checkpoint tensors are therefore irrelevant rather than a
            # warning about an incompatible ASR architecture.
            state_dict = {
                name: tensor for name, tensor in state_dict.items()
                if not name.startswith("spk_encoder.")
            }
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            log(f"GPU{gpu} missing keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            log(f"GPU{gpu} unexpected keys ({len(unexpected)}): {unexpected[:5]}")
        model.to("cuda:0").eval()
        emb_store = EnrollEmbeddingStore(emb_store_dir) if emb_store_dir else None
        if emb_store is not None:
            log(
                f"GPU{gpu} offline enroll embeddings: {emb_store.dir} "
                f"({len(emb_store.key_to_index)} x {emb_store.embeddings.shape[1]})"
            )

        results = []
        profile = {"batches": 0, "utts": 0, "wall": 0.0, "load_data": 0.0, "extract_feat": 0.0}
        for start in range(0, len(items), batch_size):
            original = items[start: start + batch_size]
            if emb_store is None:
                ref, ref_lengths, keep = make_ref_batch(original, enroll_max_seconds)
                bad = set(range(len(original))) - set(keep)
                for index in bad:
                    item = original[index]
                    results.append({**item, "hyp": "", "unk": 0, "no_tgt_spk": 0,
                                    "error": "enroll_load_failed"})
                if ref is None:
                    queue.put(("progress", gpu, len(original)))
                    continue
                batch = [original[index] for index in keep]
                conditioning = {"ref_speech": ref, "ref_speech_lengths": ref_lengths}
            else:
                batch = original
                try:
                    spk_emb = torch.from_numpy(np.stack([
                        emb_store.get(item["enroll_key"]) for item in batch
                    ])).to(torch.float32)
                except (KeyError, ValueError) as exc:
                    raise RuntimeError(f"offline embedding lookup failed at batch {start}: {exc}") from exc
                conditioning = {"spk_emb": spk_emb}
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
                    # Rejection count for negative-set analysis.  `no_tgt_spk` is
                    # counted model-side by `_strip_reject`, which also removes the
                    # label before `text` is built — leaving it in would charge every
                    # occurrence to CER as an insertion.  `unk` is kept separately and
                    # now means genuine OOV only; before 2026-08-22 both shared <unk>.
                    "no_tgt_spk": int(output.get("no_tgt_spk", 0)),
                    "unk": sum(token == "<unk>" for token in (output.get("token") or [])),
                })
            # The parent owns the sole terminal progress bar; subprocesses only
            # report completed batch sizes to avoid interleaved tqdm rendering.
            # Count the original batch so enrollment-load failures (which are
            # emitted as empty hypotheses above) also advance global progress.
            queue.put(("progress", gpu, len(original)))
        queue.put(("done", gpu, True, results, profile))
    except Exception:
        import traceback
        queue.put(("done", gpu, False, traceback.format_exc(), None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="training config.yaml")
    parser.add_argument("--init_param", required=True, help="checkpoint file or DeepSpeed directory")
    parser.add_argument("--test_sets", nargs="+", required=True, help="name:jsonl or jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--enroll_field", default="enroll_source")
    parser.add_argument("--enroll_key_field", default="enroll_key",
                        help="JSONL field containing the clean enrollment utterance key")
    parser.add_argument("--emb_store", default=None,
                        help="offline embedding store; enables key lookup instead of enroll wav loading")
    parser.add_argument("--enroll_max_seconds", type=float, default=0.0,
                        help="0 means use the entire enrollment audio")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

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
            path, name, args.enroll_field, args.enroll_key_field,
            use_offline_emb=bool(args.emb_store),
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
    ctx, queue, processes = mp.get_context("spawn"), None, []
    queue = ctx.Queue()
    for gpu, shard in zip(gpus, shards):
        if not shard:
            continue
        process = ctx.Process(target=infer_shard, args=(
            gpu, args.config, args.init_param, shard, args.batch_size,
            args.enroll_max_seconds, args.emb_store, queue,
        ))
        process.start()
        processes.append(process)
    collected = {}
    finished = 0
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
                    "enroll_source": row["enroll_source"], "enroll_key": row["enroll_key"],
                    "target": row["target"],
                    "hyp": row["hyp"], "unk": row["unk"],
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
        # Rejection rate: share of utterances whose hypothesis contained
        # `<no_tgt_spk/>`.  On a target-present set this is a false reject; on a
        # target-absent set it is the desired behaviour — the same number means
        # opposite things on the two arms, so it is reported, never thresholded here.
        rejected = sum(1 for row in rows if row.get("no_tgt_spk", 0))
        metrics[dataset]["no_tgt_spk_utt_rate"] = rejected / max(len(rows), 1)
    with open(output_dir / "eval" / "metrics.json", "w", encoding="utf-8") as output:
        json.dump(metrics, output, ensure_ascii=False, indent=2)
    save_metrics_md(metrics, str(output_dir / "eval" / "metrics.md"))
    log(f"done: {len(results)} samples -> {output_dir}")


if __name__ == "__main__":
    main()
