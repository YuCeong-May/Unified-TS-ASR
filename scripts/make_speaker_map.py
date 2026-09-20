# -*- coding: utf-8 -*-
"""Build `spk2id.json` for the auxiliary speaker CE head.

The training jsonl carries a free-form `speaker` string, and nothing in the repo
turned it into a class index — `AudioDatasetKwsSpan` only ever used it as a
hashable identity (same-speaker mixing exclusion, enrollment pooling).  A
`Linear(D, num_speakers)` needs a contiguous 0..N-1 index, so build it once here
and hand the same file to the train **and** the valid set:
`dataset_conf.spk2id`.  A speaker missing from the mapping becomes -1 and is
ignored by the head, which is the correct behaviour for AISHELL dev/test
(disjoint speakers from train).

The key must match what the index builds, `f"{corpus}:{speaker}"`
(`datasets_kws_span_tsasr.py:145`): AISHELL-1 and AISHELL-2 both use bare numeric
speaker ids, so without the corpus prefix unrelated people would collide.

Usage:

    python scripts/make_speaker_map.py \
        --jsonl data/train/train_aishell_with_speaker_tn.jsonl \
        --out conf/spk2id_aishell12.json

Multiple `--jsonl` are allowed and are merged.  Print the resulting count into
`model_conf.spk_ce_conf.num_speakers`; the script writes it on the last line and
the dataset re-checks the two agree at start-up.

Note this counts every speaker in the file, including ones whose utterances the
dataset's own length filters (`max_token_length`, `min/max_source_length`, ...)
later drop.  Those classes simply never receive a gradient; that is cheaper than
making the mapping depend on filter settings.
"""
import argparse
import collections
import json
import sys


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", action="append", required=True,
                        help="training jsonl; repeatable, merged in order")
    parser.add_argument("--out", required=True, help="output json path")
    parser.add_argument("--speaker-field", default="speaker",
                        help="must match dataset_conf.speaker_field")
    parser.add_argument("--min-utts", type=int, default=1,
                        help="drop speakers with fewer utterances than this; "
                             "the surviving ids are still renumbered 0..N-1")
    return parser.parse_args()


def collect(paths, speaker_field):
    """Count utterances per corpus-scoped speaker, preserving first-seen order."""
    counts = collections.OrderedDict()
    n_rows = 0
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                speaker = row.get(speaker_field, row.get("speaker_id"))
                if speaker in (None, ""):
                    raise ValueError(
                        f"{path}: a row has no {speaker_field!r}; the speaker CE "
                        "head needs one on every utterance"
                    )
                key = f"{row.get('corpus', 'other')}:{speaker}"
                counts[key] = counts.get(key, 0) + 1
                n_rows += 1
    return counts, n_rows


def main():
    args = parse_args()
    counts, n_rows = collect(args.jsonl, args.speaker_field)
    if not counts:
        raise SystemExit("no rows read; check --jsonl")

    kept = [name for name, n in counts.items() if n >= args.min_utts]
    dropped = len(counts) - len(kept)
    # Sorted, not first-seen: the mapping must not depend on the order the files
    # happened to be concatenated in, or a rerun silently permutes every class.
    mapping = {name: i for i, name in enumerate(sorted(kept))}
    if len(mapping) < 2:
        raise SystemExit(f"only {len(mapping)} speaker(s) survive; nothing to classify")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=1, sort_keys=True)

    per_corpus = collections.Counter(name.split(":", 1)[0] for name in mapping)
    print(f"read      : {n_rows} utterances from {len(args.jsonl)} file(s)")
    for corpus, n in sorted(per_corpus.items()):
        print(f"  {corpus:<16}: {n} speakers")
    if dropped:
        print(f"dropped   : {dropped} speakers with < {args.min_utts} utterances")
    print(f"wrote     : {args.out}")
    print(f"num_speakers: {len(mapping)}")


if __name__ == "__main__":
    sys.exit(main())
