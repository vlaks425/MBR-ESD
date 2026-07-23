#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Span-level evaluation of predicted error spans against human MQM annotations.

Compares the reranked predictions (``*.error_spans.json`` from ``rerank.py``)
against the human error-span annotations, reporting the average character-level
``F1`` and the proposed ``SoftF1`` (the paper's headline span-level metric).

Both metrics are computed with the exact functions in ``softf1.py`` so that the
utility used for MBR reranking and the metric used for evaluation are the same.

Human annotation file
---------------------
A JSON array of segment objects (e.g. ``data/wmt24/2024_en-de.json``). Each has
``system``, ``source``, ``target`` (the translation), and ``error_span`` — a
list of ``{severity, error_span_start, error_span_end, ...}`` items. Predictions
are matched to a human segment by ``system`` and ``source + target``.
"""

import argparse
import json
from tqdm import tqdm

from softf1 import build_masks, soft_f1, span_f1


def load_human_annotations(path):
    """Return ``{system: {source+target: [error_span items]}}`` and the target text."""
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    human = {}
    for row in rows:
        system = row["system"]
        key = row["source"] + row["target"]
        human.setdefault(system, {})[key] = row
    return human


def evaluate(prediction_file, human_file):
    with open(prediction_file, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    human = load_human_annotations(human_file)

    per_segment = []
    total, skipped = 0, 0
    for system, segments in tqdm(predictions.items(), desc="Scoring", total=len(predictions)):
        if system not in human:
            continue
        for pred_spans, hypo, src in segments:
            total += 1
            key = src.strip() + hypo.strip()
            if key not in human[system]:
                skipped += 1
                continue
            human_row = human[system][key]
            hypo_length = len(hypo.strip())
            assert hypo_length == len(human_row["target"]), (
                f"length mismatch: pred hypo {hypo_length} vs human target "
                f"{len(human_row['target'])}")

            pred_masks = build_masks(pred_spans, hypo_length)
            human_masks = build_masks(human_row["error_span"], hypo_length)

            per_segment.append({
                "f1": span_f1(pred_masks, human_masks)["f1"],
                "soft_f1": soft_f1(pred_masks, human_masks)["f1"],
            })
    return per_segment, total, skipped


def main():
    p = argparse.ArgumentParser(description="Span-level F1 / SoftF1 evaluation.")
    p.add_argument("--prediction_error", required=True,
                   help="'*.error_spans.json' produced by rerank.py.")
    p.add_argument("--human_errors", required=True,
                   help="Human error-span JSON, e.g. data/wmt24/2024_en-de.json.")
    p.add_argument("--output_file", required=True, help="Where to write the score summary.")
    args = p.parse_args()

    results, total, skipped = evaluate(args.prediction_error, args.human_errors)
    n = len(results)
    summary = {
        "average_f1": sum(r["f1"] for r in results) / n,
        "average_soft_f1": sum(r["soft_f1"] for r in results) / n,
        "num_scored": n,
        "num_total": total,
        "num_skipped": skipped,
    }
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
