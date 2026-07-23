#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Adapted from the MetricX evaluation script (Google LLC), Apache License 2.0:
#     http://www.apache.org/licenses/LICENSE-2.0
# Trimmed to the two meta-metrics reported in the paper.
"""System- and segment-level meta-evaluation for ESD (SPA and Acc_eq).

Given the per-segment sentence scores selected by ``rerank.py`` and the WMT
human scores, this reports:

* **system level**: Soft Pairwise Accuracy (SPA) and pairwise ranking accuracy;
* **segment level**: tie-calibrated pairwise accuracy (Acc_eq), via
  ``tau_optimization`` from mt-metrics-eval.

Requires the ``mt_metrics_eval`` package and its WMT data (install and download
from https://github.com/google-research/mt-metrics-eval).

Inputs:
* ``--metric_file``  : ``<...>.seg.scores`` from rerank.py (``<system> <score>`` per line).
* ``--human_file``   : WMT human scores, e.g.
                       ``.../wmt24/human-scores/<lp>.mqm.seg.score``.
"""

import argparse
import json
import os
from typing import Any, Tuple

import numpy as np
import scipy.stats
from mt_metrics_eval import data, stats, tau_optimization

# Number of source segments per language pair in WMT24 (for a sanity check).
LP_NUM_SEGMENTS = {"en-de": 998, "en-es": 998, "ja-zh": 722}


def _read_scores(path, bad_systems):
    """Read a '<system> <score>' file into {system: [(system, score_or_None), ...]}."""
    scores = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sep = "\t" if "\t" in line else " "
            system, value = line.split(sep)
            if system in bad_systems:
                continue
            score = None if value == "None" else -1.0 * float(value)  # negate: higher = better
            scores.setdefault(system, []).append((system, score))
    return scores


def _to_matrices(instances) -> Tuple[np.ndarray, np.ndarray]:
    row_of, col_of = {}, {}
    for inst in instances:
        row_of.setdefault(inst["system_id"], len(row_of))
        col_of.setdefault(inst["segment_id"], len(col_of))
    metric = np.full((len(row_of), len(col_of)), None, dtype=object)
    human = np.full((len(row_of), len(col_of)), None, dtype=object)
    for inst in instances:
        r, c = row_of[inst["system_id"]], col_of[inst["segment_id"]]
        metric[r, c] = inst["prediction"]
        human[r, c] = inst["label"]
    return metric, human


def main():
    p = argparse.ArgumentParser(description="System/segment-level meta-evaluation.")
    p.add_argument("--dataset", default="wmt24")
    p.add_argument("--lp", required=True, help="Language pair, e.g. en-de.")
    p.add_argument("--metric_file", required=True, help="'*.seg.scores' from rerank.py.")
    p.add_argument("--human_file", required=True, help="WMT human seg scores.")
    p.add_argument("--output_file", required=True)
    args = p.parse_args()

    # Exclude the reference-based system, outliers, and systems without scores.
    evs = data.EvalSet(args.dataset, args.lp)
    bad_systems = {evs.std_ref} | evs.outlier_sys_names
    for system_id, scores in evs.Scores("seg", "mqm").items():
        if not any(s is not None for s in scores):
            bad_systems.add(system_id)

    metric_scores = _read_scores(args.metric_file, bad_systems)
    # Human file: '<system> <score>' with sign kept as-is.
    human_scores = {}
    with open(args.human_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sep = "\t" if "\t" in line else " "
            system, value = line.split(sep)
            if system in bad_systems:
                continue
            human_scores.setdefault(system, []).append(
                (system, None if value == "None" else float(value)))

    n = LP_NUM_SEGMENTS[args.lp]
    instances = []
    for system_id in human_scores:
        assert len(human_scores[system_id]) == n, \
            f"{system_id}: {len(human_scores[system_id])} != {n} segments"
        assert len(metric_scores[system_id]) == n, \
            f"{system_id}: {len(metric_scores[system_id])} != {n} segments"
        for i in range(n):
            instances.append({
                "system_id": system_id, "segment_id": i,
                "prediction": metric_scores[system_id][i][1],
                "label": human_scores[system_id][i][1],
            })

    metric_seg, human_seg = _to_matrices(instances)
    metric_sys = np.mean(metric_seg, axis=1)
    human_sys = np.apply_along_axis(
        lambda row: np.mean(row[row != None]), 1, human_seg)  # noqa: E711

    # System level: pairwise accuracy and Soft Pairwise Accuracy (SPA).
    agree, num_pairs = stats.Agreement(human_sys, metric_sys)
    sys_accuracy = agree / num_pairs
    sys_spa = stats.PairwiseConfidenceError(
        human_seg.reshape(-1), metric_seg.reshape(-1),
        human_seg.shape[0], filter_nones=True)[0]

    # Segment level: tie-calibrated pairwise accuracy (Acc_eq).
    tie = tau_optimization.tau_optimization(
        metric_seg.T, human_seg.T, tau_optimization.TauSufficientStats.acc_23)

    metrics = {
        "system_level": {"spa": sys_spa, "accuracy": sys_accuracy},
        "segment_level_group_by_item": {"accuracy": tie.best_tau,
                                        "epsilon": tie.best_threshold},
    }
    print(json.dumps(metrics, indent=2))
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
