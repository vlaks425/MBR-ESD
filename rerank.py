#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rerank ESD candidates with MAP or one of the paper's three MBR utilities.

Consumes the ``*.gen.json`` produced by ``generate.py`` and, for every segment,
selects one annotation out of the ``N`` candidates according to the chosen
decision rule:

* ``map``          -- Maximum a Posteriori: the single candidate with the
                      highest model log-probability. (Use with N=1 for greedy.)
* ``mbr_scoresim`` -- MBR with the ScoreSim utility: pick the candidate whose
                      sentence-level MQM score is closest to the mean score of
                      all candidates (utility = -|score - mean|).
* ``mbr_f1``       -- MBR with the standard character-level span F1 utility.
* ``mbr_softf1``   -- MBR with the proposed SoftF1 utility (see softf1.py).

For the MBR span utilities, each candidate is scored by the *average* utility
against all the other candidates, which act as pseudo-references (support
hypotheses). The candidate with the highest average utility is selected.

Outputs
-------
* ``<output_prefix>.<lang_pair>.seg.scores`` : ``<system> <mqm_score>`` per line
  (sentence-level score of the selected annotation; consumed by the
  system/segment-level meta-evaluation).
* ``<output_prefix>.<lang_pair>.error_spans.json`` : the selected error spans
  per (system, segment), consumed by span-level ``evaluate.py``.
"""

import argparse
import json
import multiprocessing
from tqdm import tqdm

from softf1 import build_masks, soft_f1, span_f1

# Sentence-level MQM weights (paper §5.1, Eq. for score): major/critical = -5,
# minor = -1, minor punctuation = -0.1, non-translation = -25, floored at -25.
W_MAJOR = -5.0
W_MINOR = -1.0
W_MINOR_PUNCT = -0.1
W_NON_TRANSLATION = -25.0
SCORE_FLOOR = -25.0

# Required fields for a well-formed error object (QE / MQM mode).
REQUIRED_KEYS = ["severity", "error_span", "sub_category", "category"]


def error_spans_to_score(raw_annotation, hypo):
    """Parse a raw model annotation (JSON string) into (mqm_score, valid_spans).

    Spans that are malformed, empty, or not found verbatim in ``hypo`` are
    dropped. Returns ``(0.0, [])`` if the annotation cannot be parsed or is
    missing required fields. ``valid_spans`` carry resolved character offsets,
    ready for :func:`softf1.build_masks`.
    """
    try:
        parsed = json.loads(raw_annotation)
        for error in parsed["errors"]:
            if any(key not in error for key in REQUIRED_KEYS):
                return 0.0, []
    except Exception:
        return 0.0, []

    score = 0.0
    valid_spans = []
    for error in parsed["errors"]:
        severity = error["severity"].capitalize()
        category = error.get("category", "Unknown").capitalize()
        sub_category = error.get("sub_category", "Unknown").capitalize()
        span_text = error["error_span"]
        if not isinstance(span_text, str) or len(span_text) == 0 or span_text not in hypo:
            continue
        start = hypo.index(span_text)
        valid_spans.append({
            "error_span": span_text,
            "error_span_start": start,
            "error_span_end": start + len(span_text),
            "category": category,
            "sub-category": sub_category,
            "severity": severity,
        })
        if severity in ("Critical", "Major"):
            score += W_NON_TRANSLATION if category == "Non-translation!" else W_MAJOR
        elif severity == "Minor":
            score += W_MINOR_PUNCT if (category == "Fluency" and sub_category == "Punctuation") else W_MINOR
        # "Neutral" and unknown severities contribute nothing.

    return max(score, SCORE_FLOOR), valid_spans


# --------------------------------------------------------------------------
# Decision rules
# --------------------------------------------------------------------------
def rerank_map(logps, scores, spans):
    """MAP: the candidate with the highest model log-probability."""
    best = max(range(len(logps)), key=lambda i: logps[i])
    return scores[best], spans[best]


def rerank_scoresim(scores, spans):
    """MBR-ScoreSim: candidate whose MQM score is closest to the mean score."""
    mean_score = sum(scores) / len(scores)
    best = max(range(len(scores)), key=lambda i: -abs(scores[i] - mean_score))
    return scores[best], spans[best]


def _mbr_span_utility(scores, spans, hypo_length, metric):
    """Generic pairwise-average MBR over a span-similarity ``metric``.

    ``metric(cand_masks, support_masks) -> {"f1": ...}``. Each candidate's utility
    is the mean F1 against every other candidate. Results are cached by mask.
    """
    masks = [build_masks(s, hypo_length) for s in spans]
    n = len(masks)
    utilities = []
    cache = {}
    for i in range(n):
        key = masks[i]
        if key in cache:
            utilities.append(cache[key])
            continue
        others = [metric(masks[i], masks[j])["f1"] for j in range(n) if j != i]
        util = sum(others) / len(others) if others else 0.0
        cache[key] = util
        utilities.append(util)
    best = max(range(n), key=lambda i: utilities[i])
    return scores[best], spans[best]


def rerank_mbr_f1(scores, spans, hypo_length):
    """MBR with the standard character-level span F1 utility."""
    return _mbr_span_utility(scores, spans, hypo_length, span_f1)


def rerank_mbr_softf1(scores, spans, hypo_length):
    """MBR with the proposed SoftF1 utility."""
    return _mbr_span_utility(scores, spans, hypo_length, soft_f1)


def select(method, logps, scores, spans, hypo_length):
    """Dispatch to the requested decision rule; returns (score, error_spans)."""
    if len(scores) == 1:
        return scores[0], spans[0]
    if method == "map":
        return rerank_map(logps, scores, spans)
    if method == "mbr_scoresim":
        return rerank_scoresim(scores, spans)
    if method == "mbr_f1":
        return rerank_mbr_f1(scores, spans, hypo_length)
    if method == "mbr_softf1":
        return rerank_mbr_softf1(scores, spans, hypo_length)
    raise ValueError(f"Unknown method: {method}")


# --------------------------------------------------------------------------
# Driver (one worker per system, parallelized over systems)
# --------------------------------------------------------------------------
_METHOD = None  # set per-process via the pool initializer


def _init_worker(method):
    global _METHOD
    _METHOD = method


def _process_system(item):
    system_name, segments = item
    seg_scores, seg_spans = [], []
    for candidates in segments:
        if len(candidates) == 0:
            seg_scores.append(0.0)
            seg_spans.append([])
            continue
        hypo = candidates[0][2]
        src = candidates[0][3]
        logps, scores, spans = [], [], []
        for cand_text, logp, _hypo, _src in candidates:
            score, valid_spans = error_spans_to_score(cand_text, hypo)
            logps.append(logp)
            scores.append(score)
            spans.append(valid_spans)
        score, chosen = select(_METHOD, logps, scores, spans, len(hypo))
        seg_scores.append(score)
        seg_spans.append((chosen, hypo, src))
    return system_name, seg_scores, seg_spans


def parse_args():
    p = argparse.ArgumentParser(description="Rerank ESD candidates (MAP / MBR).")
    p.add_argument("--model_output", required=True, help="'*.gen.json' from generate.py.")
    p.add_argument("--output_prefix", required=True,
                   help="Writes '<prefix>.seg.scores' and '<prefix>.error_spans.json'.")
    p.add_argument("--method", required=True,
                   choices=["map", "mbr_scoresim", "mbr_f1", "mbr_softf1"])
    p.add_argument("--num_processes", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.model_output, "r", encoding="utf-8") as f:
        model_output = json.load(f)

    n_proc = min(args.num_processes, max(1, len(model_output)))
    with multiprocessing.Pool(n_proc, initializer=_init_worker, initargs=(args.method,)) as pool:
        results = list(tqdm(
            pool.imap(_process_system, model_output.items()),
            total=len(model_output), desc=f"Reranking ({args.method})"))

    reranked_scores = {name: scores for name, scores, _ in results}
    reranked_spans = {name: spans for name, _, spans in results}

    with open(args.output_prefix + ".seg.scores", "w", encoding="utf-8") as f:
        for name, scores in reranked_scores.items():
            for score in scores:
                f.write(f"{name} {score}\n")
    with open(args.output_prefix + ".error_spans.json", "w", encoding="utf-8") as f:
        json.dump(reranked_spans, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
