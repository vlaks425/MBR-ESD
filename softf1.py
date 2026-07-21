#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SoftF1: a soft span-level similarity/utility function for Error Span Detection.

This is the core span-level utility function proposed in

    Minimum Bayes Risk Decoding for Error Span Detection in
    Reference-Free Automatic Machine Translation Evaluation
    (Lyu et al., 2025; arXiv:2512.07540)

It is used both as the MBR utility function (``SoftF1(E_c, E_s)`` with a support
hypothesis ``E_s``) and as a span-level evaluation metric (``SoftF1(E, E_y)``
against the human gold annotation ``E_y``).

------------------------------------------------------------------------------
Definition (paper §4.2)
------------------------------------------------------------------------------
An error annotation ``E`` over a translation of ``L`` characters is a pair of
index sets ``E^maj, E^min ⊆ {1, ..., L}`` giving the character positions covered
by major and minor errors. We represent it as a dense *severity vector*
``v ∈ R^L``:

    v^(i) = β · 1[i ∈ E^maj] + γ · 1[i ∈ E^min],

where ``β`` and ``γ`` are the penalties for major and minor errors
(defaults: ``β = 1.0``, ``γ = 0.5``). The total severity is the L1 norm
``||v||_1 = Σ_i |v^(i)|``.

Given a candidate annotation ``E_c`` (vector ``v_c``) and a support/gold
annotation ``E_s`` (vector ``v_s``), we measure their discrepancy by the L1
distance ``||v_c - v_s||_1`` and define:

    SoftP(E_c, E_s) = 1 - ||v_c - v_s||_1 / (L + ||v_c||_1 + eps)
    SoftR(E_c, E_s) = 1 - ||v_c - v_s||_1 / (L + ||v_s||_1 + eps)
    SoftF1(E_c, E_s) = harmonic_mean(SoftP, SoftR)

``L`` in the denominator is a normalizer that keeps the score in ``[0, 1]``.
``eps`` (:data:`EPS`, default ``1.0``) is a small smoothing constant that
guards against division by zero: the WMT datasets contain a few *empty*
translations (``L = 0``), for which the bare denominator ``L + ||v||_1`` would
be ``0``. With ``eps = 1.0`` an empty translation matched against an empty
annotation still scores ``1.0`` instead of raising. This matches the
implementation used to run the experiments in the paper; for the common
``L > 0`` case the constant has a negligible, monotone effect on the score.

Compared with the standard character-level ``F1``, SoftF1 preserves the
desirable properties of ``F1`` (bounded in ``[0, 1]``, symmetric, equal to 1 on
an exact match) while avoiding its "zero-credit plateau": whenever two
annotations have no character-level overlap, ``F1`` collapses to 0, whereas
SoftF1 still reflects *how far apart* the two annotations are. This matters most
when the support hypothesis is empty (``v_s = 0``): ``F1`` assigns 0 to every
non-empty candidate, but SoftF1 assigns a candidate with a small error vector a
utility close to 1, letting MBR meaningfully rank candidates even when the
consensus is that the translation is error-free.
"""

import json
from typing import Dict, List, Optional, Sequence, Tuple

# Recommended penalties (paper §5.1): major = β, minor = γ.
BETA = 1.0   # penalty weight for a major (or critical) error character
GAMMA = 0.5  # penalty weight for a minor error character

# Smoothing constant added to each denominator. Its purpose is to keep the
# score well-defined for empty translations (L = 0), which occur in the WMT
# data; for L > 0 the denominator is already positive and eps only shifts the
# score slightly. Set to 1.0 to reproduce the paper's experiments.
EPS = 1.0

# A single annotation is represented as a pair of equal-length "masks", each a
# string of '0'/'1' over the L characters of the translation:
#   masks[0] -> major mask, masks[1] -> minor mask.
Masks = Tuple[str, str]


# ---------------------------------------------------------------------------
# Error-span string processing
# ---------------------------------------------------------------------------
def locate_span(error_span: str, hypo: str) -> Optional[Tuple[int, int]]:
    """Locate an error span *text* inside a hypothesis and return ``(start, end)``.

    Returns ``None`` if the span is empty, not a string, or not found verbatim in
    ``hypo`` (the same permissive behavior used during scoring: unmatched or
    malformed spans are simply skipped rather than raising).
    """
    if not isinstance(error_span, str) or len(error_span) == 0:
        return None
    if error_span not in hypo:
        return None
    start = hypo.index(error_span)
    return start, start + len(error_span)


def annotate_spans(errors: Sequence[dict], hypo: str) -> List[dict]:
    """Resolve a list of raw error dicts against ``hypo`` into indexed spans.

    Each input error is expected to carry at least ``error_span`` (the substring
    of the hypothesis flagged as an error) and ``severity``. Spans that cannot be
    located verbatim in ``hypo`` are dropped. The output dicts carry
    ``error_span_start`` / ``error_span_end`` (character offsets) and a
    normalized ``severity``.
    """
    annotated = []
    for error in errors:
        span_text = error.get("error_span", "")
        located = locate_span(span_text, hypo)
        if located is None:
            continue
        start, end = located
        annotated.append({
            "error_span": span_text,
            "error_span_start": start,
            "error_span_end": end,
            "severity": str(error.get("severity", "")).capitalize(),
        })
    return annotated


def split_by_severity(errors: Sequence[dict]) -> Dict[str, List[dict]]:
    """Bucket errors into ``{"major": [...], "minor": [...]}`` by severity.

    "Critical" is folded into "major" (following the WMT MQM convention);
    any other severity (e.g. "Neutral") is ignored for span matching.
    """
    buckets: Dict[str, List[dict]] = {"major": [], "minor": []}
    for error in errors:
        severity = str(error.get("severity", "")).lower()
        if severity in ("major", "critical"):
            buckets["major"].append(error)
        elif severity == "minor":
            buckets["minor"].append(error)
    return buckets


def spans_to_mask(error_spans: Sequence[dict], hypo_length: int) -> str:
    """Convert indexed spans into a ``'0'/'1'`` character mask of length ``hypo_length``.

    A position is marked ``'1'`` if it is covered by at least one span. Indices
    are clamped to ``[0, hypo_length)`` so out-of-range spans are handled safely.
    """
    if hypo_length < 0:
        raise ValueError("hypo_length cannot be negative")
    if hypo_length == 0:
        return ""

    char_list = ["0"] * hypo_length
    for error in error_spans:
        start = max(0, error["error_span_start"])
        end = min(hypo_length, error["error_span_end"])
        for i in range(start, end):
            char_list[i] = "1"
    return "".join(char_list)


def build_masks(errors: Sequence[dict], hypo_length: int) -> Masks:
    """Build ``(major_mask, minor_mask)`` from a flat list of indexed error dicts."""
    by_severity = split_by_severity(errors)
    major_mask = spans_to_mask(by_severity["major"], hypo_length)
    minor_mask = spans_to_mask(by_severity["minor"], hypo_length)
    return major_mask, minor_mask


# ---------------------------------------------------------------------------
# Severity vector and SoftF1
# ---------------------------------------------------------------------------
def severity_vector(masks: Masks, beta: float = BETA, gamma: float = GAMMA) -> List[float]:
    """Build the dense severity vector ``v`` (paper Eq. for ``v^(i)``) from masks.

    ``v^(i) = beta * major[i] + gamma * minor[i]``. Note that a character flagged
    as *both* major and minor accumulates ``beta + gamma``.
    """
    major_mask, minor_mask = masks
    if len(major_mask) != len(minor_mask):
        raise ValueError("major and minor masks must have the same length")
    return [
        beta * (major_mask[i] == "1") + gamma * (minor_mask[i] == "1")
        for i in range(len(major_mask))
    ]


def _l1(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


def soft_f1(
    candidate: Masks,
    support: Masks,
    beta: float = BETA,
    gamma: float = GAMMA,
    eps: float = EPS,
) -> Dict[str, float]:
    """Compute ``SoftP``, ``SoftR`` and ``SoftF1`` between two annotations.

    ``candidate`` and ``support`` are each a ``(major_mask, minor_mask)`` pair of
    equal-length ``'0'/'1'`` strings. Returns a dict with keys
    ``"soft_precision"``, ``"soft_recall"`` and ``"f1"``.

    ``eps`` is the denominator smoothing constant (see :data:`EPS`); it keeps the
    score defined for an empty translation (``L == 0``), where two empty
    annotations match exactly and score 1.0.
    """
    L = len(candidate[0])
    if not (len(candidate[0]) == len(candidate[1]) == len(support[0]) == len(support[1])):
        raise ValueError("all four masks must have the same length")

    v_c = severity_vector(candidate, beta, gamma)
    v_s = severity_vector(support, beta, gamma)

    dist = _l1(v_c, v_s)
    norm_c = sum(v_c)   # v is non-negative, so ||v||_1 == sum(v)
    norm_s = sum(v_s)

    # eps guards L == 0 (empty translation) against division by zero.
    soft_p = 1.0 - dist / (L + norm_c + eps)
    soft_r = 1.0 - dist / (L + norm_s + eps)
    f1 = 2.0 * soft_p * soft_r / (soft_p + soft_r) if (soft_p + soft_r) > 0 else 0.0

    return {"soft_precision": soft_p, "soft_recall": soft_r, "f1": f1}


def softf1_from_errors(
    candidate_errors: Sequence[dict],
    support_errors: Sequence[dict],
    hypo_length: int,
    beta: float = BETA,
    gamma: float = GAMMA,
    eps: float = EPS,
) -> Dict[str, float]:
    """Convenience wrapper: compute SoftF1 directly from two lists of indexed errors.

    Each error dict must carry ``error_span_start``, ``error_span_end`` and
    ``severity`` (see :func:`annotate_spans` to derive these from raw span text).
    """
    candidate = build_masks(candidate_errors, hypo_length)
    support = build_masks(support_errors, hypo_length)
    return soft_f1(candidate, support, beta, gamma, eps)


# ---------------------------------------------------------------------------
# Standard character-level F1 (for reference / contrast in the examples)
# ---------------------------------------------------------------------------
def span_f1(candidate: Masks, support: Masks) -> Dict[str, float]:
    """The WMT QE Shared Task character-level F1, kept for comparison.

    This is the "hard" utility SoftF1 improves upon: whenever the candidate and
    support annotations share no error characters, F1 collapses to 0 (the
    "zero-credit plateau"). Minor/major mismatches at the same position earn
    partial (0.5) credit.
    """
    c_major, c_minor = candidate
    s_major, s_minor = support
    assert len(c_major) == len(c_minor) == len(s_major) == len(s_minor)

    tp = total_pred = total_human = 0.0
    for hm, pm, hn, pn in zip(s_major, c_major, s_minor, c_minor):
        if hm == "1" or hn == "1":
            total_human += 1
        if pm == "1" or pn == "1":
            total_pred += 1
        if hm == "0" and hn == "0":
            continue  # true negative or false positive; neither adds to tp
        if hm == "1" and hn == "0":
            if pm == "1":
                tp += 1
            elif pn == "1":
                tp += 0.5
        elif hn == "1" and hm == "0":
            if pn == "1":
                tp += 1
            elif pm == "1":
                tp += 0.5
        elif hm == "1" and hn == "1":
            if pm == "1" or pn == "1":
                tp += 1

    if total_human == 0 and total_pred == 0:
        return {"recall": 1.0, "precision": 1.0, "f1": 1.0}
    recall = tp / total_human if total_human > 0 else 0.0
    precision = tp / total_pred if total_pred > 0 else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
    return {"recall": recall, "precision": precision, "f1": f1}


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
def _fmt(masks: Masks) -> str:
    return f"major={masks[0]!r} minor={masks[1]!r}"


def _run_examples() -> None:
    hypo = "The cat sat on the moon."
    L = len(hypo)

    print("=" * 78)
    print("SoftF1 examples")
    print(f"hypothesis = {hypo!r}  (L = {L} chars)")
    print("=" * 78)

    # ---- Example 1: string processing -- raw spans -> masks -------------------
    print("\n[1] Error-span string processing (raw span text -> indexed -> masks)")
    raw_errors = [
        {"error_span": "moon", "severity": "major"},   # mistranslation ("moon" vs "mat")
        {"error_span": "sat", "severity": "minor"},     # awkward word choice
        {"error_span": "not-in-hypo", "severity": "major"},  # dropped: not found verbatim
    ]
    annotated = annotate_spans(raw_errors, hypo)
    for e in annotated:
        print(f"    {e['severity']:6s} [{e['error_span_start']:2d}:{e['error_span_end']:2d}] "
              f"{e['error_span']!r}")
    masks_a = build_masks(annotated, L)
    print(f"    -> {_fmt(masks_a)}")

    # ---- Example 2: exact match scores 1.0 -----------------------------------
    print("\n[2] Exact match -> SoftF1 = 1.0")
    print(f"    SoftF1 = {soft_f1(masks_a, masks_a)['f1']:.4f}")

    # ---- Example 3: the empty-support case (SoftF1 vs F1) --------------------
    print("\n[3] Empty support hypothesis (v_s = 0): F1 collapses to 0, SoftF1 does not"
          "\n    (SoftF1 stays high for a small candidate and low for a large one;"
          " F1 gives both 0)")
    empty = ("0" * L, "0" * L)
    small = build_masks(annotate_spans([{"error_span": "moon", "severity": "minor"}], hypo), L)
    large = build_masks(
        annotate_spans([{"error_span": "The cat sat on the moon", "severity": "major"}], hypo), L)
    print(f"    small candidate ({_fmt(small)}):")
    print(f"        SoftF1 = {soft_f1(small, empty)['f1']:.4f}   F1 = {span_f1(small, empty)['f1']:.4f}")
    print(f"    large candidate ({_fmt(large)}):")
    print(f"        SoftF1 = {soft_f1(large, empty)['f1']:.4f}   F1 = {span_f1(large, empty)['f1']:.4f}")
    print("    -> SoftF1 ranks the smaller error higher (closer to the empty consensus);")
    print("       F1 assigns 0 to both and cannot distinguish them.")

    # ---- Example 4: severity matters -----------------------------------------
    print("\n[4] Severity is respected (major penalty beta > minor penalty gamma)")
    cand_major = build_masks(annotate_spans([{"error_span": "moon", "severity": "major"}], hypo), L)
    cand_minor = build_masks(annotate_spans([{"error_span": "moon", "severity": "minor"}], hypo), L)
    gold_minor = cand_minor
    print(f"    gold says 'moon' is a MINOR error:")
    print(f"        candidate MINOR : SoftF1 = {soft_f1(cand_minor, gold_minor)['f1']:.4f}  (exact)")
    print(f"        candidate MAJOR : SoftF1 = {soft_f1(cand_major, gold_minor)['f1']:.4f}  (over-penalized)")

    # ---- Example 5: partial overlap ------------------------------------------
    print("\n[5] Partial overlap degrades gracefully")
    gold = build_masks(annotate_spans([{"error_span": "on the moon", "severity": "major"}], hypo), L)
    partial = build_masks(annotate_spans([{"error_span": "the moon", "severity": "major"}], hypo), L)
    disjoint = build_masks(annotate_spans([{"error_span": "The cat", "severity": "major"}], hypo), L)
    print(f"    gold ({_fmt(gold)})")
    r_partial = soft_f1(partial, gold)
    r_disjoint = soft_f1(disjoint, gold)
    print(f"    overlapping candidate : SoftF1 = {r_partial['f1']:.4f}  "
          f"(SoftP={r_partial['soft_precision']:.4f}, SoftR={r_partial['soft_recall']:.4f})")
    print(f"    disjoint    candidate : SoftF1 = {r_disjoint['f1']:.4f}   "
          f"F1 = {span_f1(disjoint, gold)['f1']:.4f} (hard F1 = 0)")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    _run_examples()
