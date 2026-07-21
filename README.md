# MBR-ESD: SoftF1 for Error Span Detection

This repository accompanies our paper:

> **Minimum Bayes Risk Decoding for Error Span Detection in Reference-Free Automatic Machine Translation Evaluation**
> Boxuan Lyu, Haiyue Song, Hidetaka Kamigaito, Chenchen Ding, Hideki Tanaka, Masao Utiyama, Kotaro Funakoshi, Manabu Okumura.
> arXiv:2512.07540

> **Note on the release.** The paper is currently **under review**, and we have
> not yet had the time to clean up and document the full experimental codebase.
> To make the most useful part available early, we are first releasing the
> **core contribution of the paper — the `SoftF1` utility function** — as a
> small, self-contained, dependency-free module. The remaining code (MBR
> decoding pipeline, WMT24 data preparation, knowledge distillation, evaluation
> scripts) will follow once the paper is finalized.

## What is SoftF1?

Error Span Detection (ESD) asks a model to annotate the character spans of a
translation that contain errors, together with a severity label (*major* or
*minor*). To compare two such annotations — e.g. a candidate against a support
hypothesis during MBR decoding, or against the human gold annotation during
evaluation — the WMT QE Shared Task uses a character-level `F1`.

That `F1` has a **"zero-credit plateau"**: whenever two annotations share no
error character, `F1` collapses to `0`, no matter how close the two annotations
actually are. This is especially damaging when the support/consensus annotation
is *empty* (the translation is deemed error-free): `F1` then assigns `0` to
*every* non-empty candidate and can no longer rank them.

**`SoftF1`** replaces the hard set overlap with a smooth, severity-weighted
distance between dense severity vectors, so it returns a continuous value that
reflects *how far apart* two annotations are. It keeps the good properties of
`F1` (bounded in `[0, 1]`, symmetric, exactly `1` on an exact match) while
avoiding the plateau. See the paper (§4.2) for the full definition and
Appendix A for the proofs.

Formally, an annotation over an `L`-character translation is a severity vector
`v ∈ R^L`, with `v^(i) = β·1[i ∈ major] + γ·1[i ∈ minor]` (defaults `β = 1.0`,
`γ = 0.5`). For a candidate `E_c` and support/gold `E_s`:

```
SoftP(E_c, E_s) = 1 - ||v_c - v_s||_1 / (L + ||v_c||_1 + eps)
SoftR(E_c, E_s) = 1 - ||v_c - v_s||_1 / (L + ||v_s||_1 + eps)
SoftF1(E_c, E_s) = harmonic_mean(SoftP, SoftR)
```

`L` normalizes the score into `[0, 1]`. `eps` (default `1.0`) is a smoothing
constant that keeps the score defined for the empty translations (`L = 0`)
present in the WMT data; for `L > 0` it has a small, monotone effect. This is
the implementation used to produce the results in the paper.

## Contents

- [`softf1.py`](softf1.py) — the `SoftF1` implementation, including:
  - error-span string processing (locating raw span text in a hypothesis,
    bucketing by severity, converting spans to `'0'/'1'` character masks);
  - the severity vector, `SoftP`, `SoftR`, and `SoftF1`;
  - the standard character-level `F1` (`span_f1`), kept for reference/contrast;
  - a `main` block that runs several worked examples.

## Usage

No dependencies beyond the Python standard library (Python 3.7+).

Run the built-in examples:

```bash
python softf1.py
```

Use it as a library:

```python
from softf1 import annotate_spans, build_masks, soft_f1, softf1_from_errors

hypo = "The cat sat on the moon."
L = len(hypo)

# Raw model/human annotations: substring + severity.
candidate = [{"error_span": "moon", "severity": "major"}]
support   = [{"error_span": "moon", "severity": "minor"}]

# Option A: resolve span text -> indexed spans -> masks, then score.
cand_masks = build_masks(annotate_spans(candidate, hypo), L)
supp_masks = build_masks(annotate_spans(support, hypo), L)
print(soft_f1(cand_masks, supp_masks))
# {'soft_precision': ..., 'soft_recall': ..., 'f1': ...}

# Option B: if your spans already carry character offsets
# (error_span_start / error_span_end / severity), score in one call.
print(softf1_from_errors(candidate, support, hypo_length=L))
```

## Citation

If you use this code, please cite our paper:

```bibtex
@article{lyu2025mbresd,
  title   = {Minimum Bayes Risk Decoding for Error Span Detection in Reference-Free Automatic Machine Translation Evaluation},
  author  = {Lyu, Boxuan and Song, Haiyue and Kamigaito, Hidetaka and Ding, Chenchen and Tanaka, Hideki and Utiyama, Masao and Funakoshi, Kotaro and Okumura, Manabu},
  journal = {arXiv preprint arXiv:2512.07540},
  year    = {2025}
}
```

## License

Released under the [MIT License](LICENSE).
