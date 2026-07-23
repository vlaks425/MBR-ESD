# MBR-ESD: MBR Decoding & SoftF1 for Error Span Detection

This repository accompanies our paper:

> **Minimum Bayes Risk Decoding for Error Span Detection in Reference-Free Automatic Machine Translation Evaluation**
> Boxuan Lyu, Haiyue Song, Hidetaka Kamigaito, Chenchen Ding, Hideki Tanaka, Masao Utiyama, Kotaro Funakoshi, Manabu Okumura.
> arXiv:2512.07540

> **Status.** The paper is currently **under review**. This repository provides
> the core contribution — the `SoftF1` utility — together with a clean,
> minimal **inference pipeline** (generation → reranking → evaluation) for the
> `Llama-3.3-70B` base model and our MBR-distilled model. Some peripheral
> experiment code (training/distillation, data preparation, ablations) is not
> included yet and may follow once the paper is finalized.

## What is Error Span Detection, and MBR over it?

Error Span Detection (ESD) asks a model to annotate the character spans of a
translation that contain errors, each with a severity (*major* or *minor*).
Given a source and a machine translation (reference-free / QE), the model emits
a JSON list of error spans.

Rather than trusting a single greedy decode (**MAP**), we sample many candidate
annotations and pick one by **Minimum Bayes Risk (MBR)** decoding: the candidate
that is, on average, most similar to all the others under a *utility function*.
The paper studies three utilities:

| Method | Utility | Level |
|---|---|---|
| **MBR-ScoreSim** | closeness of the sentence-level MQM score to the mean | sentence |
| **MBR-F1** | standard character-level span `F1` | span |
| **MBR-SoftF1** | our proposed `SoftF1` (below) | span |

We also **distill** MBR-SoftF1 behaviour into a model, so that a single greedy
(MAP) decode of the distilled model approximates the far more expensive MBR
decoding of the base model.

## SoftF1

The character-level `F1` used by the WMT QE Shared Task has a **"zero-credit
plateau"**: whenever two annotations share no error character, `F1` collapses to
`0` regardless of how close they actually are. This is especially damaging when
the consensus annotation is *empty* (the translation is deemed error-free): `F1`
then gives `0` to *every* non-empty candidate and cannot rank them.

**`SoftF1`** replaces the hard set overlap with a smooth, severity-weighted
distance between dense severity vectors. It keeps the good properties of `F1`
(bounded in `[0, 1]`, symmetric, exactly `1` on an exact match) while avoiding
the plateau. See the paper (§4.2) and Appendix A for the definition and proofs.

An annotation over an `L`-character translation is a severity vector `v ∈ R^L`
with `v^(i) = β·1[i ∈ major] + γ·1[i ∈ minor]` (defaults `β = 1.0`, `γ = 0.5`).
For a candidate `E_c` and support/gold `E_s`:

```
SoftP(E_c, E_s) = 1 - ||v_c - v_s||_1 / (L + ||v_c||_1 + eps)
SoftR(E_c, E_s) = 1 - ||v_c - v_s||_1 / (L + ||v_s||_1 + eps)
SoftF1(E_c, E_s) = harmonic_mean(SoftP, SoftR)
```

`L` normalizes the score into `[0, 1]`. `eps` (default `1.0`) is a smoothing
constant that keeps the score defined for the empty translations (`L = 0`)
present in the WMT data; for `L > 0` it has a small, monotone effect. This
matches the implementation used to produce the results in the paper.

`SoftF1` is used **both** as the MBR utility for reranking and as the headline
span-level evaluation metric.

## Models

- Base: [`meta-llama/Llama-3.3-70B-Instruct`](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
- MBR-distilled (ours): [`lyu-boxuan/Llama-3.3-70B-ESD-MBR-DPODistill`](https://huggingface.co/lyu-boxuan/Llama-3.3-70B-ESD-MBR-DPODistill)

## Repository layout

| File | Purpose |
|---|---|
| [`softf1.py`](softf1.py) | Core `SoftF1` (and `F1`) metric + error-span string processing. Standalone, dependency-free; run it to see worked examples. |
| [`generate.py`](generate.py) | vLLM generation of error-span candidates (MAP = greedy N=1; MBR = sampling N≫1). |
| [`rerank.py`](rerank.py) | Select one annotation per segment: `map`, `mbr_scoresim`, `mbr_f1`, `mbr_softf1`. |
| [`evaluate.py`](evaluate.py) | Span-level `F1` / `SoftF1` against human MQM annotations. |
| [`evaluate_meta.py`](evaluate_meta.py) | System/segment-level meta-evaluation (SPA, Acc_eq) via mt-metrics-eval. |
| [`run.sh`](run.sh) | End-to-end driver: generate → rerank → evaluate over language pairs. |
| [`examples/`](examples/) | Tiny toy dataset to smoke-test reranking + evaluation without a GPU. |
| [`data/`](data/) | Human MQM error spans (included); pointers to download WMT data. |

## Quickstart

**SoftF1 alone (no dependencies):**

```bash
python softf1.py          # runs worked examples
```

```python
from softf1 import annotate_spans, build_masks, soft_f1
hypo = "The cat sat on the moon."
cand = build_masks(annotate_spans([{"error_span": "moon", "severity": "major"}], hypo), len(hypo))
gold = build_masks(annotate_spans([{"error_span": "moon", "severity": "minor"}], hypo), len(hypo))
print(soft_f1(cand, gold))   # {'soft_precision': ..., 'soft_recall': ..., 'f1': ...}
```

**Reranking + evaluation without a GPU** (uses the toy data in `examples/`):

```bash
python rerank.py   --model_output examples/toy.en-de.gen.json \
                   --output_prefix examples/toy_out.en-de --method mbr_softf1
python evaluate.py --prediction_error examples/toy_out.en-de.error_spans.json \
                   --human_errors examples/toy_human.en-de.json \
                   --output_file examples/toy_eval.json
```

**Full pipeline** (needs GPUs + the WMT data, see [Data](#data)). `run.sh` runs
generation → reranking → span evaluation for `en-de`, `en-es`, `ja-zh`:

```bash
# Base model + MBR-SoftF1 (256 samples). Swap the method for mbr_f1 / mbr_scoresim.
bash run.sh meta-llama/Llama-3.3-70B-Instruct mbr_softf1 256 2.0 10

# Base model + MAP (single greedy decode):
bash run.sh meta-llama/Llama-3.3-70B-Instruct map 1 0.0 0

# Distilled model + MAP — the intended use: one greedy decode approximates MBR:
bash run.sh lyu-boxuan/Llama-3.3-70B-ESD-MBR-DPODistill map 1 0.0 0
```

Set `MTME_ROOT` to your WMT data location, `GPU` to the GPU ids (their count is
the tensor-parallel size), and `RUN_META=1` to also compute SPA / Acc_eq.

## Data

- **Included:** human MQM error spans at `data/wmt24/2024_<lp>.json`
  (`en-de`, `en-es`, `ja-zh`), used by `evaluate.py`.
- **Download separately:** the WMT sources / system outputs / references, from
  **mt-metrics-eval-v2** (see below). Not redistributed here.

See [`data/README.md`](data/README.md) for formats and the expected layout.

## Installation

We no longer have access to the original experiment machine, so we cannot
provide exact environment details. The good news is that the repo needs no
unusual libraries — only two external pieces, each installed by following its
own repository:

- **vLLM** (for `generate.py`): https://github.com/vllm-project/vllm
  (also pulls in `torch` / `transformers`; plus `pydantic` and `tqdm`).
- **mt-metrics-eval-v2** (for `evaluate_meta.py` and for the WMT data):
  https://github.com/google-research/mt-metrics-eval — the data download
  instructions are in that repository too.

`softf1.py`, `rerank.py` and `evaluate.py` use only the Python standard library.

## Citation

```bibtex
@article{lyu2025mbresd,
  title   = {Minimum Bayes Risk Decoding for Error Span Detection in Reference-Free Automatic Machine Translation Evaluation},
  author  = {Lyu, Boxuan and Song, Haiyue and Kamigaito, Hidetaka and Ding, Chenchen and Tanaka, Hideki and Utiyama, Masao and Funakoshi, Kotaro and Okumura, Manabu},
  journal = {arXiv preprint arXiv:2512.07540},
  year    = {2025}
}
```

## Questions

Issues and pull requests are welcome. You can also reach me at
**lyu.b.aa@m.titech.ac.jp**.

## License

Code in this repository is released under the [MIT License](LICENSE). The WMT
data and the `mt-metrics-eval` toolkit are licensed by their respective sources.
