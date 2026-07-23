# Toy example (no GPU / no WMT data required)

This directory contains a tiny synthetic dataset in the exact formats the
pipeline uses, so you can smoke-test **reranking + evaluation** without a GPU or
the full WMT data. (Generation itself still needs the model + GPUs.)

Files:
- `toy.en-de.gen.json` — like `generate.py`'s output (2 systems × 3 segments × 3 candidates).
- `toy_human.en-de.json` — like `data/wmt24/2024_en-de.json` (human error spans).
- `make_toy_data.py` — regenerates the two files above.

Run reranking + evaluation for any method:

```bash
cd ..   # repo root

python rerank.py \
    --model_output examples/toy.en-de.gen.json \
    --output_prefix examples/toy_out.en-de \
    --method mbr_softf1            # or: map | mbr_scoresim | mbr_f1

python evaluate.py \
    --prediction_error examples/toy_out.en-de.error_spans.json \
    --human_errors examples/toy_human.en-de.json \
    --output_file examples/toy_eval.json
```
