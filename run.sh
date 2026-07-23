#!/bin/bash
# End-to-end ESD pipeline: generate candidates -> rerank (MAP / MBR) -> evaluate.
#
# Usage:
#   bash run.sh <model_path> <method> <num_hypotheses> [temperature] [top_k]
#
# The three settings reported in the paper:
#
#   # Base model + MBR-SoftF1 (also try mbr_f1 / mbr_scoresim), 256 samples:
#   bash run.sh meta-llama/Llama-3.3-70B-Instruct mbr_softf1 256 2.0 10
#
#   # Base model + MAP (single greedy decode):
#   bash run.sh meta-llama/Llama-3.3-70B-Instruct map 1 0.0 0
#
#   # Distilled model + MAP (the intended use: one greedy decode approximates MBR):
#   bash run.sh lyu-boxuan/Llama-3.3-70B-ESD-MBR-DPODistill map 1 0.0 0
#
# Data layout (see README):
#   - MTME_ROOT points at a local copy of mt-metrics-eval-v2 (download it from
#     github.com/google-research/mt-metrics-eval). Default: data/mt-metrics-eval-v2
#   - Human error spans ship in this repo at data/wmt24/2024_<lp>.json
set -e

MODEL_PATH=${1:?"model path or HF id"}
METHOD=${2:?"one of: map | mbr_scoresim | mbr_f1 | mbr_softf1"}
NUM_HYP=${3:?"number of candidates per segment (1 for MAP, e.g. 256 for MBR)"}
TEMPERATURE=${4:-0.0}
TOP_K=${5:-0}
TOP_P=${TOP_P:-1.0}

MTME_ROOT=${MTME_ROOT:-data/mt-metrics-eval-v2}
GPU=${GPU:-0}
LANG_PAIRS=${LANG_PAIRS:-"en-de en-es ja-zh"}

# Short, filesystem-safe tag for the model (last path component).
MODEL_TAG=$(basename "$MODEL_PATH")
GEN_DIR="outputs/$MODEL_TAG/gen/n${NUM_HYP}_t${TEMPERATURE}_k${TOP_K}"
RUN_DIR="outputs/$MODEL_TAG/$METHOD/n${NUM_HYP}"
mkdir -p "$GEN_DIR" "$RUN_DIR"

for LP in $LANG_PAIRS; do
    echo "==== $LP | model=$MODEL_TAG | method=$METHOD | N=$NUM_HYP ===="
    SRC_FILE="$MTME_ROOT/wmt24/sources/$LP.txt"
    HYPO_DIR="$MTME_ROOT/wmt24/system-outputs/$LP"
    HUMAN_SPANS="data/wmt24/2024_$LP.json"

    # 1) Generate candidates (reused across reranking methods with the same N/temp).
    GEN_FILE="$GEN_DIR/gen.$LP.gen.json"
    if [ -f "$GEN_FILE" ]; then
        echo "  [generate] reusing $GEN_FILE"
    else
        python generate.py \
            --model_path "$MODEL_PATH" \
            --src_file "$SRC_FILE" \
            --hypo_dir "$HYPO_DIR" \
            --lang_pair "$LP" \
            --output_prefix "$GEN_DIR/gen" \
            --num_hypotheses "$NUM_HYP" \
            --temperature "$TEMPERATURE" \
            --top_k "$TOP_K" \
            --top_p "$TOP_P" \
            --gpu "$GPU"
    fi

    # 2) Rerank (MAP or MBR utility).
    python rerank.py \
        --model_output "$GEN_FILE" \
        --output_prefix "$RUN_DIR/$LP" \
        --method "$METHOD"

    # 3) Span-level evaluation (F1 / SoftF1) against human MQM annotations.
    python evaluate.py \
        --prediction_error "$RUN_DIR/$LP.error_spans.json" \
        --human_errors "$HUMAN_SPANS" \
        --output_file "$RUN_DIR/$LP.span_scores.json"

    # 4) (Optional) system/segment-level meta-evaluation (SPA / Acc_eq).
    #    Requires the mt_metrics_eval package + WMT data. Enable with RUN_META=1.
    if [ "${RUN_META:-0}" = "1" ]; then
        python evaluate_meta.py \
            --lp "$LP" \
            --metric_file "$RUN_DIR/$LP.seg.scores" \
            --human_file "$MTME_ROOT/wmt24/human-scores/$LP.mqm.seg.score" \
            --output_file "$RUN_DIR/$LP.meta_scores.json"
    fi
done

echo "Done. Per-language results under $RUN_DIR/"
