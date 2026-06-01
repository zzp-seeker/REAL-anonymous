#!/bin/bash

# ================= Configuration =================
CKPT_ROOT="XXX"
OUTPUT_ROOT="./eval_results"
DATASETS=("aime" "aime25" "math" "amc" "minerva" "olympiad_bench")
EXP_NAMES=(
    "XXX"
)
STEPS=(
    100
)
# =================================================

# Verify array lengths
if [ "${#EXP_NAMES[@]}" -ne "${#STEPS[@]}" ]; then
    echo "Error: Length of EXP_NAMES and STEPS must match."; exit 1
fi

# Execution Loop
for i in "${!EXP_NAMES[@]}"; do
    EXP="${EXP_NAMES[$i]}"
    STEP="${STEPS[$i]}"

    MODEL_PATH="${CKPT_ROOT}/${EXP}/actor/global_step_${STEP}"
    OUTPUT_DIR="${OUTPUT_ROOT}/${EXP}/step_${STEP}"

    echo ">>> Running Eval | Exp: ${EXP} | Step: ${STEP}"

    if [ ! -d "$MODEL_PATH" ]; then
        echo "  [Skip] Model not found: $MODEL_PATH"
        continue
    fi

    mkdir -p "$OUTPUT_DIR"

    bash ./scripts/eval/eval_model_8k.sh \
        --model "$MODEL_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --datasets aime aime25 math amc minerva olympiad_bench 
done
