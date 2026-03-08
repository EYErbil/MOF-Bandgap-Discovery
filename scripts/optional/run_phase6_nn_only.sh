#!/bin/bash
#SBATCH --job-name=phase6_nn
#SBATCH --output=logs/phase6_nn_%j.out
#SBATCH --error=logs/phase6_nn_%j.err
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# =============================================================================
# OPTIONAL: Phase6 NN-Only Inference
# =============================================================================
#
# Runs ONLY the neural network models (trained MOFTransformers) on a new
# unlabeled MOF dataset. Requires a GPU. Use this when you want to run
# NN inference independently — for example, after training a new experiment
# or if ML inference was already done separately.
#
# WHAT IT DOES:
#   For each experiment in NN_EXPERIMENTS:
#     1. cd into the experiment directory (where the checkpoint lives)
#     2. Run discovery/run_inference_from_cwd.py with --data_dir pointing
#        to the Phase6 data
#     3. Produces: inference_predictions.csv + top{K}_for_DFT.txt
#
# CONFIGURATION:
#   Edit NN_EXPERIMENTS below to choose which trained NN models to use.
#   Each experiment must have a best_es-*.ckpt checkpoint from Step 02.
#
# PREREQUISITES:
#   - Step 02 completed (NN experiments have checkpoints)
#   - New MOF structures in data/phase6/ (.grid, .griddata16, .graphdata)
#   - data/phase6/test_bandgaps_regression.json
#
# USAGE:
#   sbatch scripts/optional/run_phase6_nn_only.sh
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/config.sh"
load_modules
cd "$BASE_DIR"
mkdir -p logs "$PHASE6_DATA/inference_results"

# ---- Configuration: which NN experiments to use ----------------------------
# Edit this list to match your trained experiments.
# Each must be a subdirectory of experiments/ with a best_es-*.ckpt file.
NN_EXPERIMENTS="exp364_fulltune exp370_seed2 exp371_seed3"

# How many top candidates to report per model
TOP_K=25

# =============================================================================
# NN INFERENCE
# =============================================================================
section "PHASE6 NN INFERENCE"
echo "  Experiments: $NN_EXPERIMENTS"
echo "  Data dir:    $PHASE6_DATA"
echo "  Top-K:       $TOP_K"

for exp_name in $NN_EXPERIMENTS; do
    exp_dir="$EXP_BASE/$exp_name"

    if [ ! -d "$exp_dir" ]; then
        echo "  WARNING: $exp_dir does not exist. Skipping."
        continue
    fi

    ckpt=$(ls "$exp_dir"/best_es-*.ckpt 2>/dev/null | head -1)
    if [ -z "$ckpt" ]; then
        echo "  WARNING: No checkpoint in $exp_dir. Skipping."
        continue
    fi

    section "INFERRING: $exp_name"
    echo "  Checkpoint: $(basename $ckpt)"

    cd "$exp_dir"
    python "$BASE_DIR/discovery/run_inference_from_cwd.py" \
        --data_dir "$PHASE6_DATA" \
        --top_k "$TOP_K"
    cd "$BASE_DIR"

    if [ -f "$exp_dir/inference_predictions.csv" ]; then
        echo "  OK: $exp_dir/inference_predictions.csv"
    else
        echo "  WARNING: inference_predictions.csv not produced for $exp_name"
    fi
done

section "NN INFERENCE COMPLETE"
for exp_name in $NN_EXPERIMENTS; do
    csv="$EXP_BASE/$exp_name/inference_predictions.csv"
    if [ -f "$csv" ]; then
        echo "  [OK]   $exp_name → $(wc -l < "$csv") predictions"
    else
        echo "  [MISS] $exp_name"
    fi
done
