#!/bin/bash
#SBATCH --job-name=phase6_ens
#SBATCH --output=logs/phase6_ensemble_custom_%j.out
#SBATCH --error=logs/phase6_ensemble_custom_%j.err
#SBATCH --partition=mid
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# =============================================================================
# OPTIONAL: Phase6 Custom Ensemble — Pick Your Own Model Combination
# =============================================================================
#
# This script lets you build a CUSTOM ensemble on Phase6 (unlabeled) data
# by choosing exactly which ML methods and NN experiments to include.
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  HOW TO USE:                                                           │
# │  1. Edit ML_MODELS below to list the ML methods you want              │
# │  2. Edit NN_EXPERIMENTS to list the NN experiments to include         │
# │  3. Set USE_NN=1 to include NN predictions, or USE_NN=0 for ML only  │
# │  4. Submit: sbatch scripts/optional/run_phase6_ensemble_custom.sh     │
# └─────────────────────────────────────────────────────────────────────────┘
#
# AVAILABLE ML METHODS (subdirectories of embedding_classifiers/):
#   extra_trees, random_forest, logistic_regression, svm_rbf, gradient_boosting,
#   xgboost, lda, mahalanobis, gmm, isolation_forest,
#   smote_random_forest, smote_extra_trees, smote_logistic_regression,
#   two_stage_knn_et, feature_selected_*
#
# AVAILABLE NN EXPERIMENTS (subdirectories of experiments/):
#   exp364_fulltune, exp370_seed2, exp371_seed3
#
# The output directory is auto-named based on the models you select.
# Example: ensemble_results/custom_extra_trees_smote_extra_trees_nn/
#
# PREREQUISITES:
#   - ML inference (run_phase6_ml_only.sh or 06_run_discovery.sh) completed
#   - If USE_NN=1: NN inference (run_phase6_nn_only.sh or 06) completed
#
# USAGE:
#   sbatch scripts/optional/run_phase6_ensemble_custom.sh
#   # or interactively:
#   bash scripts/optional/run_phase6_ensemble_custom.sh 2>&1 | tee custom_ensemble.log
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/config.sh"
load_modules
cd "$BASE_DIR"
mkdir -p logs

# =============================================================================
# ★ EDIT THESE VARIABLES TO CUSTOMIZE YOUR ENSEMBLE ★
# =============================================================================

# Which ML methods to include (must have inference_predictions.csv in
# data/phase6/discovery_output/individual/<method>/)
ML_MODELS="extra_trees smote_extra_trees smote_random_forest"

# Which NN experiments to include (must have inference_predictions.csv in
# experiments/<name>/)
NN_EXPERIMENTS="exp364_fulltune exp370_seed2"

# Include NN predictions? (1 = yes, 0 = ML only)
USE_NN=1

# RRF parameter (higher k = more weight to lower-ranked items)
RRF_K=60

# =============================================================================
# BUILD PREDICTION DIRECTORY LIST
# =============================================================================
section "BUILDING CUSTOM ENSEMBLE"

PRED_DIRS=""
COMBO_PARTS=""

echo "  ML models:"
for m in $ML_MODELS; do
    d="$PHASE6_DATA/discovery_output/individual/$m"
    if [ -f "$d/inference_predictions.csv" ]; then
        PRED_DIRS="$PRED_DIRS $d"
        COMBO_PARTS="${COMBO_PARTS}_${m}"
        echo "    [OK]   $m"
    else
        echo "    [MISS] $m — no inference_predictions.csv found"
        echo "           Expected at: $d/inference_predictions.csv"
        echo "           Run ML inference first (06_run_discovery.sh or run_phase6_ml_only.sh)"
    fi
done

NN_CSV_ARGS=""
if [ "$USE_NN" -eq 1 ]; then
    echo ""
    echo "  NN experiments:"
    for exp_name in $NN_EXPERIMENTS; do
        csv="$EXP_BASE/$exp_name/inference_predictions.csv"
        if [ -f "$csv" ]; then
            NN_CSV_ARGS="$NN_CSV_ARGS $csv"
            COMBO_PARTS="${COMBO_PARTS}_${exp_name}"
            echo "    [OK]   $exp_name"
        else
            echo "    [MISS] $exp_name — no inference_predictions.csv"
            echo "           Run NN inference first (06_run_discovery.sh or run_phase6_nn_only.sh)"
        fi
    done
fi

# Auto-name the output directory
COMBO_NAME="custom${COMBO_PARTS}"
OUTPUT_DIR="$PHASE6_DATA/ensemble_results/$COMBO_NAME"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "  Output directory: $OUTPUT_DIR"

# =============================================================================
# RUN ENSEMBLE
# =============================================================================
section "RUNNING ENSEMBLE: $COMBO_NAME"

EXTRA_ARGS=""
if [ -n "$NN_CSV_ARGS" ]; then
    EXTRA_ARGS="--nn_predictions $NN_CSV_ARGS"
fi

python discovery/ensemble_phase6_predictions.py \
    --prediction_dirs $PRED_DIRS \
    $EXTRA_ARGS \
    --output_dir "$OUTPUT_DIR" \
    --top_k 25 50 100 \
    --rrf_k "$RRF_K"

section "CUSTOM ENSEMBLE COMPLETE"
echo "  Results:  $OUTPUT_DIR/"
echo "  Top 25:   $OUTPUT_DIR/top25_for_DFT_rrf.txt"
echo "  Top 50:   $OUTPUT_DIR/top50_for_DFT_rrf.txt"
echo "  Top 100:  $OUTPUT_DIR/top100_for_DFT_rrf.txt"
