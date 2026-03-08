#!/bin/bash
#SBATCH --job-name=discovery
#SBATCH --output=logs/06_discovery_%j.out
#SBATCH --error=logs/06_discovery_%j.err
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# =============================================================================
# STEP 6: Phase6 Discovery — Inference on New MOF Dataset (No Labels)
# =============================================================================
#
# Applies ALL trained models (from Steps 2-4) to a new, unlabeled MOF dataset
# to rank candidates for DFT validation. No ground-truth labels are needed.
#
# PIPELINE:
#   6a. Extract pretrained embeddings for new MOFs
#   6b. Run ML inference (load saved sklearn models, score new embeddings)
#   6c. Run NN inference (load checkpoints, forward pass on structures)
#   6d. Ensemble all predictions → consensus top-25 for DFT
#   6e. Agreement analysis (which MOFs appear in multiple models' top-K?)
#
# CONFIGURATION:
#   Edit NN_EXPERIMENTS and ML_METHODS below to match your trained models.
#
# OUTPUT:
#   phase6/inference_results/
#     ├── top25_for_DFT_rrf.txt      — Top 25 by Reciprocal Rank Fusion
#     ├── top25_for_DFT_rank_avg.txt — Top 25 by rank averaging
#     └── inference_predictions.csv  — All predictions
#   phase6/ensemble_report/
#     ├── phase6_ensemble_report.md  — Full report
#     ├── agreement_heatmap_top25.png
#     └── ...
#
# PREREQUISITES:
#   - Steps 02-03 completed (trained models exist)
#   - New MOF structures in data/phase6/ with .grid, .griddata16, .graphdata
#   - data/phase6/test_bandgaps_regression.json (CIF IDs, bandgap can be 0.0)
#
# USAGE:
#   sbatch scripts/06_run_discovery.sh
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
load_modules
cd "$BASE_DIR"
mkdir -p logs "$PHASE6_DATA/inference_results" "$PHASE6_DATA/ensemble_report"

# --- Configuration -----------------------------------------------------------
TOP_K=25

# NN experiments to use for inference (must have best_es-*.ckpt)
NN_EXPERIMENTS="exp364_fulltune exp370_seed2 exp371_seed3"

# ML methods to include (subdirs of embedding_classifiers with model.joblib)
ML_METHODS="extra_trees random_forest logistic_regression smote_extra_trees"

# Include kNN? (1=yes, 0=no)
USE_KNN=1

# =============================================================================
# STEP 6a: Extract Phase6 embeddings
# =============================================================================
section "STEP 6a: EXTRACT PHASE6 EMBEDDINGS"

python data_preparation/extract_phase6_embeddings_pretrained.py \
    --data_dir "$PHASE6_DATA" \
    --output_dir "$PHASE6_DATA/embedding_analysis"

PHASE6_NPZ="$PHASE6_DATA/embedding_analysis/Phase6_embeddings.npz"
echo "  Output: $PHASE6_NPZ"

# =============================================================================
# STEP 6b: ML inference (load saved models, predict on new embeddings)
# =============================================================================
section "STEP 6b: ML INFERENCE"

python src/predict_with_embedding_classifier.py \
    --embeddings_path "$PHASE6_NPZ" \
    --models_dir "$SKLEARN_DIR" \
    --output_dir "$PHASE6_DATA/ml_predictions"

ML_PRED_COUNT=$(find "$PHASE6_DATA/ml_predictions" -name "test_predictions.csv" 2>/dev/null | wc -l)
echo "  ML methods with predictions: $ML_PRED_COUNT"

# =============================================================================
# STEP 6c: NN inference (per experiment — run from experiment directory)
# =============================================================================
section "STEP 6c: NN INFERENCE"

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
    echo "  Inferring with $exp_name (checkpoint: $(basename $ckpt))..."
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

# =============================================================================
# STEP 6d: Ensemble → Top-K for DFT
# =============================================================================
section "STEP 6d: ENSEMBLE → TOP $TOP_K"

# Build prediction directories
PRED_DIRS=""
for m in $ML_METHODS; do
    d="$PHASE6_DATA/ml_predictions/$m"
    if [ -f "$d/test_predictions.csv" ]; then
        PRED_DIRS="$PRED_DIRS $d"
        echo "  [ML] $m"
    fi
done

if [ "$USE_KNN" -eq 1 ] && [ -f "$PHASE6_DATA/knn_predictions/test_predictions.csv" ]; then
    PRED_DIRS="$PRED_DIRS $PHASE6_DATA/knn_predictions"
    echo "  [kNN] knn_predictions"
fi

# Collect NN prediction CSVs
NN_CSV=""
for exp_name in $NN_EXPERIMENTS; do
    csv="$EXP_BASE/$exp_name/inference_predictions.csv"
    if [ -f "$csv" ]; then
        NN_CSV="$NN_CSV $csv"
        echo "  [NN] $exp_name"
    fi
done

python discovery/ensemble_phase6_predictions.py \
    --prediction_dirs $PRED_DIRS \
    --nn_predictions $NN_CSV \
    --output_dir "$PHASE6_DATA/inference_results" \
    --top_k "$TOP_K"

# =============================================================================
# STEP 6e: Full ensemble report + agreement analysis
# =============================================================================
section "STEP 6e: ENSEMBLE REPORT"

python discovery/phase6_ensemble_report.py \
    --base_dir "$BASE_DIR" \
    --auto_discover \
    --output_dir "$PHASE6_DATA/ensemble_report"

section "STEP 6 COMPLETE — PHASE6 DISCOVERY"
echo ""
echo "  Top $TOP_K candidates for DFT:"
echo "    RRF:      $PHASE6_DATA/inference_results/top${TOP_K}_for_DFT_rrf.txt"
echo "    Rank avg: $PHASE6_DATA/inference_results/top${TOP_K}_for_DFT_rank_avg.txt"
echo "  Ensemble report: $PHASE6_DATA/ensemble_report/phase6_ensemble_report.md"
