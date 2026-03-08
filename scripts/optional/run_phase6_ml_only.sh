#!/bin/bash
#SBATCH --job-name=phase6_ml
#SBATCH --output=logs/phase6_ml_%j.out
#SBATCH --error=logs/phase6_ml_%j.err
#SBATCH --partition=mid
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# =============================================================================
# OPTIONAL: Phase6 ML-Only Inference
# =============================================================================
#
# Runs ONLY the ML classifiers (sklearn models) on a new unlabeled dataset.
# No GPU needed. Use this if you want to quickly score new MOFs with the
# saved ML models before committing to a full NN inference run.
#
# This is equivalent to Steps 6a + 6b from 06_run_discovery.sh, isolated
# for convenience when you want to iterate on ML models independently.
#
# WHAT IT DOES:
#   1. Extracts pretrained embeddings for the new MOFs
#   2. Loads saved sklearn classifiers (from Step 03)
#   3. Scores every new MOF with each ML method
#   4. Optionally includes NN predictions if they already exist
#
# PREREQUISITES:
#   - Step 03 completed (sklearn models saved in embedding_classifiers/)
#   - New MOF structures in data/phase6/ (.grid, .griddata16, .graphdata)
#   - data/phase6/test_bandgaps_regression.json (CIF IDs; bandgap can be 0.0)
#
# USAGE:
#   sbatch scripts/optional/run_phase6_ml_only.sh
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/config.sh"
load_modules
cd "$BASE_DIR"
mkdir -p logs "$PHASE6_DATA/embedding_analysis" "$PHASE6_DATA/ml_predictions"

# ---- Step A: Extract embeddings for new MOFs --------------------------------
section "EXTRACT PHASE6 EMBEDDINGS"

python data_preparation/extract_phase6_embeddings_pretrained.py \
    --data_dir "$PHASE6_DATA" \
    --output_dir "$PHASE6_DATA/embedding_analysis"

PHASE6_NPZ="$PHASE6_DATA/embedding_analysis/Phase6_embeddings.npz"
echo "  Output: $PHASE6_NPZ"

# ---- Step B: ML inference ---------------------------------------------------
section "ML INFERENCE ON PHASE6 DATA"

python src/predict_with_embedding_classifier.py \
    --embeddings_path "$PHASE6_NPZ" \
    --models_dir "$SKLEARN_DIR" \
    --output_dir "$PHASE6_DATA/ml_predictions"

ML_PRED_COUNT=$(find "$PHASE6_DATA/ml_predictions" -name "test_predictions.csv" 2>/dev/null | wc -l)
echo "  ML methods with predictions: $ML_PRED_COUNT"

# ---- Optional: run phase6_discovery.py if NN predictions exist already ------
NN_CSV="$PHASE6_DATA/inference_results/inference_predictions.csv"
if [ -f "$NN_CSV" ]; then
    section "RUNNING ML + NN DISCOVERY (NN predictions found)"
    python discovery/phase6_discovery.py \
        --phase6_dir "$PHASE6_DATA" \
        --embeddings_path "$PHASE6_NPZ" \
        --ml_models_dir "$SKLEARN_DIR" \
        --output_dir "$PHASE6_DATA/discovery_output" \
        --rrf_k 60 \
        --nn_predictions "$NN_CSV"
else
    section "RUNNING ML-ONLY DISCOVERY (no NN predictions yet)"
    python discovery/phase6_discovery.py \
        --phase6_dir "$PHASE6_DATA" \
        --embeddings_path "$PHASE6_NPZ" \
        --ml_models_dir "$SKLEARN_DIR" \
        --output_dir "$PHASE6_DATA/discovery_output" \
        --rrf_k 60
fi

section "ML-ONLY INFERENCE COMPLETE"
echo "  Predictions: $PHASE6_DATA/ml_predictions/"
echo "  Discovery output: $PHASE6_DATA/discovery_output/"
