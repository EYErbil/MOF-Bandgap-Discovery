#!/bin/bash
#SBATCH --job-name=ensemble_v2
#SBATCH --output=logs/ensemble_v2_%j.out
#SBATCH --error=logs/ensemble_v2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=mid
#SBATCH --time=12:00:00

# =============================================================================
# Phase6 Enhanced Ensemble Report (v2)
# =============================================================================
#
# Runs the Phase6 ensemble report with all v2 enhancements:
#   --include_singles   : Evaluate each model individually (per-model top-K files)
#   --type_groups        : Add NN-only and ML-only sub-ensembles
#   --cross_type_pairs   : Add all NN x ML pair-wise combinations (3x2 = 6)
#   type_balanced_rrf    : 2-stage RRF ensuring 50/50 type balance
#
# These features address NN domination (3 NN vs 2 ML models) by providing
# type-balanced ensembles and per-model evaluation for comparison.
#
# OUTPUT:
#   ensemble_report_v2/
#     ├── agreement_heatmap_top25.png   — Readable labels (no truncation)
#     ├── singles/                      — Per-model rankings
#     ├── nn_only_*/                    — NN-only sub-ensembles
#     ├── ml_only_*/                    — ML-only sub-ensembles
#     └── *_type_balanced_rrf_*         — Type-balanced ensemble results
#
# PREREQUISITES:
#   - Step 6a-6c completed (embeddings + model predictions exist)
#
# USAGE:
#   sbatch scripts/optional/run_phase6_ensemble_v2.sh
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../config.sh"
load_modules
cd "$BASE_DIR"
mkdir -p logs

section "PHASE6 ENHANCED ENSEMBLE REPORT (v2)"

python discovery/phase6_ensemble_report.py \
    --base_dir "$BASE_DIR" \
    --auto_discover \
    --include_singles \
    --type_groups \
    --cross_type_pairs \
    --output_dir "$PHASE6_DATA/ensemble_report_v2"

section "COMPLETE"
echo "  Output: $PHASE6_DATA/ensemble_report_v2/"
