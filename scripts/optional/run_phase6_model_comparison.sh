#!/bin/bash
#SBATCH --job-name=model_cmp
#SBATCH --output=logs/model_cmp_%j.out
#SBATCH --error=logs/model_cmp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=mid
#SBATCH --time=04:00:00
#SBATCH --mem=16G

# =============================================================================
# Phase6 Model Comparison — NN vs ML Investigation
# =============================================================================
#
# Runs the UMAP-based model comparison script to investigate where each model's
# top-K candidates sit in embedding space, and whether NN and ML models agree.
#
# PRODUCES:
#   model_comparison/
#     ├── per_model_grid_top25.png         — Each model's top-K on UMAP
#     ├── nn_vs_ml_consensus_top25.png     — NN-union vs ML-union overlap
#     ├── nn_vs_ml_sidebyside_top25.png    — Vote density per type
#     ├── jaccard_heatmap_top25.png        — Pairwise Jaccard with full labels
#     ├── model_comparison_report_top25.txt — Detailed text report
#     └── umap_coords_cache.npz            — Cached UMAP coordinates
#
# FLAGS:
#   --top_k N       : Number of top candidates per model (default: 25)
#   --all_ml        : Include all 31 ML classifiers (default: smote_extra_trees + smote_random_forest)
#   --embeddings_npz: Path to Phase6 embeddings .npz file
#
# PREREQUISITES:
#   - Phase6 embeddings extracted (Step 6a)
#   - Model predictions exist (Steps 6b-6c)
#
# USAGE:
#   sbatch scripts/optional/run_phase6_model_comparison.sh
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../config.sh"
load_modules
cd "$BASE_DIR"
mkdir -p logs

section "PHASE6 MODEL COMPARISON (NN vs ML)"

python discovery/plot_phase6_model_comparison.py \
    --embeddings_npz "$PHASE6_DATA/embedding_analysis/Phase6_embeddings.npz" \
    --top_k 25

section "COMPLETE"
echo "  Output: model_comparison/"
