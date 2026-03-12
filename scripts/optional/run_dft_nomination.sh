#!/bin/bash
#SBATCH --job-name=dft_nominate
#SBATCH --output=logs/DFT_Nomination_%j.out
#SBATCH --error=logs/DFT_Nomination_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=mid
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --mail-type=ALL

# =============================================================================
# Step 7: DFT Candidate Nomination — Multi-Perspective Consensus
# =============================================================================
#
# Runs 28 balanced ensemble perspectives (individual models, 1NN+1ML pairs,
# 2NN+2ML quads, full balanced) across RRF and rank-averaging, then selects
# the top-25 structures for DFT bandgap calculation via confidence-tiered
# consensus voting.
#
# PERSPECTIVE GROUPS (28 total):
#   A  Individuals              5  (3 NN + 2 ML ranked by own scores)
#   B  1NN+1ML pairs, RRF       6  (3×2 combos)
#   C  1NN+1ML pairs, rank_avg  6
#   D  2NN+2ML quads, RRF       3  (C(3,2) NN pairs × 1 ML pair)
#   E  2NN+2ML quads, rank_avg  3
#   F  2NN+2ML quads, type_bal  3
#   G  Full 3NN+2ML, balanced   2
#
# CONFIDENCE TIERS:
#   Tier 1 (Very High): >= 75% of perspectives agree  (>= 21/28 votes)
#   Tier 2 (High):      >= 50% of perspectives agree  (>= 14/28 votes)
#   Tier 3 (Moderate):  >= 25% of perspectives agree  (>=  7/28 votes)
#
# OUTPUT: DFT-subset-Nomination/
#   ├── FINAL_DFT_TOP25.txt         (the money file — 25 CIF IDs)
#   ├── FINAL_DFT_TOP25.csv         (with votes, tiers, avg rank)
#   ├── full_consensus_ranking.csv  (all ~9.5K structures ranked)
#   ├── nomination_report.md        (full methodology + results)
#   ├── plots/                      (7 publication-quality figures)
#   ├── individual_models/          (per-model top-25 lists)
#   ├── balanced_ensembles/         (per-combo top-25 lists)
#   └── perspective_summary.json    (machine-readable perspective data)
#
# PREREQUISITES:
#   - Step 6a-6c completed (embeddings extracted + model predictions exist)
#   - i.e. experiments/exp3*/inference_predictions.csv +
#     embedding_classifiers/strategy_d_farthest_point/smote_*/test_predictions.csv
#
# USAGE:
#   sbatch scripts/optional/run_dft_nomination.sh
#
# To skip UMAP (faster, no umap-learn dependency):
#   Uncomment the --skip_umap flag below.
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../config.sh"
load_modules
cd "$BASE_DIR"
mkdir -p logs

OUTPUT_DIR="$PHASE6_DATA/DFT-subset-Nomination"

section "DFT CANDIDATE NOMINATION — MULTI-PERSPECTIVE CONSENSUS"

python discovery/nominate_dft_candidates.py \
  --base_dir "$PHASE6_DATA" \
  --output_dir "$OUTPUT_DIR" \
  --top_k 25 \
  --rrf_k 60 \
  --tier1_pct 0.75 \
  --tier2_pct 0.50 \
  --tier3_pct 0.25
  # --skip_umap  # Uncomment to skip UMAP (saves ~5 min if umap-learn not installed)

section "COMPLETE"
echo ""
echo "  FINAL TOP 25 FOR DFT:"
echo "  ====================="
cat "${OUTPUT_DIR}/FINAL_DFT_TOP25.txt"
echo ""
echo "  Output: ${OUTPUT_DIR}/"
echo "  Report: ${OUTPUT_DIR}/nomination_report.md"
