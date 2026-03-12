#!/usr/bin/env python3
"""
DFT Candidate Nomination via Multi-Perspective Consensus
=========================================================

Systematically evaluates **28 balanced ensemble perspectives** (individual models,
1NN+1ML pairs, 2NN+2ML quads, full balanced) across RRF and rank-averaging, then
selects the top-25 structures for DFT bandgap calculations using a confidence-tiered
consensus vote.

Design rationale
----------------
- **Only balanced ensembles** are used (equal NN : ML ratio per combo) to avoid type
  domination (the Jaccard = 0.0 problem between NN and ML top-25 lists).
- Multiple fusion methods (RRF, rank_avg, type_balanced_rrf) test robustness to
  aggregation choice.  Score-based averaging is excluded because normalising
  regression bandgaps against classification probabilities is unreliable.
- Structures that survive the majority of these 28 independent views are the safest
  DFT bets: they are flagged by diverse model seeds, paradigms, and combinatorics.

Perspective groups (28 total)
-----------------------------
  A  Individuals            5  (3 NN + 2 ML ranked by their own scores)
  B  1NN + 1ML pairs, RRF   6  (3 x 2 combos)
  C  1NN + 1ML pairs, rank_avg  6
  D  2NN + 2ML quads, RRF   3  (C(3,2) NN pairs x 1 ML pair)
  E  2NN + 2ML quads, rank_avg  3
  F  2NN + 2ML quads, type_balanced_rrf  3
  G  Full 3NN + 2ML, type_balanced_rrf + rank_avg  2

Nomination logic
----------------
  1. Each perspective produces a top-K ranking.
  2. Vote count = how many of 28 perspectives place a structure in their top-K.
  3. Confidence tiers: Tier 1 >= 75 %, Tier 2 >= 50 %, Tier 3 >= 25 %.
  4. Auto-include Tier 1 + 2; fill from Tier 3 by (votes desc, avg_rank asc).
  5. All thresholds are CLI-configurable.

Usage
-----
  python nominate_dft_candidates.py --base_dir . --output_dir DFT-subset-Nomination
  python nominate_dft_candidates.py --base_dir . --output_dir DFT-subset-Nomination \\
      --top_k 25 --tier1_pct 0.75 --tier2_pct 0.50 --tier3_pct 0.25 --rrf_k 60
"""

import os
import sys
import csv
import json
import argparse
import warnings
from itertools import combinations
from collections import defaultdict, OrderedDict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Ensure we can import from the same directory
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from ensemble_phase6_predictions import (
    load_predictions_from_csv,
    find_predictions_csv,
    reciprocal_rank_fusion,
    rank_averaging,
    score_to_rank,
    infer_model_type,
    type_balanced_rrf,
)

# ---------------------------------------------------------------------------
# Matplotlib setup (Agg for headless cluster)
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TIER_COLORS = {"Tier 1": "#2ecc71", "Tier 2": "#f39c12", "Tier 3": "#e67e22",
               "Below": "#95a5a6"}
GROUP_COLORS = {"A-Individual": "#3498db", "B-Pair-RRF": "#e74c3c",
                "C-Pair-RankAvg": "#e67e22", "D-Quad-RRF": "#9b59b6",
                "E-Quad-RankAvg": "#1abc9c", "F-Quad-TypeBal": "#2c3e50",
                "G-Full": "#c0392b"}

NN_MODELS = ["exp364", "exp370", "exp371"]
ML_MODELS = ["smote_extra_trees", "smote_random_forest"]

# ============================================================================
# 1. MODEL LOADING
# ============================================================================

def discover_and_load_models(base_dir):
    """
    Find the 5 known models (3 NN + 2 ML) and load their predictions.

    Returns
    -------
    models : dict[str, dict[str, float]]
        {short_name: {cif_id: score}}, all lower-is-better.
    test_cids : list[str]
        Sorted intersection of CIF IDs across all models.
    model_types : dict[str, str]
        {short_name: 'NN' | 'ML'}.
    """
    base_dir = os.path.abspath(base_dir)
    experiments_dir = os.path.join(base_dir, "experiments")
    clf_base = os.path.join(base_dir, "embedding_classifiers",
                            "strategy_d_farthest_point")

    models = {}
    model_types = {}

    # --- NN models ---
    for short in NN_MODELS:
        found = False
        if os.path.isdir(experiments_dir):
            for dname in sorted(os.listdir(experiments_dir)):
                if dname.startswith(short):
                    csv_path = find_predictions_csv(
                        os.path.join(experiments_dir, dname))
                    if csv_path:
                        preds = load_predictions_from_csv(csv_path)
                        if preds:
                            models[short] = preds
                            model_types[short] = "NN"
                            found = True
                            break
        if not found:
            print(f"  [WARN] NN model '{short}' not found under {experiments_dir}")

    # --- ML models ---
    for short in ML_MODELS:
        csv_path = find_predictions_csv(os.path.join(clf_base, short))
        if csv_path:
            preds = load_predictions_from_csv(csv_path)
            if preds:
                models[short] = preds
                model_types[short] = "ML"
            else:
                print(f"  [WARN] ML model '{short}' has no predictions")
        else:
            print(f"  [WARN] ML model '{short}' not found under {clf_base}")

    if not models:
        raise FileNotFoundError(
            f"No model predictions found under {base_dir}. "
            "Expected experiments/exp364*/inference_predictions.csv and "
            "embedding_classifiers/strategy_d_farthest_point/smote_*/test_predictions.csv"
        )

    # Intersection of CIF IDs
    cid_sets = [set(m.keys()) for m in models.values()]
    common = cid_sets[0]
    for s in cid_sets[1:]:
        common &= s
    test_cids = sorted(common)

    print(f"  Loaded {len(models)} models: "
          f"{sum(1 for t in model_types.values() if t == 'NN')} NN + "
          f"{sum(1 for t in model_types.values() if t == 'ML')} ML")
    print(f"  Common CIFs: {len(test_cids)}")

    return models, test_cids, model_types


# ============================================================================
# 2. PERSPECTIVE GENERATION
# ============================================================================

def _ranked_top_k(scores_dict, k):
    """Return the top-k CIF IDs (lowest score = best) and full ranking dict."""
    ranked = sorted(scores_dict.keys(), key=lambda c: scores_dict[c])
    return ranked[:k], {cid: rank for rank, cid in enumerate(ranked, 1)}


def build_perspectives(models, test_cids, model_types, top_k=25, rrf_k=60):
    """
    Build all 28 balanced perspectives.

    Returns list of dicts, each with keys:
        name, group, models_used, method, top_k_cids, full_ranking
    """
    perspectives = []
    nn_names = [n for n in models if model_types[n] == "NN"]
    ml_names = [n for n in models if model_types[n] == "ML"]

    # --- Group A: Individuals (5) ---
    for name in list(nn_names) + list(ml_names):
        scores = {cid: models[name].get(cid, 1e18) for cid in test_cids}
        top_cids, full_rank = _ranked_top_k(scores, top_k)
        perspectives.append({
            "name": f"{name}_individual",
            "group": "A-Individual",
            "models_used": [name],
            "method": "individual",
            "top_k_cids": top_cids,
            "full_ranking": full_rank,
        })

    # --- Group B: 1NN+1ML pairs, RRF (6) ---
    for nn in nn_names:
        for ml in ml_names:
            sub = {nn: models[nn], ml: models[ml]}
            fused = reciprocal_rank_fusion(sub, test_cids, k=rrf_k,
                                           lower_is_better=True)
            top_cids, full_rank = _ranked_top_k(fused, top_k)
            perspectives.append({
                "name": f"{nn}+{ml}_rrf",
                "group": "B-Pair-RRF",
                "models_used": [nn, ml],
                "method": "rrf",
                "top_k_cids": top_cids,
                "full_ranking": full_rank,
            })

    # --- Group C: 1NN+1ML pairs, rank_avg (6) ---
    for nn in nn_names:
        for ml in ml_names:
            sub = {nn: models[nn], ml: models[ml]}
            fused = rank_averaging(sub, test_cids, lower_is_better=True)
            top_cids, full_rank = _ranked_top_k(fused, top_k)
            perspectives.append({
                "name": f"{nn}+{ml}_rank_avg",
                "group": "C-Pair-RankAvg",
                "models_used": [nn, ml],
                "method": "rank_avg",
                "top_k_cids": top_cids,
                "full_ranking": full_rank,
            })

    # --- Groups D/E/F: 2NN+2ML quads (9) ---
    # Only 1 ML pair (both ML models), enumerate NN pairs
    if len(ml_names) == 2:
        ml_pair = ml_names  # both
        for nn_pair in combinations(nn_names, 2):
            combo_names = list(nn_pair) + list(ml_pair)
            sub = {n: models[n] for n in combo_names}
            sub_types = {n: model_types[n] for n in combo_names}
            nn_tag = "+".join(nn_pair)
            ml_tag = "+".join(ml_pair)
            tag = f"{nn_tag}+{ml_tag}"

            # D: RRF
            fused = reciprocal_rank_fusion(sub, test_cids, k=rrf_k,
                                           lower_is_better=True)
            top_cids, full_rank = _ranked_top_k(fused, top_k)
            perspectives.append({
                "name": f"{tag}_rrf",
                "group": "D-Quad-RRF",
                "models_used": combo_names,
                "method": "rrf",
                "top_k_cids": top_cids,
                "full_ranking": full_rank,
            })

            # E: rank_avg
            fused = rank_averaging(sub, test_cids, lower_is_better=True)
            top_cids, full_rank = _ranked_top_k(fused, top_k)
            perspectives.append({
                "name": f"{tag}_rank_avg",
                "group": "E-Quad-RankAvg",
                "models_used": combo_names,
                "method": "rank_avg",
                "top_k_cids": top_cids,
                "full_ranking": full_rank,
            })

            # F: type_balanced_rrf
            fused = type_balanced_rrf(sub, test_cids, sub_types, k=rrf_k)
            top_cids, full_rank = _ranked_top_k(fused, top_k)
            perspectives.append({
                "name": f"{tag}_type_balanced",
                "group": "F-Quad-TypeBal",
                "models_used": combo_names,
                "method": "type_balanced_rrf",
                "top_k_cids": top_cids,
                "full_ranking": full_rank,
            })

    # --- Group G: Full 3NN+2ML balanced (2) ---
    all_types = {n: model_types[n] for n in models}

    # G1: type_balanced_rrf
    fused = type_balanced_rrf(models, test_cids, all_types, k=rrf_k)
    top_cids, full_rank = _ranked_top_k(fused, top_k)
    perspectives.append({
        "name": "all5_type_balanced",
        "group": "G-Full",
        "models_used": list(models.keys()),
        "method": "type_balanced_rrf",
        "top_k_cids": top_cids,
        "full_ranking": full_rank,
    })

    # G2: rank_avg
    fused = rank_averaging(models, test_cids, lower_is_better=True)
    top_cids, full_rank = _ranked_top_k(fused, top_k)
    perspectives.append({
        "name": "all5_rank_avg",
        "group": "G-Full",
        "models_used": list(models.keys()),
        "method": "rank_avg",
        "top_k_cids": top_cids,
        "full_ranking": full_rank,
    })

    print(f"  Built {len(perspectives)} perspectives across "
          f"{len(set(p['group'] for p in perspectives))} groups")
    return perspectives


# ============================================================================
# 3. CONSENSUS COMPUTATION
# ============================================================================

def compute_consensus(perspectives, test_cids, top_k=25,
                      tier1_pct=0.75, tier2_pct=0.50, tier3_pct=0.25):
    """
    Count votes and average ranks across all perspectives.

    Returns a list of dicts sorted by (votes desc, avg_rank asc):
        [{cif_id, votes, pct, avg_rank, tier}, ...]
    """
    n_persp = len(perspectives)
    votes = defaultdict(int)
    rank_sums = defaultdict(float)
    rank_counts = defaultdict(int)

    for p in perspectives:
        top_set = set(p["top_k_cids"][:top_k])
        for cid in top_set:
            votes[cid] += 1
        for cid in test_cids:
            r = p["full_ranking"].get(cid)
            if r is not None:
                rank_sums[cid] += r
                rank_counts[cid] += 1

    tier1_thresh = int(np.ceil(tier1_pct * n_persp))
    tier2_thresh = int(np.ceil(tier2_pct * n_persp))
    tier3_thresh = int(np.ceil(tier3_pct * n_persp))

    rows = []
    for cid in test_cids:
        v = votes.get(cid, 0)
        avg_r = rank_sums[cid] / rank_counts[cid] if rank_counts[cid] else len(test_cids)
        if v >= tier1_thresh:
            tier = "Tier 1"
        elif v >= tier2_thresh:
            tier = "Tier 2"
        elif v >= tier3_thresh:
            tier = "Tier 3"
        else:
            tier = "Below"
        rows.append({"cif_id": cid, "votes": v,
                      "pct": v / n_persp if n_persp else 0,
                      "avg_rank": avg_r, "tier": tier})

    rows.sort(key=lambda r: (-r["votes"], r["avg_rank"]))
    return rows, {"n_perspectives": n_persp,
                  "tier1_thresh": tier1_thresh,
                  "tier2_thresh": tier2_thresh,
                  "tier3_thresh": tier3_thresh}


# ============================================================================
# 4. CANDIDATE SELECTION
# ============================================================================

def select_final_candidates(consensus_rows, target_k=25):
    """
    Pick final nominees using threshold + fill logic.

    1. Auto-include all Tier 1 + Tier 2.
    2. If < target_k: fill from Tier 3 by (votes desc, avg_rank asc).
    3. If still < target_k: fill from remaining by avg_rank asc.
    4. If > target_k at Tier 2: take top target_k overall.

    Returns list of target_k dicts with 'nominated' = True/reason.
    """
    tier12 = [r for r in consensus_rows if r["tier"] in ("Tier 1", "Tier 2")]
    tier3 = [r for r in consensus_rows if r["tier"] == "Tier 3"]
    rest = [r for r in consensus_rows if r["tier"] == "Below"]

    if len(tier12) >= target_k:
        final = tier12[:target_k]
        for r in final:
            r["nomination_reason"] = r["tier"]
    else:
        final = list(tier12)
        for r in final:
            r["nomination_reason"] = r["tier"]
        remaining = target_k - len(final)
        if remaining > 0 and tier3:
            fill = tier3[:remaining]
            for r in fill:
                r["nomination_reason"] = "Tier 3 (fill)"
            final.extend(fill)
            remaining = target_k - len(final)
        if remaining > 0 and rest:
            fill = rest[:remaining]
            for r in fill:
                r["nomination_reason"] = "Below threshold (fill by avg rank)"
            final.extend(fill)

    for i, r in enumerate(final, 1):
        r["final_rank"] = i
    return final


# ============================================================================
# 5-11. VISUALIZATION
# ============================================================================

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def plot_consensus_barplot(final_candidates, n_perspectives, output_dir):
    """Plot 5: Horizontal bar chart of vote counts for the final 25 nominees."""
    fig, ax = plt.subplots(figsize=(10, 8))
    names = [r["cif_id"].replace("_FSR", "") for r in reversed(final_candidates)]
    vote_counts = [r["votes"] for r in reversed(final_candidates)]
    tiers = [r["tier"] for r in reversed(final_candidates)]
    colors = [TIER_COLORS.get(t, TIER_COLORS["Below"]) for t in tiers]

    bars = ax.barh(range(len(names)), vote_counts, color=colors, edgecolor="white",
                   linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel(f"Votes (out of {n_perspectives} perspectives)")
    ax.set_title("DFT Candidate Consensus Votes", fontweight="bold", fontsize=13)

    # Tier threshold lines
    for label, pct, ls in [("75%", 0.75, "--"), ("50%", 0.50, "-."),
                           ("25%", 0.25, ":")]:
        thresh = int(np.ceil(pct * n_perspectives))
        ax.axvline(thresh, color="grey", linestyle=ls, linewidth=0.8, alpha=0.7)
        ax.text(thresh + 0.2, len(names) - 0.5, label, fontsize=7, color="grey")

    # Legend
    patches = [mpatches.Patch(color=TIER_COLORS[t], label=t)
               for t in ["Tier 1", "Tier 2", "Tier 3", "Below"]
               if any(r["tier"] == t for r in final_candidates)]
    ax.legend(handles=patches, loc="lower right", fontsize=8)

    ax.set_xlim(0, n_perspectives + 1)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "consensus_votes_barplot.png"), dpi=200)
    plt.close(fig)


def plot_consensus_heatmap(perspectives, final_candidates, consensus_rows,
                           output_dir, n_near_miss=25):
    """Plot 6: Binary heatmap — rows = nominees + near-misses, cols = 28 perspectives."""
    # Collect nominees + near-miss
    nominee_ids = [r["cif_id"] for r in final_candidates]
    nominee_set = set(nominee_ids)
    near_miss = [r["cif_id"] for r in consensus_rows
                 if r["cif_id"] not in nominee_set and r["votes"] > 0][:n_near_miss]
    row_ids = nominee_ids + near_miss

    # Sort perspectives by group
    persp_sorted = sorted(perspectives, key=lambda p: p["group"])
    col_names = [p["name"] for p in persp_sorted]

    mat = np.zeros((len(row_ids), len(col_names)), dtype=int)
    for j, p in enumerate(persp_sorted):
        top_set = set(p["top_k_cids"])
        for i, cid in enumerate(row_ids):
            if cid in top_set:
                mat[i, j] = 1

    fig, ax = plt.subplots(figsize=(16, max(8, len(row_ids) * 0.25)))
    cmap = ListedColormap(["#f0f0f0", "#2ecc71"])
    ax.imshow(mat, aspect="auto", cmap=cmap, interpolation="nearest")

    # Row labels
    row_labels = []
    for cid in row_ids:
        short = cid.replace("_FSR", "")
        marker = " ***" if cid in nominee_set else ""
        row_labels.append(f"{short}{marker}")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=6)

    # Column labels
    short_cols = [p["name"].replace("_individual", "").replace("smote_", "SM-")
                  .replace("extra_trees", "ET").replace("random_forest", "RF")
                  for p in persp_sorted]
    ax.set_xticks(range(len(short_cols)))
    ax.set_xticklabels(short_cols, fontsize=5, rotation=90)

    # Group separators
    groups_seen = []
    for j, p in enumerate(persp_sorted):
        if not groups_seen or groups_seen[-1] != p["group"]:
            if groups_seen:
                ax.axvline(j - 0.5, color="black", linewidth=1)
            groups_seen.append(p["group"])

    # Nominee / near-miss separator
    if near_miss:
        ax.axhline(len(nominee_ids) - 0.5, color="red", linewidth=1.5,
                    linestyle="--")

    ax.set_title("Perspective Support Matrix (*** = nominated for DFT)",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "consensus_heatmap.png"), dpi=200)
    plt.close(fig)


def plot_perspective_agreement(perspectives, top_k, output_dir):
    """Plot 7: 28x28 Jaccard similarity heatmap between perspectives."""
    n = len(perspectives)
    jaccard = np.zeros((n, n))
    top_sets = [set(p["top_k_cids"][:top_k]) for p in perspectives]

    for i in range(n):
        for j in range(n):
            inter = len(top_sets[i] & top_sets[j])
            union = len(top_sets[i] | top_sets[j])
            jaccard[i, j] = inter / union if union else 0

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(jaccard, cmap="YlOrRd", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Jaccard Similarity", shrink=0.8)

    labels = [p["name"].replace("_individual", "").replace("smote_", "SM-")
              .replace("extra_trees", "ET").replace("random_forest", "RF")
              for p in perspectives]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=5, rotation=90)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=5)
    ax.set_title(f"Perspective Agreement (Jaccard @ top-{top_k})",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "perspective_agreement_jaccard.png"),
                dpi=200)
    plt.close(fig)


def plot_nn_ml_agreement(final_candidates, perspectives, output_dir, top_k=25):
    """Plot 8: NN vs ML individual top-K overlap with final nominees marked."""
    nn_indiv = [p for p in perspectives
                if p["group"] == "A-Individual" and any(n in p["name"] for n in NN_MODELS)]
    ml_indiv = [p for p in perspectives
                if p["group"] == "A-Individual" and any(n in p["name"] for n in ML_MODELS)]

    nn_union = set()
    for p in nn_indiv:
        nn_union |= set(p["top_k_cids"][:top_k])
    ml_union = set()
    for p in ml_indiv:
        ml_union |= set(p["top_k_cids"][:top_k])

    both = nn_union & ml_union
    nn_only = nn_union - ml_union
    ml_only = ml_union - nn_union
    nominee_set = set(r["cif_id"] for r in final_candidates)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Stacked horizontal bar data
    categories = ["NN-only", "Both NN & ML", "ML-only", "Neither"]
    nom_counts = []
    non_nom_counts = []
    for cat, cid_set in [("NN-only", nn_only), ("Both NN & ML", both),
                         ("ML-only", ml_only)]:
        nom = len(cid_set & nominee_set)
        nom_counts.append(nom)
        non_nom_counts.append(len(cid_set) - nom)

    # "Neither" = nominees not in any individual top-K
    neither_noms = len(nominee_set - nn_union - ml_union)
    nom_counts.append(neither_noms)
    non_nom_counts.append(0)

    y = range(len(categories))
    ax.barh(y, nom_counts, color="#2ecc71", label="In final 25", height=0.6)
    ax.barh(y, non_nom_counts, left=nom_counts, color="#bdc3c7",
            label="Not nominated", height=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels(categories, fontsize=10)
    ax.set_xlabel("Number of structures")
    ax.set_title("NN vs ML Individual Top-25 Agreement", fontweight="bold",
                 fontsize=12)
    ax.legend(loc="lower right")

    # Annotate counts
    for i in range(len(categories)):
        total = nom_counts[i] + non_nom_counts[i]
        if total > 0:
            ax.text(total + 0.5, i, str(total), va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "nn_ml_agreement.png"), dpi=200)
    plt.close(fig)


def plot_acceptance_ratio(perspectives, final_candidates, output_dir, top_k=25):
    """Plot 9: For each perspective, what % of its top-K made the final list."""
    nominee_set = set(r["cif_id"] for r in final_candidates)

    names = []
    ratios = []
    colors = []
    for p in perspectives:
        top_set = set(p["top_k_cids"][:top_k])
        ratio = len(top_set & nominee_set) / top_k if top_k else 0
        names.append(p["name"].replace("_individual", "")
                     .replace("smote_", "SM-").replace("extra_trees", "ET")
                     .replace("random_forest", "RF"))
        ratios.append(ratio)
        colors.append(GROUP_COLORS.get(p["group"], "#95a5a6"))

    fig, ax = plt.subplots(figsize=(10, 8))
    y = range(len(names))
    ax.barh(y, ratios, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("Acceptance Ratio (fraction of top-25 in final nominees)")
    ax.set_title("Per-Perspective Acceptance Ratio", fontweight="bold", fontsize=12)
    ax.set_xlim(0, 1.05)

    # Group legend
    seen = {}
    for p in perspectives:
        if p["group"] not in seen:
            seen[p["group"]] = GROUP_COLORS.get(p["group"], "#95a5a6")
    patches = [mpatches.Patch(color=c, label=g) for g, c in seen.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "acceptance_ratio.png"), dpi=200)
    plt.close(fig)


def plot_rank_stability(perspectives, final_candidates, output_dir):
    """Plot 10: Boxplot of ranks across all 28 perspectives for each nominee."""
    fig, ax = plt.subplots(figsize=(12, 7))

    data = []
    labels = []
    for r in final_candidates:
        cid = r["cif_id"]
        ranks = [p["full_ranking"].get(cid, len(p["full_ranking"]) + 1)
                 for p in perspectives]
        data.append(ranks)
        labels.append(cid.replace("_FSR", ""))

    bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.6)
    for i, (patch, r) in enumerate(zip(bp["boxes"], final_candidates)):
        patch.set_facecolor(TIER_COLORS.get(r["tier"], "#95a5a6"))
        patch.set_alpha(0.7)

    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Rank across 28 perspectives (lower = better)")
    ax.set_title("Rank Stability of Final 25 Nominees", fontweight="bold",
                 fontsize=12)
    ax.axvline(25, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(26, len(data), "top-25 line", fontsize=7, color="red")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "rank_stability_boxplot.png"), dpi=200)
    plt.close(fig)


def plot_umap_nominees(base_dir, final_candidates, output_dir):
    """Plot 11: UMAP of all Phase6 structures with nominees highlighted by tier."""
    emb_path = os.path.join(base_dir, "embedding_analysis",
                            "Phase6_embeddings.npz")
    if not os.path.isfile(emb_path):
        print("  [SKIP] UMAP plot: Phase6_embeddings.npz not found")
        return

    try:
        from umap import UMAP
    except ImportError:
        print("  [SKIP] UMAP plot: umap-learn not installed")
        return

    # Load embeddings
    data = np.load(emb_path, allow_pickle=True)
    keys = list(data.keys())
    if "cif_ids" in keys and "embeddings" in keys:
        cif_ids = data["cif_ids"]
        embeddings = data["embeddings"]
    else:
        cif_ids = data[keys[0]]
        embeddings = data[keys[1]]
    cif_ids = np.array([c.decode() if isinstance(c, bytes) else str(c)
                        for c in cif_ids])

    # Compute or load UMAP
    cache_path = os.path.join(output_dir, "umap_cache.npz")
    if os.path.isfile(cache_path):
        cached = np.load(cache_path)
        coords = cached["coords"]
        print("  Loaded cached UMAP coordinates")
    else:
        print("  Computing UMAP (this may take a few minutes)...")
        reducer = UMAP(n_neighbors=30, min_dist=0.3, metric="cosine",
                       random_state=42)
        coords = reducer.fit_transform(embeddings)
        np.savez_compressed(cache_path, coords=coords, cif_ids=cif_ids)
        print("  UMAP done, cached to umap_cache.npz")

    # Map nominees to tiers
    nominee_tiers = {r["cif_id"]: r["tier"] for r in final_candidates}
    cid_to_idx = {c: i for i, c in enumerate(cif_ids)}

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords[:, 0], coords[:, 1], c="#e0e0e0", s=2, alpha=0.3,
               rasterized=True)

    for tier, color in [("Tier 3", TIER_COLORS["Tier 3"]),
                        ("Tier 2", TIER_COLORS["Tier 2"]),
                        ("Tier 1", TIER_COLORS["Tier 1"])]:
        idxs = [cid_to_idx[cid] for cid, t in nominee_tiers.items()
                if t == tier and cid in cid_to_idx]
        if idxs:
            ax.scatter(coords[idxs, 0], coords[idxs, 1], c=color, s=60,
                       edgecolors="black", linewidth=0.5, label=tier, zorder=5)

    # Also plot Below-threshold nominees (fill candidates)
    below_idxs = [cid_to_idx[cid] for cid, t in nominee_tiers.items()
                  if t == "Below" and cid in cid_to_idx]
    if below_idxs:
        ax.scatter(coords[below_idxs, 0], coords[below_idxs, 1],
                   c=TIER_COLORS["Below"], s=60, edgecolors="black",
                   linewidth=0.5, label="Below (fill)", zorder=5)

    ax.set_title("UMAP: Phase6 Structures with DFT Nominees",
                 fontweight="bold", fontsize=12)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "umap_nominees.png"), dpi=200)
    plt.close(fig)


# ============================================================================
# 12. OUTPUT FILES
# ============================================================================

def write_output_files(perspectives, final_candidates, consensus_rows,
                       meta, output_dir, top_k):
    """Write all CSVs, TXTs, and JSON outputs."""
    # --- FINAL_DFT_TOP25.txt ---
    with open(os.path.join(output_dir, f"FINAL_DFT_TOP{top_k}.txt"), "w",
              encoding="utf-8") as f:
        for r in final_candidates:
            f.write(r["cif_id"] + "\n")

    # --- FINAL_DFT_TOP25.csv ---
    with open(os.path.join(output_dir, f"FINAL_DFT_TOP{top_k}.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["final_rank", "cif_id", "votes",
                                          "pct", "avg_rank", "tier",
                                          "nomination_reason"])
        w.writeheader()
        for r in final_candidates:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k])
                        for k in w.fieldnames})

    # --- full_consensus_ranking.csv ---
    with open(os.path.join(output_dir, "full_consensus_ranking.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rank", "cif_id", "votes", "pct",
                                          "avg_rank", "tier"])
        w.writeheader()
        for i, r in enumerate(consensus_rows, 1):
            w.writerow({"rank": i, "cif_id": r["cif_id"], "votes": r["votes"],
                         "pct": f"{r['pct']:.4f}",
                         "avg_rank": f"{r['avg_rank']:.1f}",
                         "tier": r["tier"]})

    # --- Per-perspective top-K files ---
    for p in perspectives:
        group = p["group"]
        if group == "A-Individual":
            subdir = os.path.join(output_dir, "individual_models")
        elif "Pair" in group:
            subdir = os.path.join(output_dir, "balanced_ensembles", "pairs")
        elif "Quad" in group or "TypeBal" in group:
            subdir = os.path.join(output_dir, "balanced_ensembles", "quads")
        else:
            subdir = os.path.join(output_dir, "balanced_ensembles", "full")
        _ensure_dir(subdir)
        fname = f"{p['name']}_top{top_k}.txt"
        with open(os.path.join(subdir, fname), "w", encoding="utf-8") as f:
            for cid in p["top_k_cids"][:top_k]:
                f.write(cid + "\n")

    # --- perspective_summary.json ---
    summary = []
    for p in perspectives:
        summary.append({
            "name": p["name"],
            "group": p["group"],
            "method": p["method"],
            "models_used": p["models_used"],
            "top_k_cids": p["top_k_cids"][:top_k],
        })
    with open(os.path.join(output_dir, "perspective_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump({"n_perspectives": meta["n_perspectives"],
                    "top_k": top_k,
                    "tier1_thresh": meta["tier1_thresh"],
                    "tier2_thresh": meta["tier2_thresh"],
                    "tier3_thresh": meta["tier3_thresh"],
                    "perspectives": summary}, f, indent=2)


# ============================================================================
# 13. NOMINATION REPORT
# ============================================================================

def write_report(perspectives, final_candidates, consensus_rows, meta,
                 output_dir, top_k, args):
    """Write nomination_report.md with full methodology and results."""
    n_p = meta["n_perspectives"]
    lines = []
    w = lines.append

    w("# DFT Candidate Nomination Report")
    w(f"\n**Generated**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}")
    w(f"**Target**: Top {top_k} structures for DFT bandgap calculation")
    w(f"**Perspectives evaluated**: {n_p}")
    w(f"**Structures considered**: {len(consensus_rows):,}")
    w("")

    # Methodology
    w("## 1. Methodology")
    w("")
    w("### 1.1 Multi-Perspective Consensus")
    w("")
    w("We evaluate **28 independent ranking perspectives** constructed from")
    w("3 neural network models (MOFTransformer, different random seeds) and")
    w("2 machine learning classifiers (SMOTE-boosted, trained on embeddings).")
    w("Only **balanced ensembles** are used: every combination contains an")
    w("equal number of NN and ML models to prevent type domination.")
    w("")
    w("| Group | Composition | Fusion Method | Count |")
    w("|-------|------------|---------------|-------|")
    w("| A | Individual models | Own scores | 5 |")
    w("| B | 1 NN + 1 ML pairs | Reciprocal Rank Fusion (k=60) | 6 |")
    w("| C | 1 NN + 1 ML pairs | Rank Averaging | 6 |")
    w("| D | 2 NN + 2 ML quads | RRF | 3 |")
    w("| E | 2 NN + 2 ML quads | Rank Averaging | 3 |")
    w("| F | 2 NN + 2 ML quads | Type-Balanced RRF | 3 |")
    w("| G | Full 3 NN + 2 ML | Type-Balanced RRF + Rank Avg | 2 |")
    w(f"| | | **Total** | **{n_p}** |")
    w("")
    w("**Rationale**: Unbalanced combinations (e.g. 2 NN + 1 ML, or 3 NN + 2 ML with")
    w("plain RRF) are excluded because our analysis showed Jaccard similarity = 0.0")
    w("between NN and ML top-25 lists, indicating that without balancing, one paradigm")
    w("completely dominates the fused ranking.")
    w("")
    w("### 1.2 Fusion Methods")
    w("")
    w("- **Reciprocal Rank Fusion (RRF)**: `score(cid) = sum(1/(k + rank_i))`, k=60.")
    w("  Rank-based, paradigm-agnostic, robust to score scale differences.")
    w("- **Rank Averaging**: `score(cid) = mean(rank_i)`. Simplest rank-based fusion.")
    w("- **Type-Balanced RRF**: Two-stage RRF. Stage 1: RRF within each type (NN, ML)")
    w("  independently. Stage 2: RRF across the 2 type-level scores. Ensures each")
    w("  model family contributes exactly 50%.")
    w("")
    w("Score averaging is excluded: normalising NN regression bandgaps against ML")
    w("classification probabilities is unreliable.")
    w("")
    w("### 1.3 Confidence Tiers")
    w("")
    w(f"- **Tier 1** (Very High Confidence): >= {meta['tier1_thresh']}/{n_p} votes "
      f"(>= 75%)")
    w(f"- **Tier 2** (High Confidence): >= {meta['tier2_thresh']}/{n_p} votes "
      f"(>= 50%)")
    w(f"- **Tier 3** (Moderate Confidence): >= {meta['tier3_thresh']}/{n_p} votes "
      f"(>= 25%)")
    w("")
    w("Nomination: auto-include all Tier 1 + Tier 2 structures. If fewer than")
    w(f"{top_k}, fill from Tier 3 ordered by (votes descending, average rank ascending).")
    w("")

    # Results
    w("## 2. Final Nominees")
    w("")
    w(f"| Rank | CIF ID | Votes/{n_p} | Agreement | Avg Rank | Tier |")
    w("|------|--------|------------|-----------|----------|------|")
    for r in final_candidates:
        w(f"| {r['final_rank']} | {r['cif_id']} | {r['votes']} | "
          f"{r['pct']:.1%} | {r['avg_rank']:.1f} | {r['tier']} |")
    w("")

    # Tier distribution
    tier_counts = defaultdict(int)
    for r in final_candidates:
        tier_counts[r["tier"]] += 1
    w("### Tier Breakdown")
    w("")
    for t in ["Tier 1", "Tier 2", "Tier 3", "Below"]:
        if tier_counts[t]:
            w(f"- **{t}**: {tier_counts[t]} structures")
    w("")

    # Agreement statistics
    w("## 3. Agreement Statistics")
    w("")

    # Compute group-level Jaccard averages
    group_jaccards = defaultdict(list)
    cross_jaccards = []
    top_sets = {p["name"]: set(p["top_k_cids"][:top_k]) for p in perspectives}
    for i, p1 in enumerate(perspectives):
        for j, p2 in enumerate(perspectives):
            if i >= j:
                continue
            s1, s2 = top_sets[p1["name"]], top_sets[p2["name"]]
            inter = len(s1 & s2)
            union = len(s1 | s2)
            jac = inter / union if union else 0
            if p1["group"] == p2["group"]:
                group_jaccards[p1["group"]].append(jac)
            else:
                cross_jaccards.append(jac)

    w("| Group | Avg Jaccard (within group) | Pairs |")
    w("|-------|--------------------------|-------|")
    for g in sorted(group_jaccards.keys()):
        vals = group_jaccards[g]
        w(f"| {g} | {np.mean(vals):.3f} | {len(vals)} |")
    if cross_jaccards:
        w(f"| Cross-group | {np.mean(cross_jaccards):.3f} | {len(cross_jaccards)} |")
    w("")

    # NN vs ML overlap
    nn_union = set()
    ml_union = set()
    for p in perspectives:
        if p["group"] == "A-Individual":
            if any(n in p["name"] for n in NN_MODELS):
                nn_union |= set(p["top_k_cids"][:top_k])
            elif any(n in p["name"] for n in ML_MODELS):
                ml_union |= set(p["top_k_cids"][:top_k])
    both = nn_union & ml_union
    w(f"- NN individual union (top-{top_k}): **{len(nn_union)}** unique structures")
    w(f"- ML individual union (top-{top_k}): **{len(ml_union)}** unique structures")
    w(f"- Cross-paradigm overlap: **{len(both)}** structures")
    w("")

    # Acceptance ratios
    w("## 4. Per-Perspective Acceptance Ratios")
    w("")
    w(f"Fraction of each perspective's top-{top_k} that appears in the final nominees:")
    w("")
    w("| Perspective | Group | Acceptance |")
    w("|------------|-------|------------|")
    nominee_set = set(r["cif_id"] for r in final_candidates)
    for p in perspectives:
        top_set = set(p["top_k_cids"][:top_k])
        ratio = len(top_set & nominee_set) / top_k if top_k else 0
        w(f"| {p['name']} | {p['group']} | {ratio:.1%} |")
    w("")

    # Plots
    w("## 5. Plots")
    w("")
    w("| File | Description |")
    w("|------|-------------|")
    w("| `plots/consensus_votes_barplot.png` | Horizontal bar chart of vote counts per nominee, colored by tier |")
    w("| `plots/consensus_heatmap.png` | Binary heatmap: nominees x perspectives, showing which perspectives support each nominee |")
    w("| `plots/perspective_agreement_jaccard.png` | 28x28 Jaccard similarity between perspectives |")
    w("| `plots/nn_ml_agreement.png` | NN vs ML individual top-25 overlap with nominees marked |")
    w("| `plots/acceptance_ratio.png` | Per-perspective acceptance ratio (how aligned each is with consensus) |")
    w("| `plots/rank_stability_boxplot.png` | Boxplot of each nominee's rank across 28 perspectives |")
    w("| `plots/umap_nominees.png` | UMAP of all Phase6 structures with nominees highlighted by tier |")
    w("")

    # Reproducibility
    w("## 6. Reproducibility")
    w("")
    w("```bash")
    cmd_parts = ["python nominate_dft_candidates.py"]
    cmd_parts.append(f"  --base_dir {args.base_dir}")
    cmd_parts.append(f"  --output_dir {args.output_dir}")
    cmd_parts.append(f"  --top_k {args.top_k}")
    cmd_parts.append(f"  --rrf_k {args.rrf_k}")
    cmd_parts.append(f"  --tier1_pct {args.tier1_pct}")
    cmd_parts.append(f"  --tier2_pct {args.tier2_pct}")
    cmd_parts.append(f"  --tier3_pct {args.tier3_pct}")
    w(" \\\n  ".join(cmd_parts))
    w("```")
    w("")
    w("**Models used**:")
    w(f"- NN: {', '.join(NN_MODELS)} (MOFTransformer, 3 random seeds)")
    w(f"- ML: {', '.join(ML_MODELS)} (SMOTE-boosted sklearn classifiers on embeddings)")
    w("")

    report_path = os.path.join(output_dir, "nomination_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return report_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DFT Candidate Nomination via Multi-Perspective Consensus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--base_dir", type=str, default=".",
                        help="Phase6 working directory (default: current dir)")
    parser.add_argument("--output_dir", type=str, default="DFT-subset-Nomination",
                        help="Output directory (default: DFT-subset-Nomination)")
    parser.add_argument("--top_k", type=int, default=25,
                        help="Number of candidates to nominate (default: 25)")
    parser.add_argument("--rrf_k", type=int, default=60,
                        help="RRF smoothing constant (default: 60)")
    parser.add_argument("--tier1_pct", type=float, default=0.75,
                        help="Tier 1 threshold as fraction (default: 0.75)")
    parser.add_argument("--tier2_pct", type=float, default=0.50,
                        help="Tier 2 threshold as fraction (default: 0.50)")
    parser.add_argument("--tier3_pct", type=float, default=0.25,
                        help="Tier 3 threshold as fraction (default: 0.25)")
    parser.add_argument("--skip_umap", action="store_true",
                        help="Skip UMAP computation (faster)")
    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    output_dir = os.path.join(base_dir, args.output_dir) \
        if not os.path.isabs(args.output_dir) else args.output_dir
    plots_dir = _ensure_dir(os.path.join(output_dir, "plots"))
    _ensure_dir(output_dir)

    print("=" * 72)
    print("  DFT Candidate Nomination — Multi-Perspective Consensus")
    print("=" * 72)

    # Step 1: Load models
    print("\n[1/7] Loading models...")
    models, test_cids, model_types = discover_and_load_models(base_dir)

    # Step 2: Build perspectives
    print("\n[2/7] Building 28 balanced perspectives...")
    perspectives = build_perspectives(models, test_cids, model_types,
                                      top_k=args.top_k, rrf_k=args.rrf_k)

    # Step 3: Compute consensus
    print("\n[3/7] Computing consensus votes...")
    consensus_rows, meta = compute_consensus(
        perspectives, test_cids, top_k=args.top_k,
        tier1_pct=args.tier1_pct, tier2_pct=args.tier2_pct,
        tier3_pct=args.tier3_pct)

    # Count tier distribution
    tier_dist = defaultdict(int)
    for r in consensus_rows:
        if r["votes"] > 0:
            tier_dist[r["tier"]] += 1
    print(f"  Structures with >= 1 vote: "
          f"{sum(tier_dist.values())}")
    for t in ["Tier 1", "Tier 2", "Tier 3"]:
        print(f"    {t}: {tier_dist.get(t, 0)}")

    # Step 4: Select final candidates
    print(f"\n[4/7] Selecting final {args.top_k} nominees...")
    final = select_final_candidates(consensus_rows, target_k=args.top_k)
    print(f"  Selected {len(final)} nominees:")
    for r in final:
        print(f"    #{r['final_rank']:2d}  {r['cif_id']:<30s}  "
              f"votes={r['votes']:2d}/{meta['n_perspectives']}  "
              f"avg_rank={r['avg_rank']:7.1f}  {r['tier']}")

    # Step 5-11: Plots
    print(f"\n[5/7] Generating plots in {plots_dir} ...")
    plot_consensus_barplot(final, meta["n_perspectives"], plots_dir)
    print("  - consensus_votes_barplot.png")
    plot_consensus_heatmap(perspectives, final, consensus_rows, plots_dir)
    print("  - consensus_heatmap.png")
    plot_perspective_agreement(perspectives, args.top_k, plots_dir)
    print("  - perspective_agreement_jaccard.png")
    plot_nn_ml_agreement(final, perspectives, plots_dir, top_k=args.top_k)
    print("  - nn_ml_agreement.png")
    plot_acceptance_ratio(perspectives, final, plots_dir, top_k=args.top_k)
    print("  - acceptance_ratio.png")
    plot_rank_stability(perspectives, final, plots_dir)
    print("  - rank_stability_boxplot.png")
    if not args.skip_umap:
        plot_umap_nominees(base_dir, final, plots_dir)
    else:
        print("  - [SKIPPED] umap_nominees.png (--skip_umap)")

    # Step 12: Write output files
    print(f"\n[6/7] Writing output files to {output_dir} ...")
    write_output_files(perspectives, final, consensus_rows, meta,
                       output_dir, args.top_k)
    print(f"  - FINAL_DFT_TOP{args.top_k}.txt")
    print(f"  - FINAL_DFT_TOP{args.top_k}.csv")
    print(f"  - full_consensus_ranking.csv")
    print(f"  - perspective_summary.json")
    print(f"  - individual_models/ ({len([p for p in perspectives if p['group'] == 'A-Individual'])} files)")
    print(f"  - balanced_ensembles/ ({len([p for p in perspectives if p['group'] != 'A-Individual'])} files)")

    # Step 13: Write report
    print(f"\n[7/7] Writing nomination report...")
    report_path = write_report(perspectives, final, consensus_rows, meta,
                               output_dir, args.top_k, args)
    print(f"  - {os.path.basename(report_path)}")

    # Final summary to stdout
    print("\n" + "=" * 72)
    print(f"  DONE. Final {args.top_k} DFT candidates:")
    print("=" * 72)
    for r in final:
        print(f"  {r['final_rank']:2d}. {r['cif_id']}")
    print("=" * 72)
    print(f"  Output: {output_dir}")
    print(f"  Report: {report_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
