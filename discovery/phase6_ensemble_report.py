#!/usr/bin/env python3
"""
Phase6 Ensemble Report (No Ground Truth)
========================================

Ensembles NN and ML predictions on Phase6 (new test set, no labels), runs ALL
ensemble methods (RRF, rank_avg, vote_topK, score_avg, weighted_rrf), produces
per-combo top-25/50/100 lists, agreement analysis, and a well-documented report.

Usage:
  python phase6_ensemble_report.py \\
    --base_dir ./data/phase6 \\
    --models exp364 exp370 smote_extra_trees smote_random_forest \\
    --output_dir ./ensemble_report

  # Auto-discover all models:
  python phase6_ensemble_report.py --base_dir . --auto_discover --output_dir ./ensemble_report

  # Exhaustive combos of size 2, 3, 4:
  python phase6_ensemble_report.py --base_dir . --models exp364 exp370 smote_extra_trees \\
    --combo_size 2 3 4 --output_dir ./ensemble_report

  # Auto-discover + 4 explicit combos in one run:
  python phase6_ensemble_report.py --base_dir . --auto_discover \\
    --combo exp364 exp370 --combo exp364 smote_random_forest \\
    --combo exp370 random_forest --combo exp364 exp370 smote_extra_trees \\
    --output_dir ./ensemble_report
"""

import os
import sys
import csv
import glob
import json
import argparse
from itertools import combinations
from collections import defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
for d in (SCRIPT_DIR, PROJECT_ROOT, os.path.join(PROJECT_ROOT, "src")):
    if d not in sys.path:
        sys.path.insert(0, d)

from ensemble_phase6_predictions import (
    load_predictions_from_csv,
    find_predictions_csv,
    reciprocal_rank_fusion,
    rank_averaging,
    score_to_rank,
    _fill_missing_scores,
)
from ensemble_discovery import (
    top_k_voting,
    score_averaging,
    weighted_rrf,
)


# =============================================================================
# MODEL RESOLUTION (Phase6: test_predictions.csv or inference_predictions.csv)
# =============================================================================

def resolve_phase6_models(base_dir, model_names):
    """
    Resolve short model names to dirs with test_predictions.csv or
    inference_predictions.csv. Same logic as resolve_models_to_dirs but checks
    both CSV names.
    """
    resolved = []
    base_dir = os.path.abspath(base_dir or ".")
    experiments_dir = os.path.join(base_dir, "experiments")
    clf_base = os.path.join(base_dir, "embedding_classifiers", "strategy_d_farthest_point")
    knn_base = os.path.join(base_dir, "knn_results", "strategy_d_farthest_point")

    def has_predictions(path):
        return find_predictions_csv(path) is not None

    for name in model_names:
        name = name.strip()
        if not name:
            continue
        # Already a path?
        if os.path.sep in name or (os.path.exists(name) and os.path.isdir(name)):
            cand = os.path.abspath(name)
            if has_predictions(cand):
                resolved.append(cand)
                continue
            cand = os.path.join(base_dir, name)
            if has_predictions(cand):
                resolved.append(os.path.abspath(cand))
                continue
        # kNN baseline
        if name.lower() in ("knn", "knn_baseline", "knn_regression"):
            if has_predictions(knn_base):
                resolved.append(knn_base)
                continue
        # NN experiment
        if name.lower().startswith("exp") and any(c.isdigit() for c in name):
            if os.path.isdir(experiments_dir):
                exact = os.path.join(experiments_dir, name)
                if os.path.isdir(exact) and has_predictions(exact):
                    resolved.append(os.path.abspath(exact))
                else:
                    for d in sorted(glob.glob(os.path.join(experiments_dir, name + "*"))):
                        if os.path.isdir(d) and has_predictions(d):
                            resolved.append(os.path.abspath(d))
                            break
                continue
        # ML classifier
        cand = os.path.join(clf_base, name)
        if os.path.isdir(cand) and has_predictions(cand):
            resolved.append(os.path.abspath(cand))
            continue
        cand = os.path.join(knn_base, name)
        if os.path.isdir(cand) and has_predictions(cand):
            resolved.append(os.path.abspath(cand))
            continue
        print(f"  WARNING: Could not resolve model '{name}'")
    return resolved


def auto_discover_phase6_models(base_dir):
    """Discover all experiments + embedding_classifiers with predictions."""
    dirs = []
    base_dir = os.path.abspath(base_dir or ".")
    experiments_dir = os.path.join(base_dir, "experiments")
    clf_base = os.path.join(base_dir, "embedding_classifiers", "strategy_d_farthest_point")

    if os.path.isdir(experiments_dir):
        for name in sorted(os.listdir(experiments_dir)):
            sub = os.path.join(experiments_dir, name)
            if os.path.isdir(sub) and find_predictions_csv(sub):
                dirs.append(sub)
    if os.path.isdir(clf_base):
        for name in sorted(os.listdir(clf_base)):
            sub = os.path.join(clf_base, name)
            if os.path.isdir(sub) and find_predictions_csv(sub):
                dirs.append(sub)
    return dirs


# =============================================================================
# LOADING
# =============================================================================

def load_models(pred_dirs):
    """Load predictions from dirs. Returns (models, test_cids) with intersection of CIDs."""
    models = {}
    all_cid_sets = []

    for pred_dir in pred_dirs:
        csv_path = find_predictions_csv(pred_dir)
        if not csv_path:
            continue
        label = os.path.basename(pred_dir.rstrip(os.path.sep))
        preds = load_predictions_from_csv(csv_path)
        if not preds:
            continue
        models[label] = preds
        all_cid_sets.append(set(preds.keys()))

    if not models:
        return {}, []

    common_cids = set.intersection(*all_cid_sets) if all_cid_sets else set()
    test_cids = sorted(common_cids)

    for name in list(models.keys()):
        models[name] = {cid: models[name][cid] for cid in test_cids if cid in models[name]}

    return models, test_cids


# =============================================================================
# ENSEMBLE METHODS (adapt ensemble_discovery for optional missing cids)
# =============================================================================

def _safe_scores(models, test_cids):
    """Ensure each model has scores for all test_cids (fill NaN for missing)."""
    out = {}
    for name, scores in models.items():
        arr = np.array([scores.get(cid, float("nan")) for cid in test_cids], dtype=float)
        filled = _fill_missing_scores(arr, lower_is_better=True)
        out[name] = {cid: filled[i] for i, cid in enumerate(test_cids)}
    return out


def run_rrf(models, test_cids, k=60):
    return reciprocal_rank_fusion(models, test_cids, k=k, lower_is_better=True)


def run_rank_avg(models, test_cids):
    return rank_averaging(models, test_cids, lower_is_better=True)


def run_vote_top_k(models, test_cids, K):
    return top_k_voting(models, test_cids, K=K)


def run_score_avg(models, test_cids):
    return score_averaging(models, test_cids)


def run_weighted_rrf(models, test_cids, k=60):
    weights = {name: 1.0 for name in models}
    return weighted_rrf(models, test_cids, weights, k=k)


# =============================================================================
# AGREEMENT ANALYSIS
# =============================================================================

def jaccard(a, b):
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def compute_agreement(rankings_dict, top_k):
    """
    rankings_dict: {label: sorted list of cids (best first)}
    Returns: (jaccard_matrix, labels), vote_per_cid
    """
    labels = sorted(rankings_dict.keys())
    top_k_sets = {lbl: set(rankings_dict[lbl][:top_k]) for lbl in labels}
    n = len(labels)
    jaccard_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            jaccard_mat[i, j] = jaccard(top_k_sets[labels[i]], top_k_sets[labels[j]])

    all_in_topk = set()
    for s in top_k_sets.values():
        all_in_topk |= s
    vote_per_cid = defaultdict(int)
    for cid in all_in_topk:
        for lbl in labels:
            if cid in top_k_sets[lbl]:
                vote_per_cid[cid] += 1

    return jaccard_mat, labels, dict(vote_per_cid)


# =============================================================================
# REPORT & PLOTS
# =============================================================================

def write_report(
    output_dir,
    combo_results,
    agreement_by_k,
    vote_per_cid_by_k,
    top_k_list,
):
    """Write phase6_ensemble_report.md."""
    path = os.path.join(output_dir, "phase6_ensemble_report.md")
    lines = []
    lines.append("# Phase6 Ensemble Report (No Ground Truth)")
    lines.append("")
    lines.append("This report shows ensemble rankings and agreement analysis.")
    lines.append("Only CIFs present in ALL model prediction files are ranked (intersection).")
    lines.append("Score direction: ML (multiclass) and NN (regression) are normalized to lower=better before fusion.")
    lines.append("")

    lines.append("## 1. Model Combos and Top-K Lists")
    lines.append("")
    for combo_slug, results in combo_results.items():
        lines.append(f"### {combo_slug}")
        lines.append("")
        for method_name, sorted_cids in results.items():
            lines.append(f"- **{method_name}**:")
            for k in top_k_list:
                top = sorted_cids[:k]
                top_str = ", ".join(top[:10])
                if len(top) > 10:
                    top_str += f", ... (+{len(top) - 10} more)"
                lines.append(f"  - Top {k}: [{top_str}]")
            lines.append("")
        lines.append("")

    lines.append("## 2. Agreement Analysis (No Ground Truth)")
    lines.append("")
    for k in top_k_list:
        if k not in agreement_by_k:
            continue
        jaccard_mat, labels, vote_per_cid = agreement_by_k[k]
        lines.append(f"### Top-{k} Agreement")
        lines.append("")
        lines.append("Pairwise Jaccard (models/ensembles):")
        lines.append("")
        header = "|" + "|".join([f" {lbl[:15]:>15} " for lbl in labels]) + "|"
        sep = "|" + "|".join(["---" for _ in labels]) + "|"
        lines.append(header)
        lines.append(sep)
        for i, lbl in enumerate(labels):
            row = "|" + "|".join([f" {jaccard_mat[i, j]:.3f} " for j in range(len(labels))]) + "|"
            lines.append(row)
        lines.append("")
        lines.append(f"High-agreement CIFs (in top-{k} of most models):")
        sorted_votes = sorted(vote_per_cid.items(), key=lambda x: -x[1])
        n_models = len(labels)
        in_all = [c for c, v in sorted_votes if v == n_models]
        in_80pct = [c for c, v in sorted_votes if v >= int(0.8 * n_models)]
        if in_all:
            lines.append(f"  In ALL ({n_models}) models: {', '.join(in_all)}")
        lines.append(f"  In >=80% ({len(in_80pct)} CIFs): {', '.join(in_80pct[:15])}{'...' if len(in_80pct) > 15 else ''}")
        for cid, votes in sorted_votes[:25]:
            lines.append(f"  - {cid}: {votes}/{n_models} models")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved report: {path}")
    return path


def plot_agreement_heatmap(output_dir, agreement_by_k, top_k_list):
    """Plot Jaccard agreement heatmaps for each K."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed, skipping plots")
        return

    for k in top_k_list:
        if k not in agreement_by_k:
            continue
        jaccard_mat, labels = agreement_by_k[k][:2]
        short_labels = [lbl.split("/")[-1][:12] for lbl in labels]
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), max(6, len(labels) * 0.5)))
        im = ax.imshow(jaccard_mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(short_labels, fontsize=8)
        ax.set_title(f"Top-{k} Agreement (Jaccard)")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{jaccard_mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
        plt.colorbar(im, ax=ax, label="Jaccard")
        plt.tight_layout()
        path = os.path.join(output_dir, f"agreement_heatmap_top{k}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Saved {path}")


def plot_vote_distribution(output_dir, vote_per_cid, top_k=25):
    """Histogram of how many models put each CIF in top-K."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not vote_per_cid:
        return
    votes = list(vote_per_cid.values())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(votes, bins=range(1, max(votes) + 2), align="left", edgecolor="black", alpha=0.7)
    ax.set_xlabel(f"Number of models putting CIF in top-{top_k}")
    ax.set_ylabel("Number of CIFs")
    ax.set_title(f"Vote Distribution (top-{top_k})")
    plt.tight_layout()
    path = os.path.join(output_dir, f"vote_distribution_top{top_k}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_structure_overlap_heatmap(output_dir, rankings_dict, top_k, max_cifs=50):
    """
    Heatmap: rows = CIFs (high-vote first), cols = models/ensembles,
    cell = 1 if CIF in that model's top-K else 0.
    Shows structural consensus across models.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    labels = sorted(rankings_dict.keys())
    top_k_sets = {lbl: set(rankings_dict[lbl][:top_k]) for lbl in labels}
    all_in_topk = set()
    for s in top_k_sets.values():
        all_in_topk |= s
    vote_per_cid = defaultdict(int)
    for cid in all_in_topk:
        for lbl in labels:
            if cid in top_k_sets[lbl]:
                vote_per_cid[cid] += 1
    sorted_cifs = sorted(vote_per_cid.items(), key=lambda x: -x[1])
    cifs_to_plot = [c for c, _ in sorted_cifs[:max_cifs]]
    if not cifs_to_plot or not labels:
        return
    n_cifs = len(cifs_to_plot)
    n_models = len(labels)
    mat = np.zeros((n_cifs, n_models), dtype=float)
    for i, cid in enumerate(cifs_to_plot):
        for j, lbl in enumerate(labels):
            mat[i, j] = 1.0 if cid in top_k_sets[lbl] else 0.0
    short_labels = [lbl.split("/")[-1][:10] for lbl in labels]
    short_cifs = [c[:14] for c in cifs_to_plot]
    fig, ax = plt.subplots(figsize=(max(6, n_models * 0.5), max(6, n_cifs * 0.25)))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n_models))
    ax.set_yticks(range(n_cifs))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(short_cifs, fontsize=6)
    ax.set_xlabel("Model / ensemble")
    ax.set_ylabel("CIF ID (high-vote first)")
    ax.set_title(f"Structure Overlap in Top-{top_k} (1=in top-K, 0=not)")
    plt.colorbar(im, ax=ax, label="In top-K")
    plt.tight_layout()
    path = os.path.join(output_dir, f"structure_overlap_top{top_k}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# =============================================================================
# MAIN
# =============================================================================

def _run_slug(names):
    """Short filesystem-safe slug from model names."""
    def short(n):
        if "/" in n:
            n = n.split("/")[-1]
        if n.startswith("exp") and "_" in n:
            return n.split("_")[0]
        return n.replace(" ", "_")
    return "_".join(sorted(set(short(n) for n in names)))[:80]


def main():
    parser = argparse.ArgumentParser(
        description="Phase6 Ensemble Report (no ground truth)"
    )
    parser.add_argument("--base_dir", type=str, default=None,
                        help="Base directory (Phase6 root)")
    parser.add_argument("--models", type=str, nargs="*", default=None,
                        help="Model names: exp364, exp370, smote_extra_trees, etc.")
    parser.add_argument("--auto_discover", action="store_true",
                        help="Auto-discover all experiments + embedding_classifiers")
    parser.add_argument("--output_dir", type=str, default="./ensemble_report",
                        help="Output directory")
    parser.add_argument("--top_k", type=int, nargs="+", default=[25, 50, 100],
                        help="Top-K values")
    parser.add_argument("--rrf_k", type=int, default=60,
                        help="RRF smoothing parameter k")
    parser.add_argument("--combo_size", type=int, nargs="*", default=None,
                        help="If set, run exhaustive combos of these sizes (e.g. 2 3 4)")
    parser.add_argument("--combo", type=str, nargs="*", action="append", default=None,
                        help="Explicit combo: --combo exp364 exp370 --combo exp364 smote_random_forest (repeatable)")
    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir or SCRIPT_DIR)
    output_dir = os.path.join(base_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    top_k_list = sorted(set(args.top_k))

    # Resolve prediction dirs
    pred_dirs = []
    if args.auto_discover:
        pred_dirs = auto_discover_phase6_models(base_dir)
        print(f"  Auto-discovered {len(pred_dirs)} prediction dirs")
    elif args.combo:
        all_combo_names = []
        for c in args.combo:
            all_combo_names.extend(c)
        pred_dirs = list(dict.fromkeys(resolve_phase6_models(base_dir, all_combo_names)))
        print(f"  Resolved {len(pred_dirs)} models from --combo")
    elif args.models:
        pred_dirs = resolve_phase6_models(base_dir, args.models)
        print(f"  Resolved {len(pred_dirs)} models from --models")
    else:
        print("ERROR: Provide --models, --auto_discover, or --combo")
        sys.exit(1)

    if not pred_dirs:
        print("ERROR: No prediction directories found")
        sys.exit(1)

    # Load all models
    all_models, test_cids = load_models(pred_dirs)
    model_names = sorted(all_models.keys())
    print(f"  Loaded {len(model_names)} models, {len(test_cids)} common test CIFs")

    # Ensure all models have all cids (for ensemble_discovery functions)
    all_models = _safe_scores(all_models, test_cids)

    # Build combos to run
    combos = []
    if args.combo or (args.auto_discover and args.combo):
        if args.auto_discover:
            combos.append(sorted(all_models.keys()))
        for c in (args.combo or []):
            resolved = resolve_phase6_models(base_dir, c)
            names = [os.path.basename(p.rstrip(os.path.sep)) for p in resolved]
            if all(n in all_models for n in names):
                combos.append(sorted(names))
            else:
                missing = [n for n in names if n not in all_models]
                print(f"  WARNING: Skipping combo {c}: missing models {missing}")
    if not combos and args.combo_size:
        for size in args.combo_size:
            for combo in combinations(model_names, size):
                combos.append(list(combo))
    if not combos:
        combos = [model_names]

    print(f"  Running {len(combos)} model combo(s)")
    print("=" * 70)

    combo_results = {}
    all_rankings = {}  # label -> sorted cids for agreement

    for i, combo in enumerate(combos):
        if args.auto_discover and i == 0 and set(combo) == set(all_models.keys()):
            combo_slug = "auto_discover"
        else:
            combo_slug = _run_slug(combo)
        if len(combos) > 1 and combo_slug in combo_results:
            combo_slug = f"{combo_slug}_{i + 1}"
        sub_models = {n: all_models[n] for n in combo}
        sub_models = _safe_scores(sub_models, test_cids)

        results = {}
        method_scores = {}  # method_name -> {cid: score}

        # RRF
        rrf_scores = run_rrf(sub_models, test_cids, k=args.rrf_k)
        rrf_sorted = sorted(rrf_scores.keys(), key=lambda c: rrf_scores[c])
        results["rrf_k60"] = rrf_sorted
        method_scores["rrf_k60"] = rrf_scores
        all_rankings[f"{combo_slug}_rrf"] = rrf_sorted

        # Rank avg
        ra_scores = run_rank_avg(sub_models, test_cids)
        ra_sorted = sorted(ra_scores.keys(), key=lambda c: ra_scores[c])
        results["rank_avg"] = ra_sorted
        method_scores["rank_avg"] = ra_scores
        all_rankings[f"{combo_slug}_rank_avg"] = ra_sorted

        # Vote top-K
        for vote_k in [50, 100, 200]:
            if vote_k > len(test_cids):
                continue
            v_scores = run_vote_top_k(sub_models, test_cids, vote_k)
            v_sorted = sorted(v_scores.keys(), key=lambda c: v_scores[c])
            mk = f"vote_top{vote_k}"
            results[mk] = v_sorted
            method_scores[mk] = v_scores
            all_rankings[f"{combo_slug}_{mk}"] = v_sorted

        # Score avg
        sa_scores = run_score_avg(sub_models, test_cids)
        sa_sorted = sorted(sa_scores.keys(), key=lambda c: sa_scores[c])
        results["score_avg"] = sa_sorted
        method_scores["score_avg"] = sa_scores
        all_rankings[f"{combo_slug}_score_avg"] = sa_sorted

        # Weighted RRF
        wrrf_scores = run_weighted_rrf(sub_models, test_cids, k=args.rrf_k)
        wrrf_sorted = sorted(wrrf_scores.keys(), key=lambda c: wrrf_scores[c])
        results["weighted_rrf"] = wrrf_sorted
        method_scores["weighted_rrf"] = wrrf_scores
        all_rankings[f"{combo_slug}_weighted_rrf"] = wrrf_sorted

        combo_results[combo_slug] = results

        # Write per-combo per-method files
        for method_name, sorted_cids in results.items():
            for k in top_k_list:
                top_cids = sorted_cids[:k]
                out_path = os.path.join(output_dir, f"{combo_slug}_{method_name}_top{k}.txt")
                with open(out_path, "w", encoding="utf-8") as f:
                    for cid in top_cids:
                        f.write(cid + "\n")
            scores = method_scores.get(method_name, {})
            csv_path = os.path.join(output_dir, f"{combo_slug}_{method_name}_predictions.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                f.write("rank,cif_id,score\n")
                for r, cid in enumerate(sorted_cids, 1):
                    f.write(f"{r},{cid},{scores.get(cid, 0):.6f}\n")

    # Add individual model rankings for agreement
    for name in model_names:
        scores = all_models[name]
        sorted_cids = sorted(scores.keys(), key=lambda c: scores[c])
        all_rankings[f"[single]_{name}"] = sorted_cids

    # Agreement analysis
    agreement_by_k = {}
    vote_per_cid_by_k = {}
    for k in top_k_list:
        jaccard_mat, labels, vote_per_cid = compute_agreement(all_rankings, k)
        agreement_by_k[k] = (jaccard_mat, labels, vote_per_cid)
        vote_per_cid_by_k[k] = vote_per_cid

    # Report
    write_report(output_dir, combo_results, agreement_by_k, vote_per_cid_by_k, top_k_list)

    # Plots
    plot_agreement_heatmap(output_dir, agreement_by_k, top_k_list)
    if 25 in top_k_list and 25 in vote_per_cid_by_k:
        plot_vote_distribution(output_dir, vote_per_cid_by_k[25], top_k=25)
    for k in top_k_list:
        plot_structure_overlap_heatmap(output_dir, all_rankings, k, max_cifs=50)

    # Save agreement JSON
    agreement_path = os.path.join(output_dir, "agreement_analysis.json")
    ser = {}
    for k in top_k_list:
        if k not in agreement_by_k:
            continue
        jm, labels, vpc = agreement_by_k[k]
        ser[k] = {
            "labels": labels,
            "jaccard_matrix": jm.tolist(),
            "vote_per_cid": {str(c): v for c, v in sorted(vpc.items(), key=lambda x: -x[1])[:100]},
        }
    with open(agreement_path, "w", encoding="utf-8") as f:
        json.dump(ser, f, indent=2)
    print(f"  Saved {agreement_path}")

    print("=" * 70)
    print("  Phase6 Ensemble Report complete.")
    print(f"  Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
