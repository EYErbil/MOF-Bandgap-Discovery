# MOF-Bandgap-Discovery

**Discovering low-bandgap Metal-Organic Frameworks through multi-model ensemble learning on MOFTransformer embeddings.**

This repository implements a complete, reproducible pipeline for screening Metal-Organic Frameworks (MOFs) with potentially conductive bandgaps (< 1.0 eV). It combines deep learning (MOFTransformer fine-tuning) with classical ML classifiers trained on pretrained embeddings, all fused via Reciprocal Rank Fusion (RRF) ensembles.

> **For newcomers:** A MOF (Metal-Organic Framework) is a porous crystalline material made of metal nodes connected by organic linkers. The *bandgap* measures how easily electrons can flow — a low bandgap (< 1.0 eV) suggests the material may conduct electricity, which is valuable for electronics, sensors, and catalysis. This pipeline finds those rare low-bandgap MOFs among thousands of candidates.

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Repository Structure](#repository-structure)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Quick Start (6 Commands)](#quick-start-6-commands)
6. [Detailed Walkthrough](#detailed-walkthrough)
   - [Step 1: Extract Embeddings & Create Splits](#step-1-extract-embeddings--create-splits)
   - [Step 2: Train Neural Network Regressors](#step-2-train-neural-network-regressors)
   - [Step 3: Train ML Classifiers & kNN Baselines](#step-3-train-ml-classifiers--knn-baselines)
   - [Step 4: Exhaustive Ensemble Ablation](#step-4-exhaustive-ensemble-ablation)
   - [Step 5: Generate Report](#step-5-generate-report)
   - [Step 6: Discovery on New MOFs](#step-6-discovery-on-new-mofs-optional)
7. [Customizing Experiments (Creating Your Own)](#customizing-experiments)
8. [Custom Ensembles (Choosing Your Own Model Combinations)](#custom-ensembles)
9. [Split Modification Workflow](#split-modification-workflow)
10. [Optional Analysis Scripts](#optional-analysis-scripts)
11. [Key Design Decisions](#key-design-decisions)
12. [Output Artifacts](#output-artifacts)
13. [Troubleshooting](#troubleshooting)
14. [Citation](#citation)
15. [License](#license)

---

## Pipeline Overview

```
 ┌───────────────────────────────────────────────────────────────────────────┐
 │                     MOF-Bandgap-Discovery Pipeline                       │
 ├───────────────────────────────────────────────────────────────────────────┤
 │                                                                           │
 │  STEP 1  Extract pretrained 768-dim CLS embeddings (MOFTransformer)      │
 │     │    Create Strategy D farthest-point train/val/test splits           │
 │     ▼                                                                     │
 │  STEP 2  Fine-tune MOFTransformer for bandgap regression                 │
 │     │    (3 seed variants: exp364, exp370, exp371)                        │
 │     ▼                                                                     │
 │  STEP 3  Train 15+ sklearn classifiers + kNN baselines                   │
 │     │    on pretrained embeddings (no GPU needed)                         │
 │     ▼                                                                     │
 │  STEP 4  Exhaustive ensemble ablation (all 2/3/4-model combos)           │
 │     │    Optimize recall@50 via RRF, rank averaging, voting              │
 │     ▼                                                                     │
 │  STEP 5  Generate comprehensive analysis report (15+ figures)            │
 │     ▼                                                                     │
 │  STEP 6  Phase6 Discovery — Inference on new unlabeled MOFs              │
 │          → Consensus top-25 candidates for DFT validation                │
 └───────────────────────────────────────────────────────────────────────────┘
```

**How the models work together:**

- **Neural Networks (Step 2):** Fine-tuned MOFTransformer models that learn to *predict* bandgap values directly. They understand atomic structure. Each model (3 different random seeds) may make slightly different predictions — this diversity helps the ensemble.

- **ML Classifiers (Step 3):** Scikit-learn models (Random Forest, SVM, etc.) trained on 768-dimensional embedding vectors extracted from the *pretrained* (not fine-tuned) MOFTransformer. They classify whether a MOF is likely low-bandgap based on its embedding proximity to known positives.

- **Ensemble (Step 4):** Combines all models using Reciprocal Rank Fusion (RRF). Each model produces a ranked list of MOFs. RRF merges these lists by giving high scores to MOFs ranked highly by *multiple* models, without requiring the raw scores to be on the same scale.

**Why two phases — validation then discovery:**

The pipeline has two distinct stages with different goals:

1. **Validation on labeled data (Steps 1–5):** We have ~6,000 MOFs with known DFT-computed bandgaps. We train models, then evaluate ensembles on a held-out test set where we *know* which MOFs are truly low-bandgap. The recall metrics and heatmaps from this phase prove that our ensemble can reliably push true low-bandgap MOFs to the top of a ranked list. This is the evidence that the models actually work — without it, deploying them on new data would be unjustified.

2. **Discovery on unlabeled data (Step 6):** Once validated, we deploy the ensemble on a much larger set of MOFs whose bandgaps are unknown. The output is an enriched shortlist — the top 25, 50, or 100 candidates most likely to be low-bandgap — ranked by consensus across all models. Computing a MOF's true bandgap via DFT typically costs hours to days of CPU time per structure. By pre-screening thousands of candidates down to a high-confidence shortlist, the ensemble reduces the number of expensive DFT calculations by orders of magnitude while concentrating discovery on the structures most likely to be electronically interesting (conductive, narrow-gap semiconductor, photocatalytic, etc.).

In short: Steps 1–5 answer *"can we find the needles in the haystack?"* using ground truth. Step 6 answers *"where are the needles in a new haystack?"* using the validated models.

---

## Repository Structure

```
MOF-Bandgap-Discovery/
├── README.md                             # This file
├── LICENSE                               # MIT License
├── requirements.txt                      # Python dependencies
├── .gitignore
│
├── src/                                  # Core Python modules (shared library)
│   ├── train_regressor.py                #   MOFTransformer fine-tuning + metrics
│   ├── embedding_classifier.py           #   15+ sklearn classifier methods
│   ├── ensemble_discovery.py             #   RRF / exhaustive ensemble ablation
│   ├── knn_baseline.py                   #   kNN regression & similarity baselines
│   ├── generate_final_report.py          #   Analysis figures & markdown report
│   ├── compare_results.py                #   Cross-method comparison reports
│   ├── verify_ml_heatmap.py              #   ML performance heatmap verification
│   ├── reinfer_nn.py                     #   Re-run NN inference from checkpoints
│   ├── predict_with_embedding_classifier.py  # Predict on new data with saved models
│   ├── extract_posttrain_embeddings.py   #   Extract post-finetune embeddings
│   ├── pormake_screen.py                 #   Structural screening with PORMAKE
│   ├── report_split_bandgap_distribution.py  # Split quality analysis
│   ├── ml_training_plots.py              #   ML training analysis plots
│   ├── umap_analysis_split_d.py          #   UMAP visualization (Strategy D split)
│   ├── umap_analysis_original_split.py   #   UMAP on original split
│   └── umap_posttrain.py                #   UMAP on post-finetune embeddings
│
├── data_preparation/                     # Embedding extraction & data splitting
│   ├── analyze_embeddings.py             #   Extract 768-dim pretrained embeddings
│   ├── embedding_split.py                #   Strategy D farthest-point splits
│   ├── resplit_data.py                   #   Re-create splits from scratch
│   ├── repair_split_symlinks.py          #   Fix broken symlinks in split dirs
│   ├── fix_split_symlinks.sh             #   Shell-based symlink repair helper
│   └── extract_phase6_embeddings_pretrained.py  # Embeddings for new MOFs
│
├── experiments/                          # NN experiment configurations (human-editable)
│   ├── exp364_fulltune/                  #   seed=42, freeze=0, primary model
│   │   ├── run.py                        #     ★ Hyperparameter config — edit this
│   │   └── run.sh                        #     SLURM submission script
│   ├── exp370_seed2/                     #   seed=123, ensemble variant
│   │   ├── run.py
│   │   └── run.sh
│   └── exp371_seed3/                     #   seed=456, ensemble variant
│       ├── run.py
│       └── run.sh
│
├── discovery/                            # Phase6: inference on unlabeled MOFs
│   ├── run_inference_from_cwd.py         #   NN inference from checkpoint
│   ├── phase6_discovery.py               #   Per-model + ensemble ranking
│   ├── ensemble_phase6_predictions.py    #   RRF + rank averaging on new data
│   ├── phase6_ensemble_report.py         #   Full discovery report + agreement
│   └── collect_inference_structures.sh   #   Gather MOF files from scattered dirs
│
├── scripts/                              # SLURM pipeline orchestration
│   ├── config.sh                         #   ★ Centralized cluster configuration
│   ├── 01_extract_embeddings.sh          #   Extract embeddings + create splits
│   ├── 02_train_nn.sh                    #   Fine-tune MOFTransformer (3 seeds)
│   ├── 03_train_ml.sh                    #   Train sklearn classifiers + kNN
│   ├── 04_run_ensemble.sh                #   Exhaustive ensemble ablation
│   ├── 05_generate_report.sh             #   Comprehensive analysis report
│   ├── 06_run_discovery.sh               #   Phase6 — full inference pipeline
│   └── optional/
│       ├── run_umap_analysis.sh          #   UMAP embedding visualizations
│       ├── run_verify_ml.sh              #   ML heatmap verification
│       ├── run_reinfer.sh                #   Re-infer NN from checkpoints
│       ├── run_screening.sh              #   Structural screening
│       ├── run_phase6_ml_only.sh         #   Phase6 ML inference only (no GPU)
│       ├── run_phase6_nn_only.sh         #   Phase6 NN inference only (GPU)
│       └── run_phase6_ensemble_custom.sh #   Phase6 custom model combination
│
├── tools/                                # Split modification utilities
│   ├── move_test_negatives_to_val.py     #   Move samples between splits
│   ├── move_val_to_test.py               #   Rebalance val/test sets
│   └── fix_split_symlinks.sh             #   Repair symlinks after moving
│
└── data/                                 # Data directory (not tracked in Git)
    └── README.md                         #   Dataset format documentation
```

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **SLURM cluster** | GPU nodes with CUDA 12.x (for Steps 1, 2, 6c). Steps 3-5 need CPU only. |
| **Python 3.9+** | Tested with Python 3.9.5 |
| **MOFTransformer** | See [MOFTransformer repo](https://github.com/hspark1212/MOFTransformer). Provides the pretrained transformer model + data loaders. |
| **MOF structure files** | Preprocessed into MOFTransformer format: `.grid`, `.griddata16`, `.graphdata` per MOF. See [data/README.md](data/README.md) for format details. |
| **Bandgap labels** | JSON files mapping CIF IDs to bandgap values in eV. See [data/README.md](data/README.md). |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/MOF-Bandgap-Discovery.git
cd MOF-Bandgap-Discovery

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate       # On Linux/Mac
# venv\Scripts\activate        # On Windows

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install MOFTransformer (follow their documentation for GPU support)
pip install moftransformer
```

### Cluster Setup

On a SLURM cluster, edit `scripts/config.sh` — this is the **only file** that contains cluster-specific paths. Every SLURM script sources it automatically:

```bash
# scripts/config.sh — edit these 4 lines:
export BASE_DIR="/path/to/MOF-Bandgap-Discovery"    # Where you cloned the repo
export VENV_PATH="/path/to/venv/bin/activate"        # Your Python virtualenv
export SLURM_PARTITION_GPU="ai"                      # Your GPU partition name
export MODULE_LOADS="cuda/12.3 cudnn/8.9.5 python/3.9.5"  # Your module system
```

All other paths (data, experiments, results) are automatically derived from `BASE_DIR`.

---

## Quick Start (6 Commands)

After installation and data preparation:

```bash
# 1. Configure your cluster (one-time)
#    Edit scripts/config.sh with your paths

# 2. Run the full pipeline — each step depends on the previous one:
sbatch scripts/01_extract_embeddings.sh     # GPU, ~2-4h   — embeddings + splits
sbatch scripts/02_train_nn.sh               # GPU, ~24-69h — 3 NN experiments
sbatch scripts/03_train_ml.sh               # CPU, ~2-6h   — 15+ sklearn classifiers
sbatch scripts/04_run_ensemble.sh           # CPU, ~1-2h   — exhaustive ensemble search
sbatch scripts/05_generate_report.sh        # CPU, ~15min  — analysis report
sbatch scripts/06_run_discovery.sh          # GPU, ~4-8h   — inference on new MOFs (optional)
```

Wait for each job to complete before submitting the next. Check job status with:
```bash
squeue -u $USER
```

---

## Detailed Walkthrough

### Step 1: Extract Embeddings & Create Splits

```bash
sbatch scripts/01_extract_embeddings.sh
```

| Substep | What happens | Output |
|---------|-------------|--------|
| 1a | Loads the **pretrained** MOFTransformer (no fine-tuning) and extracts a 768-dimensional CLS embedding vector for every MOF in the dataset. These embeddings capture structural and chemical information learned during MOFTransformer's pretraining. | `data/embeddings/embeddings_pretrained.npz` |
| 1b | Creates **Strategy D** train/val/test splits using farthest-point sampling in embedding space. This guarantees every positive (low-bandgap) MOF in val/test has at least one structurally similar positive in the training set. | `data/splits/strategy_d_farthest_point/` |

**Why Strategy D?** Standard random splits can leave validation positives completely isolated — surrounded only by high-bandgap MOFs in embedding space. If the model has never seen a similar low-bandgap MOF during training, high recall on that positive is meaningless. Strategy D uses farthest-point coverage to methodically ensure structural diversity in every split, yielding honest evaluation metrics.

---

### Step 2: Train Neural Network Regressors

```bash
sbatch scripts/02_train_nn.sh
```

Fine-tunes MOFTransformer for bandgap regression. Three experiments with different random seeds create diverse models for the ensemble:

| Experiment | Seed | Purpose |
|------------|------|---------|
| `exp364_fulltune` | 42 | Primary model |
| `exp370_seed2` | 123 | Ensemble diversity |
| `exp371_seed3` | 456 | Ensemble diversity |

**Shared hyperparameters** (configured in each experiment's `run.py`):

| Parameter | Value | What it does |
|-----------|-------|-------------|
| `loss_type` | `"huber"` | Huber loss — robust to outlier bandgaps (unlike MSE) |
| `pooling_type` | `"mean"` | Average over all atom token embeddings (not just the CLS token) |
| `learning_rate` | `1e-4` | Base LR for the pretrained transformer backbone |
| `lr_mult` | `10.0` | The regression head trains at 10× the backbone LR |
| `es_monitor` | `"val/spearman_rho"` | Early stopping tracks ranking quality, not loss magnitude |
| `es_mode` | `"max"` | Higher Spearman ρ = better ranking = stop improving |
| `patience` | `15` | Stop if val/spearman_rho doesn't improve for 15 epochs |
| `freeze_layers` | `0` | All transformer layers are fine-tuned (not frozen) |
| `batch_size` | `32` | Total batch size (split across GPUs as `per_gpu_batchsize=8`) |
| `max_epochs` | `100` | Hard cap; early stopping usually triggers around epoch 30-50 |
| `weight_decay` | `0.01` | L2 regularization |
| `use_sample_weights` | `False` | No upweighting of minority class |

**Outputs per experiment** (saved in `experiments/<exp_name>/`):

| File | Description |
|------|-------------|
| `best_es-*.ckpt` | Best checkpoint by validation Spearman ρ |
| `test_predictions.csv` | Per-MOF bandgap predictions (columns: `cif_id`, `target`, `prediction`) |
| `final_results.json` | All test metrics (Spearman, MAE, recall@K, etc.) |
| `training_dashboard.png` | Loss/metric curves over training |
| `discovery_curve.png` | Recall vs. top-K plot |

> **To train experiments in parallel** (faster on multi-GPU clusters):
> ```bash
> cd experiments/exp364_fulltune && sbatch run.sh
> cd experiments/exp370_seed2   && sbatch run.sh
> cd experiments/exp371_seed3   && sbatch run.sh
> ```

See [Customizing Experiments](#customizing-experiments) to create your own experiment configurations.

---

### Step 3: Train ML Classifiers & kNN Baselines

```bash
sbatch scripts/03_train_ml.sh
```

Trains **15+ classifiers** directly on the 768-dim pretrained embeddings. **No GPU needed** — these are traditional sklearn models that treat each MOF as a 768-feature vector.

**Why use pretrained embeddings (not fine-tuned)?** The pretrained MOFTransformer CLS token already captures rich structural information. Training sklearn classifiers on these gives a complementary signal — they find positives by structural *similarity* rather than learned bandgap prediction.

**Base methods:**

| Method | What it does |
|--------|-------------|
| Logistic Regression | Linear boundary in embedding space |
| SVM-RBF | Non-linear kernel boundary |
| Random Forest | Ensemble of decision trees on embedding features |
| Extra Trees | Like RF but with randomized split thresholds |
| Gradient Boosting | Sequential boosted trees |
| XGBoost | Optimized gradient boosting |
| LDA | Linear Discriminant Analysis |
| Mahalanobis distance | Distance to positive class centroid |
| Gaussian Mixture Model | Density estimation for positive class |
| Isolation Forest | Anomaly detection (positives as anomalies) |

**Enhanced methods** (enabled with `--enhanced`):

| Method | What it does |
|--------|-------------|
| SMOTE-RF, SMOTE-ET, SMOTE-LR | Synthetic minority oversampling before training |
| Two-stage (kNN → Extra Trees) | Pre-filter with kNN, then classify with ET |
| Feature-selected variants | Automatic feature selection before classification |

**kNN baselines:**

| Method | What it does |
|--------|-------------|
| k-NN regression | Distance-weighted average of k nearest neighbors' bandgaps |
| Similarity-to-positive | Max cosine similarity to any training positive |
| Hybrid (NN + kNN) | Combine neural network score with kNN score |
| Novelty-aware | Flag MOFs that are far from all training data |

**Outputs:**
- `data/embedding_classifiers/strategy_d_farthest_point/<method>/model.joblib` — Saved model
- `data/embedding_classifiers/strategy_d_farthest_point/<method>/test_predictions.csv` — Predictions
- `data/knn_results/strategy_d_farthest_point/test_predictions.csv` — kNN predictions

---

### Step 4: Exhaustive Ensemble Ablation

```bash
sbatch scripts/04_run_ensemble.sh
```

This is where the pipeline finds the **optimal combination of models**. It tests *every possible 2/3/4-model subset* from all trained models and evaluates which combination maximizes **recall@50** (the fraction of true positives found in the top 50 candidates).

**This script runs three phases:**

**Phase C — Exhaustive search:**
- Collects all models with `test_predictions.csv` (NN experiments + sklearn + kNN)
- Tests every 2-model, 3-model, and 4-model combination
- For each combination, applies 5 ensemble methods:

  | Ensemble method | How it works |
  |----------------|-------------|
  | **Reciprocal Rank Fusion** (RRF, k=60) | Score = Σ 1/(k + rank_i). Robust to different score scales. |
  | **Rank averaging** | Average of per-model ranks |
  | **Top-K voting** | Count how many models place the MOF in their top-K |
  | **Score averaging** | Average of min-max normalized scores |
  | **Weighted RRF** | Like RRF but weights models by individual quality |

- Reports the best 2/3/4-model combinations in `recommended_combinations.txt`

**Phase D — Selective ensemble:**
- Uses only signal-bearing models (those with recall significantly above random baseline)
- Provides a curated default ensemble

**Phase E — Comparison report:**
- Cross-method comparison of all individual models and ensemble results

**Outputs:**
- `data/ensemble_results/exhaustive/` — All exhaustive search results
- `data/ensemble_results/selective/` — Signal-bearing models only
- `data/comparison_report/` — Cross-method comparison

---

### Step 5: Generate Report

```bash
sbatch scripts/05_generate_report.sh
```

Reads all results and produces **15+ figures with full metrics** and a markdown summary:

- Model leaderboard (ensemble vs. individual methods)
- Discovery confusion matrices (top-K)
- Recall/precision curves per model
- Complementarity analysis (which MOFs does each model uniquely discover?)
- Bandgap distribution and metal center breakdowns
- Top-20 candidate lists with structural metadata

**Output:** `data/final_results/` (summary.md + fig*.png)

---

### Step 6: Discovery on New MOFs (Optional)

```bash
sbatch scripts/06_run_discovery.sh
```

Applies ALL trained models to a **new, unlabeled** MOF dataset and produces a consensus ranking of candidates for DFT validation.

**This is the deployment step** — where you use your trained pipeline to discover real candidates.

**Before running**, prepare the new data:

1. Place MOF structure files in `data/phase6/` (see [data/README.md](data/README.md))
2. Create `data/phase6/test_bandgaps_regression.json` with CIF IDs as keys (bandgap values can be `0.0` as placeholders since they are unknown)
3. Optionally, use `discovery/collect_inference_structures.sh` to gather scattered structure files into the right directory

**What 06_run_discovery.sh does (5 substeps):**

| Substep | Action | Output |
|---------|--------|--------|
| 6a | Extract pretrained embeddings for new MOFs | `data/phase6/embedding_analysis/Phase6_embeddings.npz` |
| 6b | Score with saved sklearn models | `data/phase6/ml_predictions/<method>/test_predictions.csv` |
| 6c | Run NN forward pass from checkpoints | `experiments/<exp>/inference_predictions.csv` |
| 6d | Ensemble all predictions → consensus top-25 | `data/phase6/inference_results/top25_for_DFT_rrf.txt` |
| 6e | Agreement analysis across models | `data/phase6/ensemble_report/phase6_ensemble_report.md` |

**Configuration** (edit at the top of the script):
```bash
NN_EXPERIMENTS="exp364_fulltune exp370_seed2 exp371_seed3"  # Which NN models to use
ML_METHODS="extra_trees random_forest logistic_regression smote_extra_trees"  # Which ML models
TOP_K=25  # How many top candidates to report
```

**Running individual substeps independently:**

If you want to run ML or NN inference separately (e.g., ML first on CPU, NN later when a GPU is free), use the optional per-step scripts:

```bash
# ML inference only (CPU, no GPU needed)
sbatch scripts/optional/run_phase6_ml_only.sh

# NN inference only (GPU)
sbatch scripts/optional/run_phase6_nn_only.sh

# Custom ensemble — pick exactly which models to combine
# Edit ML_MODELS and NN_EXPERIMENTS inside the script first
sbatch scripts/optional/run_phase6_ensemble_custom.sh
```

These are subsets of `06_run_discovery.sh` for when you need more control.

---

### What the Results Mean

**From the labeled split (Steps 2–5):**

The ensemble and individual models are evaluated on a test set where every MOF has a known DFT-computed bandgap. The key outputs are:

- **Recall@K heatmaps** — for each model and ensemble, what fraction of the true low-bandgap MOFs (bandgap < 1.0 eV) appear in the top K predictions? A high recall@25 means the model pushes real positives to the very top of its ranked list. The heatmap across all models and ensemble methods shows exactly which combinations are best at recovering known positives.
- **Complementarity analysis** — which MOFs does each model *uniquely* discover? If Model A finds 8 out of 10 positives and Model B finds a different 8 out of 10, their ensemble may find 10 out of 10. The report quantifies this overlap.
- **Confusion matrices and precision/recall curves** — standard classification diagnostics on the binary "low-bandgap vs. not" task, so you can characterize false positive rates and model confidence.

These results establish **trust in the ensemble** before deploying it on unknown data. If recall@50 on the labeled test set is, say, 80%, you have concrete evidence that the models rank true positives highly — and can expect similar enrichment on new data drawn from the same chemical space.

**From unlabeled discovery (Step 6):**

When the validated ensemble is applied to new MOFs with unknown bandgaps, there is no ground truth to compute recall against. Instead, the outputs focus on **consensus and confidence**:

- **`top25_for_DFT_rrf.txt`** — the 25 MOFs ranked highest by the RRF ensemble. These are the structures where multiple independent models agree: "this MOF is very likely low-bandgap." This is the shortlist to prioritize for DFT validation.
- **Agreement heatmaps** — which models agree on which candidates? A MOF flagged by 6 out of 7 models is a stronger candidate than one flagged by 2. The heatmap makes this visible at a glance.
- **Per-model rankings** — individual model predictions are preserved so researchers can inspect whether a candidate is favored primarily by structural-similarity classifiers, by the regression NN, or by both.

The practical outcome: instead of running DFT on all N thousand candidate MOFs (each calculation costing hours to days of compute time), you run DFT on 25–100 high-confidence candidates selected by the ensemble. This focuses expensive computational resources on the structures most likely to exhibit the target property — low bandgap, and by extension, potential conductivity, photocatalytic activity, or suitability for electronic device integration.

---

## Customizing Experiments

Each experiment lives in its own directory under `experiments/` and is defined by a single file: **`run.py`**. This is the human-editable hyperparameter interface — the place where you control how the neural network trains.

### How experiments work

```
experiments/exp364_fulltune/
├── run.py      ← You edit this: all hyperparameters are plain Python arguments
└── run.sh      ← SLURM submission script (usually no edits needed)
```

`run.py` imports the `run()` function from `src/train_regressor.py` and calls it with your chosen hyperparameters. The training script handles everything else: data loading, model creation, training loop, early stopping, test evaluation, and result saving.

### Creating a new experiment

1. **Copy an existing experiment directory:**
   ```bash
   cp -r experiments/exp364_fulltune experiments/exp999_my_experiment
   ```

2. **Edit `run.py`** — change any hyperparameters you want to explore:
   ```python
   # experiments/exp999_my_experiment/run.py
   run(
       data_dir=args.data_dir,
       downstream="bandgaps_regression",
       threshold=1.0,
       loss_type="huber",          # Try: "mse", "huber", "mae"
       pooling_type="mean",        # Try: "mean", "cls"
       freeze_layers=0,            # Try: 0 (full finetune), 1, 2, 3 (freeze bottom N)
       use_sample_weights=False,   # Try: True (upweight rare positives)
       es_monitor="val/spearman_rho",
       es_mode="max",
       batch_size=32,
       per_gpu_batchsize=8,
       learning_rate=1e-4,         # Try: 5e-5, 1e-4, 3e-4
       weight_decay=0.01,
       lr_mult=10.0,               # Regression head LR multiplier
       max_epochs=100,
       patience=15,                # Early stopping patience
       log_dir=".",
       seed=42,                    # Change seed for ensemble diversity
       num_workers=4,
   )
   ```

3. **Run it:**
   ```bash
   cd experiments/exp999_my_experiment
   sbatch run.sh
   # or via the central script with --exp:
   sbatch scripts/02_train_nn.sh --exp exp999_my_experiment
   ```

4. **After training**, the experiment directory will contain:
   ```
   experiments/exp999_my_experiment/
   ├── run.py                      # Your config (unchanged)
   ├── run.sh                      # SLURM script
   ├── best_es-*.ckpt              # Best checkpoint
   ├── test_predictions.csv        # Per-MOF predictions ← used by ensemble
   ├── final_results.json          # All metrics
   ├── training_dashboard.png      # Training curves
   └── discovery_curve.png         # Recall vs top-K
   ```

5. **The new experiment is automatically picked up by the ensemble.** When you re-run `04_run_ensemble.sh`, it discovers all experiments with `test_predictions.csv` and includes them in the ablation search.

### Key parameters to tune

| Parameter | Effect | Suggested range |
|-----------|--------|----------------|
| `seed` | Different random initialization → different model for ensemble diversity | Any integer |
| `freeze_layers` | 0 = full finetune (best), 1-3 = freeze bottom layers (faster, less overfitting risk) | 0-3 |
| `learning_rate` | Base LR for transformer backbone | 5e-5 to 3e-4 |
| `lr_mult` | How much faster the regression head trains vs backbone | 1.0 to 20.0 |
| `loss_type` | `"huber"` is robust to outliers; `"mse"` penalizes large errors more | `"huber"`, `"mse"` |
| `pooling_type` | `"mean"` averages all atom tokens; `"cls"` uses only CLS token | `"mean"`, `"cls"` |
| `use_sample_weights` | Upweight rare positives if class imbalance is severe | `True`, `False` |
| `patience` | Epochs without improvement before stopping | 10-30 |

---

## Custom Ensembles

### On labeled data (Step 4)

The default `04_run_ensemble.sh` tests *every* model combination exhaustively. But if you want to run a specific subset:

```bash
# Edit the PRED_DIRS variable in the script, or call ensemble_discovery.py directly:
python src/ensemble_discovery.py \
    --prediction_dirs \
        experiments/exp364_fulltune \
        experiments/exp370_seed2 \
        data/embedding_classifiers/strategy_d_farthest_point/extra_trees \
        data/embedding_classifiers/strategy_d_farthest_point/smote_extra_trees \
    --output_dir data/ensemble_results/my_custom_combo \
    --threshold 1.0 \
    --rrf_k 60 \
    --ablation
```

Each directory passed to `--prediction_dirs` must contain a `test_predictions.csv`.

### On unlabeled data (Phase6)

Use the custom ensemble script described in [Step 6](#step-6-discovery-on-new-mofs-optional):

```bash
# 1. Edit the model selection at the top of the script:
#    ML_MODELS="extra_trees smote_extra_trees smote_random_forest"
#    NN_EXPERIMENTS="exp364_fulltune exp370_seed2"
#    USE_NN=1

# 2. Submit:
sbatch scripts/optional/run_phase6_ensemble_custom.sh
```

The output directory is auto-named based on your selection (e.g., `ensemble_results/custom_extra_trees_smote_extra_trees_exp364_fulltune_exp370_seed2/`).

---

## Split Modification Workflow

If you need to modify the train/val/test splits after initial training (e.g., move a specific MOF from test to val), use the tools in `tools/`:

```
Step 1: Modify the split
─────────────────────────
  python tools/move_test_negatives_to_val.py   # Move negatives from test → val
  # or
  python tools/move_val_to_test.py             # Move samples from val → test

Step 2: Fix file symlinks
─────────────────────────
  bash tools/fix_split_symlinks.sh             # Repair broken symlinks from the move

Step 3: Re-run NN inference (model weights are NOT retrained — only test set changed)
──────────────────────────
  sbatch scripts/optional/run_reinfer.sh       # Re-produces test_predictions.csv
                                               # from existing checkpoints

Step 4: Re-run ensemble
───────────────────────
  sbatch scripts/04_run_ensemble.sh            # Recompute ensemble with new predictions
```

**Important:** `run_reinfer.sh` does NOT retrain the models. It loads the saved checkpoints and re-runs the test inference on the updated test set. This is fast (~5-10 min on GPU) and verifies that predictions for unchanged MOFs are identical to before (within GPU non-determinism tolerance ~1e-5).

---

## Optional Analysis Scripts

These scripts provide additional analysis and visualization. None are required for the core pipeline.

```bash
# UMAP embedding visualizations — see how MOFs cluster in 2D
sbatch scripts/optional/run_umap_analysis.sh

# ML performance heatmap — verify classifier performance across metrics
sbatch scripts/optional/run_verify_ml.sh

# Re-infer NN predictions — recompute test_predictions.csv from checkpoints
# (use after modifying splits, see Split Modification Workflow above)
sbatch scripts/optional/run_reinfer.sh

# Structural screening with PORMAKE — filter by pore size, surface area, etc.
sbatch scripts/optional/run_screening.sh

# Phase6 — run ML models only (CPU, no GPU needed)
sbatch scripts/optional/run_phase6_ml_only.sh

# Phase6 — run NN models only (GPU)
sbatch scripts/optional/run_phase6_nn_only.sh

# Phase6 — build a custom ensemble from hand-picked models
# Edit ML_MODELS and NN_EXPERIMENTS at the top of the script first
sbatch scripts/optional/run_phase6_ensemble_custom.sh
```

---

## Key Design Decisions

| Decision | Why |
|----------|-----|
| **Strategy D farthest-point split** | Standard random splits create train/val/test sets where some val positives have no structurally similar training positive. Strategy D samples farthest points first, guaranteeing coverage across embedding space. This yields honest recall metrics. |
| **Pretrained embeddings for ML classifiers** | The 768-dim MOFTransformer CLS token is a powerful structural fingerprint *before* any fine-tuning. Training sklearn classifiers on these gives a complementary retrieval signal — they find positives by structural similarity rather than learned bandgap regression. |
| **Multi-seed NN training** | Same architecture, same data, different random seeds → models that agree on easy cases but disagree on hard ones. This diversity is what makes ensembling powerful. |
| **Reciprocal Rank Fusion (RRF)** | Different models produce scores on different scales (regression logits vs. classification probabilities vs. distance metrics). RRF works on *ranks*, not scores, so it fuses heterogeneous models without normalization artifacts. k=60 is a standard choice. |
| **Exhaustive ablation** | With ~20 models and max combo size 4, there are ~6000 combinations. Testing all of them guarantees finding the true optimum rather than a locally greedy solution. This takes ~30 minutes on a single CPU. |
| **Bandgap < 1.0 eV threshold** | Conventional threshold for identifying potentially conductive or narrow-gap semiconductor MOFs. MOFs below this threshold are candidates for electronic applications, photocatalysis, and sensor design. |
| **Huber loss for regression** | The QMOF bandgap distribution has a long tail of high-bandgap MOFs (some > 8 eV). Huber loss caps the gradient for large errors, preventing these outliers from dominating training. |
| **Early stopping on Spearman ρ** | We care about *ranking* MOFs correctly (top-K discovery), not predicting exact bandgap values. Spearman ρ measures rank correlation, aligning the early stopping criterion with our actual goal. |

---

## Output Artifacts

After a full pipeline run (Steps 1-5), the following outputs are produced:

```
data/
├── embeddings/
│   └── embeddings_pretrained.npz             # 768-dim embeddings for all MOFs
│
├── splits/strategy_d_farthest_point/
│   ├── train/ val/ test/                     # Split structure files (symlinks)
│   └── {train,val,test}_bandgaps_regression.json   # Labels per split
│
├── embedding_classifiers/strategy_d_farthest_point/
│   └── <method>/                             # One dir per sklearn method
│       ├── model.joblib                      #   Saved model (reusable)
│       ├── scaler.joblib                     #   Feature scaler
│       ├── test_predictions.csv              #   Predictions ← used by ensemble
│       └── final_results.json                #   Metrics
│
├── knn_results/strategy_d_farthest_point/
│   ├── test_predictions.csv                  # kNN predictions
│   └── knn_hybrid_results.json               # Metrics
│
├── ensemble_results/
│   ├── exhaustive/                           # All 2/3/4-model combo results
│   │   └── RRF/
│   │       ├── ensemble_results.json         # Full metrics
│   │       ├── recommended_combinations.txt  # ★ Best model subsets
│   │       └── top{25,50,100}_for_discovery.txt
│   └── selective/                            # Signal-bearing models only
│
├── final_results/                            # Comprehensive analysis report
│   ├── summary.md                            # Text report
│   └── fig*.png                              # 15+ figures
│
└── comparison_report/
    ├── comparison_report.json                # All models compared
    └── comparison_summary.csv                # Sortable summary table

experiments/
├── exp364_fulltune/
│   ├── best_es-*.ckpt                        # NN checkpoint
│   ├── test_predictions.csv                  # NN predictions
│   └── final_results.json                    # NN metrics
├── exp370_seed2/...
└── exp371_seed3/...
```

After Step 6 (Phase6 discovery):

```
data/phase6/
├── embedding_analysis/Phase6_embeddings.npz  # Embeddings for new MOFs
├── ml_predictions/<method>/test_predictions.csv
├── inference_results/
│   ├── top25_for_DFT_rrf.txt                # ★ Top candidates by RRF
│   ├── top25_for_DFT_rank_avg.txt           # Top candidates by rank avg
│   └── inference_predictions.csv             # All predictions
└── ensemble_report/
    ├── phase6_ensemble_report.md             # Full report
    └── agreement_heatmap_top25.png           # Model agreement visualization
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'moftransformer'` | Install MOFTransformer: `pip install moftransformer`. Make sure you're in the correct virtualenv. |
| `FileNotFoundError: embeddings_pretrained.npz` | Run Step 01 first. The embeddings must be extracted before any ML or ensemble step. |
| `No test_predictions.csv found` for an experiment | The NN training may have failed or not finished. Check `experiments/<exp>/logs/` for error output. |
| `CUDA out of memory` during NN training | Reduce `per_gpu_batchsize` in the experiment's `run.py` (e.g., from 8 to 4). Total `batch_size` will be accumulated via gradient accumulation. |
| Ensemble finds no models | Ensure Steps 02 + 03 completed successfully and each model has `test_predictions.csv`. |
| Phase6 discovery produces empty results | Check that `data/phase6/test_bandgaps_regression.json` exists and that MOF structure files (.grid, .griddata16, .graphdata) are in the right location. |
| Symlinks broken after moving files between splits | Run `bash tools/fix_split_symlinks.sh` to repair, then `sbatch scripts/optional/run_reinfer.sh` to recompute predictions. |
| Different results on re-run | GPU non-determinism in CUDA/cuDNN causes tiny floating-point differences (~1e-5). Use `--deterministic` flag in `reinfer_nn.py` for exact reproducibility (slower). |
| Cannot find structure files for Phase6 | Use `bash discovery/collect_inference_structures.sh` — it searches multiple directories and gathers all 3 files per MOF into `data/phase6/test/`. |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{mof-bandgap-discovery,
  title  = {MOF-Bandgap-Discovery: Multi-Model Ensemble Learning for
            Low-Bandgap Metal--Organic Framework Screening},
  author = {Erbil, Ege Yi\u011fit},
  year   = {2025},
  note   = {Ko\c{c} University},
  url    = {https://github.com/<your-username>/MOF-Bandgap-Discovery}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
