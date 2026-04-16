# MOF-Bandgap-Discovery

**Discovering low-bandgap Metal-Organic Frameworks through diversity-aware ensemble learning on PMTransformer embeddings.**

This repository implements a reproducible pipeline for screening Metal-Organic Frameworks (MOFs) with potentially conductive bandgaps (< 1.0 eV). It combines a fine-tuned PMTransformer (the [MOFTransformer](https://github.com/hspark1212/MOFTransformer) architecture) with classical ML classifiers trained on pretrained embeddings, fused via Reciprocal Rank Fusion (RRF). Final DFT candidates are selected using a diversity-aware nomination strategy that leverages SOAP descriptors to maximize structural spread across the shortlist.

---

## Key Contribution

The pipeline extracts two complementary prediction signals from a single foundation model:

1. **Embedding-based ML classifiers** -- Fixed 768-dim representations from the pretrained (non-fine-tuned) PMTransformer encoder are used to train lightweight tree classifiers (Extra Trees, Random Forest with SMOTE). This leverages the general structural knowledge learned during pretraining on ~660K MOFs, without modifying the encoder.

2. **Fine-tuned PMTransformer** -- The full model is fine-tuned end-to-end for bandgap regression, adapting both encoder and prediction head to the target property.

These two approaches capture complementary signal: the trees operate on frozen general-purpose features while the fine-tuned model has task-adapted features. Their predictions are fused via RRF, and NN-ML disagreement provides an uncertainty signal.

**Diversity-aware candidate nomination (Step 7):** Rather than selecting the top-K candidates by score alone, the nomination pipeline clusters the RRF shortlist in embedding space and applies multiple diversity-aware strategies (cluster-quota round-robin, Maximal Marginal Relevance, uncertainty-weighted selection). When SOAP descriptors are used as the diversity space instead of PMTransformer embeddings, the resulting nominees achieve greater structural spread -- SOAP measures purely geometric/chemical similarity independent of the learned representations used for scoring.

---

## Pipeline Overview

```
  STEP 0   Preprocess CIF files into MOFTransformer format (.grid, .griddata16, .graphdata)
    v
  STEP 1   Extract pretrained 768-dim embeddings → Strategy D farthest-point splits
    v
  STEP 2   Fine-tune PMTransformer for bandgap regression (3 seeds)
    |        ↳ trains on raw MOFTransformer files, NOT on extracted embeddings
    v
  STEP 3   Train 15+ sklearn classifiers + kNN on pretrained embeddings
    |        ↳ uses the frozen 768-dim embeddings from Step 1
    v
  STEP 4   Exhaustive ensemble ablation (all 2/3/4-model combos, optimise recall@50)
    v
  STEP 5   Generate comprehensive analysis report (15+ figures)
    v
  ─ ─ ─ ─ ─ ─  labeled set complete, now switch to unlabeled set  ─ ─ ─ ─ ─ ─
    v
  STEP 6   Discovery -- deploy models on ~10K NEW unlabeled MOFs, RRF ranking
    v
  STEP 7   Diversity-aware DFT nomination (cluster + MMR + SOAP verification)
```

> **Labeled vs. unlabeled sets.** Steps 1-5 work on ~10K MOFs with known HSE bandgaps (only ~76 positives -- a needle-in-a-haystack retrieval problem). Steps 6-7 work on a **completely separate** set of ~10K unlabeled MOFs that the models have never seen during training, validation, or testing. These are not the test split from Steps 1-5; they are new structures for which we want to discover low-bandgap candidates.

> **NN vs. ML data flow.** The fine-tuned NN (Step 2) reads the raw preprocessed MOF files directly -- MOFTransformer handles tokenisation internally. The ML classifiers (Step 3) train on the frozen 768-dim pretrained embeddings extracted in Step 1. Both paths use the same train/val/test split.

---

## Repository Structure

```
MOF-Bandgap-Discovery/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── src/                        # Core Python modules
│   ├── train_regressor.py      #   PMTransformer fine-tuning + metrics
│   ├── embedding_classifier.py #   15+ sklearn classifier training
│   ├── ensemble_discovery.py   #   RRF / exhaustive ensemble ablation
│   ├── knn_baseline.py         #   kNN regression & similarity baselines
│   ├── generate_final_report.py#   Analysis figures & markdown report
│   └── ...                     #   (comparison, reinference, UMAP, etc.)
│
├── data_preparation/           # Embedding extraction & data splitting
│   ├── analyze_embeddings.py
│   ├── embedding_split.py      #   Strategy D farthest-point splits
│   └── extract_unlabeled_embeddings.py
│
├── experiments/                # NN experiment configs (edit run.py per experiment)
│   ├── exp364_fulltune/        #   seed=42,  primary model
│   ├── exp370_seed2/           #   seed=123, ensemble variant
│   └── exp371_seed3/           #   seed=456, ensemble variant
│
├── discovery/                  # Inference & nomination on unlabeled MOFs
│   ├── run_inference_from_cwd.py
│   ├── discovery_pipeline.py
│   ├── ensemble_predictions.py
│   ├── ensemble_report.py
│   ├── plot_model_comparison.py
│   └── nominate_diverse_dft.py #   Step 7: diversity-aware DFT nomination
│
├── figures/                    # Paper figure generation scripts
│   ├── forward_pretrained_embeddings.py   # F1: pretrained PMTransformer → embeddings
│   ├── umap_pretrained.py                 # F2: UMAP of pretrained embeddings
│   ├── forward_finetuned_umap.py          # F3: fine-tuned forward pass + UMAP
│   ├── soap_descriptors_umap.py           # F4: SOAP from CIF files + UMAP
│   ├── soap_validation.py                 # F5: SOAP structural validation
│   ├── umap_ensemble_nominations.py       # F6: ensemble nominations on UMAP
│   └── umap_dft_nominations.py            # F7: nominated structures + DFT bandgap
│
├── scripts/                    # SLURM pipeline orchestration
│   ├── config.sh               #   Centralised cluster configuration
│   ├── 01_extract_embeddings.sh
│   ├── 02_train_nn.sh
│   ├── 03_train_ml.sh
│   ├── 04_run_ensemble.sh
│   ├── 05_generate_report.sh
│   ├── 06_run_discovery.sh
│   ├── 07_nominate_candidates.sh
│   ├── figures/                #   SLURM wrappers for figure generation (F1-F7)
│   └── optional/               #   UMAP, verify ML, reinfer, screening, etc.
│
├── tools/                      # Split modification utilities
│
└── data/                       # Data directory (not tracked in Git)
    └── README.md               #   Dataset format documentation
```

---

## Getting Started

### Prerequisites

| Requirement | Details |
|-------------|---------|
| SLURM cluster | GPU nodes with CUDA 12.x (Steps 0-2, 6, F1, F3). Steps 3-5, 7 are CPU-only. |
| Python 3.9+ | Tested with Python 3.9.5 |
| MOFTransformer | `pip install moftransformer` ([docs](https://github.com/hspark1212/MOFTransformer)) -- needed for Step 0 (preprocessing) and Steps 1-2, 6 (forward passes). |
| MOF structure files | Raw CIF files; preprocessed into MOFTransformer format in Step 0. See [data/README.md](data/README.md). |
| `qmof.csv` *(optional)* | QMOF Database metadata for metal center analysis in figures F2/F3. Download from the [QMOF Database](https://github.com/Andrew-S-Rosen/QMOF) and place at `data/qmof.csv`. |

### Installation

```bash
git clone https://github.com/EYErbil/MOF-Bandgap-Discovery.git
cd MOF-Bandgap-Discovery
python -m venv venv && source venv/bin/activate

# 1. Install PyTorch + PyTorch Geometric for your CUDA version first
#    (see https://pytorch.org/get-started and https://pytorch-geometric.readthedocs.io)

# 2. Install MOFTransformer (depends on PyTorch/PyG)
pip install moftransformer

# 3. Install remaining dependencies
pip install -r requirements.txt
```

### Configuration

Edit `scripts/config.sh` -- the **only file** with cluster-specific paths:

```bash
export BASE_DIR="/path/to/MOF-Bandgap-Discovery"
export VENV_PATH="/path/to/venv/bin/activate"
export SLURM_PARTITION_GPU="ai"
export MODULE_LOADS="cuda/12.3 cudnn/8.9.5 python/3.9.5"
```

Each SLURM script also has `#SBATCH` headers for partition/account/QoS hardcoded at the top of the file (e.g., `#SBATCH --account=ai`). Changing `config.sh` alone does **not** update these headers. If your cluster uses different partition or account names, edit both `config.sh` **and** the `#SBATCH` lines at the top of each script you plan to run.

---

## Running the Pipeline

Submit each step after the previous one completes. Check job status with `squeue -u $USER`.

### Step 0: Preprocess CIF Files (one-time, prerequisite)

Before anything else, convert your raw CIF files into MOFTransformer's input format. This produces three files per MOF: `.grid` (energy grid), `.griddata16` (voxelised grid), and `.graphdata` (atom graph). Follow the [MOFTransformer preprocessing guide](https://github.com/hspark1212/MOFTransformer) -- the `prepare_data` utility handles this.

Place all preprocessed files under `data/raw/` (see [data/README.md](data/README.md) for the expected layout).

### Step 1: Extract Embeddings and Create Splits

```bash
sbatch scripts/01_extract_embeddings.sh    # GPU, ~2-4h
```

Runs a forward pass of the **pretrained** (non-fine-tuned) PMTransformer on every labeled MOF and saves the 768-dim CLS embeddings to `data/embeddings/embeddings_pretrained.npz`. These embeddings serve two purposes: (1) input features for the ML classifiers in Step 3, and (2) the basis for **Strategy D** farthest-point train/val/test splitting, which ensures every positive in val/test has a structurally similar positive in training. The NN in Step 2 does **not** use these embeddings -- it reads the raw preprocessed MOF files directly.

### Step 2: Train Neural Network Regressors

```bash
sbatch scripts/02_train_nn.sh              # GPU, ~24-69h
```

Fine-tunes PMTransformer for bandgap regression with three random seeds (`exp364`, `exp370`, `exp371`). Each experiment is configured via `experiments/<name>/run.py` -- edit hyperparameters there directly. Key settings: Huber loss, mean pooling, early stopping on validation Spearman rho.

### Step 3: Train ML Classifiers

```bash
sbatch scripts/03_train_ml.sh              # CPU, ~2-6h
```

Trains 15+ sklearn classifiers (Random Forest, SVM, Extra Trees, XGBoost, SMOTE variants, etc.) and kNN baselines on the 768-dim pretrained embeddings. No GPU needed.

### Step 4: Exhaustive Ensemble Ablation

```bash
sbatch scripts/04_run_ensemble.sh          # CPU, ~1-2h
```

Tests every 2/3/4-model combination across five fusion methods (RRF, rank averaging, voting, score averaging, weighted RRF). Reports the optimal combination maximising recall@50 on the labeled test set.

### Step 5: Generate Report

```bash
sbatch scripts/05_generate_report.sh       # CPU, ~15min
```

Produces 15+ figures: recall heatmaps, complementarity analysis, confusion matrices, bandgap distributions, and a markdown summary in `data/final_results/`.

### Step 6: Discovery on Unlabeled MOFs

```bash
sbatch scripts/06_run_discovery.sh         # GPU, ~4-8h
```

> **This uses a completely separate MOF set.** The ~10K unlabeled structures here were never part of the train/val/test split used in Steps 1-5.

Deploys all trained models on unlabeled MOFs. The script: (a) extracts pretrained embeddings for the unlabeled set → `unlabeled_embeddings.npz`, (b) runs ML inference using saved sklearn models, (c) runs NN inference using each fine-tuned checkpoint, and (d) fuses predictions via RRF to produce a consensus ranking. Before running, prepare the data:

1. Preprocess unlabeled CIF files into MOFTransformer format (Step 0)
2. Place them in `data/unlabeled/test/` (or use `discovery/collect_inference_structures.sh` to gather them)
3. Create `data/unlabeled/test_bandgaps_regression.json` mapping CIF IDs to placeholder bandgap values (`0.0`)

Edit `NN_EXPERIMENTS` and `ML_METHODS` at the top of the script to select which models to deploy.

### Step 7: Diversity-Aware DFT Candidate Nomination

```bash
sbatch scripts/07_nominate_candidates.sh   # CPU, ~1-2h
```

This is the final step: selecting 25 structures for DFT bandgap calculation. Rather than taking the top 25 by score, the pipeline ensures structural diversity:

1. **RRF shortlist** -- Build a pool of the top 500 candidates from 1 NN + 1 ML model fused by RRF
2. **Cluster** -- PCA-50 + KMeans groups the pool into 20 structural clusters
3. **Diverse selection** via four strategies:
   - **A. Cluster-quota round-robin** -- best candidate per cluster, cycling until budget is filled
   - **B. Maximal Marginal Relevance (MMR)** -- iteratively picks the candidate that best balances quality and distance from already-selected nominees
   - **C. Uncertainty-weighted quota** -- like A, but ranks within clusters by a combined quality + NN-ML disagreement score
   - **D. Long-tail exploration** -- reserves 5 slots for high-disagreement structures outside the top-500 pool
4. **Combined list** -- structures nominated by the most strategies are selected first

The script runs twice: once using PMTransformer embeddings as the diversity space, once using SOAP descriptors. SOAP-based diversity is preferred because it provides a purely geometric measure of structural similarity, independent of the learned representations.

**Key parameters** (edit at the top of `scripts/07_nominate_candidates.sh`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NN_EXP` | `exp364_fulltune` | Which NN experiment to use |
| `ML_METHOD` | `extra_trees` | Which ML classifier to use |
| `POOL_SIZE` | `500` | Size of RRF shortlist pool |
| `N_CLUSTERS` | `20` | Number of KMeans clusters |
| `BUDGET` | `25` | Number of structures to nominate |
| `EXPLORATION_BUDGET` | `5` | Slots reserved for long-tail picks |

**Outputs:**

```
data/unlabeled/nomination-SOAP/
├── FINAL_TOP25_diverse.txt    # The 25 CIF IDs for DFT
├── FINAL_TOP25_diverse.csv    # With RRF ranks, cluster IDs, uncertainty metrics
├── shortlist_pool.csv         # Full shortlist with cluster assignments
├── diversity_report.md        # Methodology and comparison with old nominees
└── plots/                     # UMAP visualisations
```

---

## Customisation

### Adding a new NN experiment

```bash
cp -r experiments/exp364_fulltune experiments/exp999_my_experiment
# Edit experiments/exp999_my_experiment/run.py — change seed, LR, freeze_layers, etc.
cd experiments/exp999_my_experiment && sbatch run.sh
```

The ensemble ablation (Step 4) automatically discovers all experiments with `test_predictions.csv`.

### Key hyperparameters (in each experiment's `run.py`)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `seed` | varies | Different initialisation for ensemble diversity |
| `freeze_layers` | `0` | 0 = full finetune; 1-3 = freeze bottom layers |
| `learning_rate` | `1e-4` | Base LR for transformer backbone |
| `lr_mult` | `10.0` | Regression head trains at 10x backbone LR |
| `loss_type` | `"huber"` | Robust to outlier bandgaps (vs MSE) |
| `patience` | `15` | Early stopping patience (epochs) |

### Optional analysis scripts

```bash
sbatch scripts/optional/run_umap_analysis.sh        # UMAP embedding visualisations
sbatch scripts/optional/run_verify_ml.sh             # ML performance heatmap
sbatch scripts/optional/run_reinfer.sh               # Recompute NN predictions from checkpoints
sbatch scripts/optional/run_screening.sh             # Two-signal candidate screening (NN + kNN)
sbatch scripts/optional/run_discovery_ml_only.sh     # ML-only inference (CPU, no GPU)
sbatch scripts/optional/run_discovery_nn_only.sh     # NN-only inference (GPU)
sbatch scripts/optional/run_model_comparison.sh      # NN vs ML UMAP investigation
```

---

## Paper Figures and Analysis

The `figures/` directory generates all publication figures. These scripts fit into the main pipeline at two points:

- **F1-F5** can run after Step 5 (once all models are trained) to visualise the embedding spaces -- useful before committing to candidate nomination.
- **F6-F7** require Step 7 outputs (nomination lists) and, for F7, completed DFT calculations on the nominated structures.

Some scripts **compute embeddings** (PMTransformer forward pass or SOAP descriptors), others **plot UMAPs**, and some do both. Labeling (labeled vs. unlabeled) is determined automatically from your split JSONs -- no manual annotation needed.

### `qmof.csv` (optional, for metal center panels)

Scripts F2 and F3 produce a metal-center UMAP panel (panel c) that colors each MOF by its central metal atom. This requires `qmof.csv` from the [QMOF Database](https://github.com/Andrew-S-Rosen/QMOF) -- specifically the `name` and `info.formula` columns. Download it and place it at `data/qmof.csv` (the path set by `$QMOF_CSV` in `config.sh`). If the file is missing, the metal center panel will display "Unknown" for all MOFs. This file is too large to include in the repository.

### What each script does

| Script | What it computes | GPU? |
|--------|-----------------|------|
| `forward_pretrained_embeddings.py` | Runs **pretrained PMTransformer** on ALL MOFs → 768-dim embeddings NPZ | GPU |
| `umap_pretrained.py` | Takes embeddings from F1 → 4-panel UMAP (labeled/unlabeled, bandgap, metal center, splits) | CPU |
| `forward_finetuned_umap.py` | Runs **fine-tuned PMTransformer** on ALL MOFs → embeddings + 4-panel UMAP (incl. metal center) | GPU |
| `soap_descriptors_umap.py` | Computes **SOAP descriptors from CIF files** → 4-panel UMAP (NN-independent) | CPU |
| `soap_validation.py` | SOAP structural validation: coverage, structure-bandgap correlation, Mantel test | CPU |
| `umap_ensemble_nominations.py` | Overlays the 25 nominated structures on fine-tuned UMAPs | CPU |
| `umap_dft_nominations.py` | Shows the 25 nominated structures colored by their DFT bandgap | CPU |

### Dependency diagram

```
F1 (pretrained forward pass)  ──→ F2 (pretrained UMAP, +metal center panel if qmof.csv)
                               ──→ F5 (SOAP validation, needs F1 + CIF)
                               ──→ F7 (DFT nomination UMAP, needs F1 + bandgap_results.csv)

F3 (finetuned forward pass)   ──→ F6 (ensemble nominations on finetuned UMAPs)
                               ──→ F7 (optional finetuned overlay)

F4 (SOAP from CIF)             [independent — only needs CIF files and split JSONs]
```

### Running the figure pipeline

| Step | Command | Time |
|------|---------|------|
| **F1** | `sbatch scripts/figures/01_forward_pretrained_embeddings.sh` | GPU, ~2-4h |
| **F2** | `sbatch scripts/figures/02_umap_pretrained.sh` | CPU, ~30min |
| **F3** | `sbatch scripts/figures/03_forward_finetuned_umap.sh` | GPU, ~6-8h |
| **F4** | `sbatch scripts/figures/04_soap_descriptors_umap.sh` | CPU, ~2-4h |
| **F5** | `sbatch scripts/figures/05_soap_validation.sh` | CPU, ~1-2h |
| **F6** | `sbatch scripts/figures/06_umap_ensemble_nominations.sh` | CPU, ~1h |
| **F7** | `sbatch scripts/figures/07_umap_dft_nominations.sh` | CPU, ~30min |

**Quick start:** Run F1 first (GPU), then F2 and F4 can run in parallel (CPU). F3 can also run in parallel with F1 if GPU resources allow. F5-F7 depend on earlier outputs as shown above.

Each SLURM wrapper sources `scripts/config.sh` and uses `$CIF_DIR`, `$QMOF_CSV`, `$FIGURES_OUTPUT`, `$SPLITS_DIR`. Edit the wrapper scripts to configure experiment names, nomination file paths, and optional arguments (e.g., `--load_umap_cache` for fast re-runs after the first UMAP computation).

All generated figures go to `figures_output/` (git-ignored). Each script also saves a JSON summary with statistics alongside the plots.

---

## Data

Each MOF is represented by three files (`.grid`, `.griddata16`, `.graphdata`) in MOFTransformer format. Labels are JSON files mapping CIF IDs to bandgap values in eV; the classification threshold is **bandgap < 1.0 eV** (positive = potentially conductive). See [data/README.md](data/README.md) for format details.

| Dataset | MOFs | Purpose |
|---------|------|---------|
| Labeled (QMOF, HSE level) | ~10,000 | Training + evaluation (Steps 1-5) |
| Unlabeled | ~10,000 | Discovery screening (Steps 6-7) |

The labeled set is split via Strategy D farthest-point coverage: ~600 train (~60 positives), ~600 val (~7 positives), ~8,800 test (~9 positives). The extreme imbalance (~0.1% positive rate in test) makes this a needle-in-a-haystack retrieval problem evaluated by recall@K.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Strategy D farthest-point split** | Guarantees every val/test positive has a structurally similar training positive, yielding honest recall metrics. |
| **Pretrained embeddings for ML** | The 768-dim PMTransformer CLS token is a powerful structural fingerprint before any fine-tuning, providing an independent retrieval signal complementary to the fine-tuned regression model. |
| **Multi-seed NN training** | Same architecture, different seeds produce models that agree on easy cases but disagree on hard ones, making ensemble fusion effective. |
| **Reciprocal Rank Fusion** | Rank-based fusion handles heterogeneous score scales (regression logits vs classification probabilities) without normalisation artifacts. |
| **Huber loss + Spearman early stopping** | Huber is robust to outlier bandgaps; Spearman rho measures ranking quality, aligning training with the discovery objective. |
| **1 NN + 1 ML for nomination** | Simpler than multi-model ensembles; diversity comes from SOAP-based selection rather than model proliferation. Ensemble experiments with 3 NN + 2 ML remain available via Step 4. |
| **SOAP diversity lens** | SOAP descriptors provide a purely geometric/chemical measure of structural dissimilarity, independent of the model features. This prevents the nominees from clustering in a learned-feature artifact. |
| **Long-tail exploration** | Reserves 5 of the 25 slots for high-uncertainty structures outside the main pool -- a hedge against the ensemble's blind spots. |

---

## Citation

```bibtex
@misc{mof-bandgap-discovery,
  title  = {MOF-Bandgap-Discovery: Diversity-Aware Ensemble Learning for
            Low-Bandgap Metal--Organic Framework Screening},
  author = {Erbil, Ege Yi\u{g}it},
  year   = {2026},
  note   = {Ko\c{c} University},
  url    = {https://github.com/EYErbil/MOF-Bandgap-Discovery}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
