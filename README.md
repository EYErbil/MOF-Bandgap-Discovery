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
  STEP 1   Extract pretrained 768-dim embeddings (PMTransformer)
    |       Create Strategy D farthest-point train/val/test splits
    v
  STEP 2   Fine-tune PMTransformer for bandgap regression (3 seeds)
    v
  STEP 3   Train 15+ sklearn classifiers + kNN baselines on pretrained embeddings
    v
  STEP 4   Exhaustive ensemble ablation (all 2/3/4-model combos, optimise recall@50)
    v
  STEP 5   Generate comprehensive analysis report (15+ figures)
    v
  STEP 6   Discovery -- deploy models on new unlabeled MOFs, RRF consensus ranking
    v
  STEP 7   Diversity-aware DFT nomination (cluster + MMR + SOAP verification)
```

**Steps 1-5** validate the ensemble on ~10K MOFs with known HSE bandgaps (only ~76 positives across all splits -- a needle-in-a-haystack retrieval problem). **Steps 6-7** deploy the validated models on ~10K unlabeled MOFs and select the final 25 candidates for DFT calculation, prioritizing both prediction confidence and structural diversity.

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
├── scripts/                    # SLURM pipeline orchestration
│   ├── config.sh               #   Centralised cluster configuration
│   ├── 01_extract_embeddings.sh
│   ├── 02_train_nn.sh
│   ├── 03_train_ml.sh
│   ├── 04_run_ensemble.sh
│   ├── 05_generate_report.sh
│   ├── 06_run_discovery.sh
│   ├── 07_nominate_candidates.sh
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
| SLURM cluster | GPU nodes with CUDA 12.x (Steps 1, 2, 6). Steps 3-5 are CPU-only. |
| Python 3.9+ | Tested with Python 3.9.5 |
| MOFTransformer | `pip install moftransformer` ([docs](https://github.com/hspark1212/MOFTransformer)) |
| MOF structure files | Preprocessed into MOFTransformer format (`.grid`, `.griddata16`, `.graphdata`). See [data/README.md](data/README.md). |

### Installation

```bash
git clone https://github.com/EYErbil/MOF-Bandgap-Discovery.git
cd MOF-Bandgap-Discovery
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install moftransformer
```

### Configuration

Edit `scripts/config.sh` -- the **only file** with cluster-specific paths:

```bash
export BASE_DIR="/path/to/MOF-Bandgap-Discovery"
export VENV_PATH="/path/to/venv/bin/activate"
export SLURM_PARTITION_GPU="ai"
export MODULE_LOADS="cuda/12.3 cudnn/8.9.5 python/3.9.5"
```

Each SLURM script also has `#SBATCH` headers for partition/account/QoS. Edit those to match your cluster if the partition names differ.

---

## Running the Pipeline

Submit each step after the previous one completes. Check job status with `squeue -u $USER`.

### Step 1: Extract Embeddings and Create Splits

```bash
sbatch scripts/01_extract_embeddings.sh    # GPU, ~2-4h
```

Extracts 768-dim CLS embeddings from the pretrained PMTransformer for every MOF. Creates **Strategy D** train/val/test splits using farthest-point sampling in embedding space, ensuring every positive in val/test has a structurally similar positive in training.

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

Deploys all trained models on a new unlabeled MOF dataset (~10K structures). Extracts embeddings, runs ML and NN inference, then fuses predictions via RRF to produce a consensus ranking. Before running, prepare the data:

1. Place MOF structure files in `data/unlabeled/` (see [data/README.md](data/README.md))
2. Create `data/unlabeled/test_bandgaps_regression.json` mapping CIF IDs to placeholder bandgap values (`0.0`)

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
sbatch scripts/optional/run_screening.sh             # Structural screening with PORMAKE
sbatch scripts/optional/run_discovery_ml_only.sh     # ML-only inference (CPU, no GPU)
sbatch scripts/optional/run_discovery_nn_only.sh     # NN-only inference (GPU)
sbatch scripts/optional/run_model_comparison.sh      # NN vs ML UMAP investigation
```

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
