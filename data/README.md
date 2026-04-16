# Data Directory

This directory should contain the MOF datasets used by the pipeline.
The data files are **not included** in this repository due to their size.

## Expected Structure

After Step 0 (preprocessing) and Step 1 (embedding extraction + splitting), the directory will look like this:

```
data/
├── raw/                                # All labeled MOF structures (pre-split)
│   └── test/                           # MOFTransformer expects a parent/split layout
│       ├── <CIF_ID>.grid               #   — all ~10K labeled structures go here
│       ├── <CIF_ID>.griddata16
│       └── <CIF_ID>.graphdata
│
├── splits/                             # Created by Step 1 (embedding_split.py)
│   └── strategy_d_farthest_point/
│       ├── train/                      #   symlinks to raw/ files
│       ├── val/
│       ├── test/
│       ├── train_bandgaps_regression.json
│       ├── val_bandgaps_regression.json
│       └── test_bandgaps_regression.json
│
├── embeddings/                         # Created by Step 1
│   └── embeddings_pretrained.npz       #   768-dim pretrained CLS embeddings
│
├── embedding_classifiers/              # Created by Step 3
│   └── strategy_d_farthest_point/      #   saved sklearn models
│
├── ensemble_results/                   # Created by Step 4
├── final_results/                      # Created by Step 5
│
├── unlabeled/                          # Separate unlabeled MOFs for Steps 6-7
│   ├── test/
│   │   ├── <CIF_ID>.grid
│   │   ├── <CIF_ID>.griddata16
│   │   └── <CIF_ID>.graphdata
│   ├── test_bandgaps_regression.json   #   CIF IDs → 0.0 (placeholder)
│   └── embedding_analysis/             # Created by Step 6
│       └── unlabeled_embeddings.npz
│
├── raw/cif/                            # (optional) Original CIF files for SOAP computation
│   └── <CIF_ID>.cif                    #   needed by figures/soap_descriptors_umap.py
│
└── qmof.csv                           # (optional) QMOF Database metadata
                                        #   needed for metal center UMAP panels (F2, F3)
```

## File Formats

Each MOF structure is represented by **three files** sharing the same CIF ID stem:

| File              | Description                                           |
|-------------------|-------------------------------------------------------|
| `<CIF_ID>.grid`        | 3-D energy grid metadata (dimensions, spacing)   |
| `<CIF_ID>.griddata16`  | Flattened energy grid values (float16)            |
| `<CIF_ID>.graphdata`   | Graph representation (atoms, bonds, unit cell)    |

These are the input formats expected by **MOFTransformer**. Generate them from raw CIF files using MOFTransformer's `prepare_data` utility (see Step 0 in the main [README](../README.md)).

## Label Files

Label JSON files map CIF IDs to bandgap values in eV:

```json
{
  "QMOF-a1b2c3": 0.42,
  "QMOF-d4e5f6": 2.15
}
```

**Classification threshold:** bandgap < 1.0 eV → "positive" (potentially conductive).

For the **unlabeled** set, create `test_bandgaps_regression.json` with placeholder values:

```json
{
  "QMOF-x1y2z3": 0.0,
  "QMOF-a9b8c7": 0.0
}
```

The pipeline only uses the keys (CIF IDs) for inference — the values are ignored.

## `qmof.csv` (optional)

This file contains QMOF Database metadata, particularly the `name` and `info.formula` columns used to extract metal center information for UMAP panels in figures F2 and F3. Download it from the [QMOF Database](https://github.com/Andrew-S-Rosen/QMOF). If missing, metal center panels will display "Unknown" for all MOFs.

## Obtaining the Data

The raw data originates from the **QMOF Database**:

1. Download structures from the [QMOF Database](https://github.com/Andrew-S-Rosen/QMOF).
2. Convert CIF files to MOFTransformer format using `moftransformer.utils.prepare_data` (Step 0).
3. Place preprocessed files in `data/raw/test/`.
4. Run Step 1 (`scripts/01_extract_embeddings.sh`) to create embeddings and Strategy D splits.
5. For discovery, prepare a separate unlabeled set in `data/unlabeled/test/`.
