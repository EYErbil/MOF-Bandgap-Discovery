# Data Directory

This directory should contain the MOF datasets used by the pipeline.
The data files are **not included** in this repository due to their size.

## Expected Structure

```
data/
├── raw/                          # Full labeled dataset (train+val+test before splitting)
│   ├── train/
│   │   ├── <CIF_ID>.grid
│   │   ├── <CIF_ID>.griddata16
│   │   └── <CIF_ID>.graphdata
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
├── splits/
│   └── strategy_d_farthest_point/   # Smart split (farthest-point coverage)
│       ├── train/  → symlinks or copies of raw files
│       ├── val/
│       └── test/
├── phase6/                       # Unlabeled MOFs for discovery inference
│   ├── <CIF_ID>.grid
│   ├── <CIF_ID>.griddata16
│   └── <CIF_ID>.graphdata
└── labels/
    ├── train.json                # { "CIF_ID": bandgap_eV, ... }
    ├── val.json
    └── test.json
```

## File Formats

Each MOF structure is represented by **three files** sharing the same CIF ID stem:

| File              | Description                                           |
|-------------------|-------------------------------------------------------|
| `<CIF_ID>.grid`        | 3-D energy grid metadata (dimensions, spacing)   |
| `<CIF_ID>.griddata16`  | Flattened energy grid values (float16)            |
| `<CIF_ID>.graphdata`   | Graph representation (atoms, bonds, unit cell)    |

These are the input formats expected by **MOFTransformer**.

## Label Files

Label JSON files map CIF IDs to bandgap values in eV:

```json
{
  "QMOF-a1b2c3": 0.42,
  "QMOF-d4e5f6": 2.15,
  ...
}
```

**Classification threshold:** bandgap < 1.0 eV → "positive" (potentially conductive).

## Obtaining the Data

The raw data originates from the **QMOF Database**.
To reproduce the dataset:

1. Download structures from the [QMOF Database](https://github.com/Andrew-S-Rosen/QMOF).
2. Convert CIF files to the MOFTransformer format using the `moftransformer` preprocessing tools.
3. Run the split scripts in `data_preparation/` to create the strategy-D farthest-point splits.
