# DISRPT 2025

This repository provides:

- a data pipeline for converting DISRPT `.rels` files into cleaned TSV files
- a label verification script for checking train coverage, label mapping consistency, and rare labels
- a plot-based class distribution EDA script for generating label distribution figures

## Data Pipeline

Run:

```bash
python data_pipeline.py
```

Input:

```text
data_subset/
```

Output:

```text
processed_tsv/
```

Generated files:

- `by_source_file/`: one cleaned TSV for each original `.rels` file
- `by_split/`: one merged TSV for each split (`train`, `dev`, `test`)
- `pipeline_summary.json`: overall cleaning statistics

Cleaning:

1. Keep only `unit1_txt`, `unit2_txt`, `dir`, `rel_type`, `orig_label`, and `label`.
2. Trim leading and trailing whitespace.
3. Collapse repeated whitespace into a single space.
4. Normalize `contingency.cause` to `causal` during cleaning to keep a strict 17-label setup. This is done because `orig_label = contingency.cause.result` maps to `causal` almost everywhere in the data, while `contingency.cause` appears as a single inconsistent outlier.
5. Skip rows with empty target fields, if any appear.
6. Drop exact duplicates after cleaning.

## Label Verification

Run:

```bash
python label_verification.py
```

Output:

```text
verification/
```

Important files:

- `label_quality_summary.json`: overall verification summary

## Class Distribution EDA Plots

Run:

```bash
python class_distribution_plots.py
```

Output:

```text
eda_plots/
```

Generated figures:

- `01_label_distribution.png`: Label Distribution for the 17 normalized labels
- `02_label_by_framework_heatmap.png`: Label x Framework heatmap
- `03_label_by_language_heatmap.png`: Label x Language heatmap
- `04_train_dev_test_grouped_bar.png`: grouped bar chart for train / dev / test label distributions
- `05_rare_label_long_tail.png`: long-tail view of rare labels
- `plot_generation_summary.json`: metadata for plot generation, including the label normalization used for the strict 17-label setup
