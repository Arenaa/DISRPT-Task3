# DISRPT 2025

This repository provides:

- a `results/` directory as the default root for pipeline outputs, model checkpoints, metrics, and plots (overridable via each script’s CLI)
- a data pipeline for converting DISRPT `.rels` files into cleaned TSV files
- a label verification script for checking train coverage, label mapping consistency, and rare labels
- a plot-based class distribution EDA script for generating label distribution figures

## Repository layout

```text
data_subset/                 # input .rels (subset) for the data pipeline
results/
  processed_tsv/             # default output of data_pipeline.py (by_split, by_source_file)
  comparison/                  # pooled/per-corpus comparison CSVs + qualitative summary tables (corpus_*.csv, corpus_test_tables.*)
  *_results/                   # each training run (metrics, best_model/, test_predictions/, qualitative_analysis/ …)
  per_dataset_plots/, eda_plots/, per_dataset_eval/, verification/, …
scripts/                     # Python training, eval, pipeline, and shell launch helpers
```

All Python entrypoints live under **`scripts/`**. Run them from the **repository root** so paths like `data_subset/` and `results/` resolve correctly.

Run Qwen3-4B instruction SFT with:

```bash
./scripts/run_finetune_qwen3_4b.sh
```

## Data Pipeline

Run:

```bash
python scripts/data_pipeline.py
```

Input:

```text
data_subset/
```

Output (default):

```text
results/processed_tsv/
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
6. Remove duplicate rows within each source file after cleaning, based on the retained columns.

## Label Verification

Run:

```bash
python scripts/label_verification.py
```

Output (default):

```text
results/verification/
```

Important files:

- `label_quality_summary.json`: overall verification summary

## Class Distribution EDA Plots

Run:

```bash
python scripts/class_distribution_plots.py
```

Output (default):

```text
results/eda_plots/
```

Generated figures:

- `01_label_distribution.png`: Label Distribution for the 17 normalized labels
- `02_label_by_framework_heatmap.png`: Label x Framework heatmap
- `03_label_by_language_heatmap.png`: Label x Language heatmap
- `04_train_dev_test_grouped_bar.png`: grouped bar chart for train / dev / test label distributions
- `05_rare_label_long_tail.png`: long-tail view of rare labels
- `plot_generation_summary.json`: metadata for plot generation, including the label normalization used for the strict 17-label setup
