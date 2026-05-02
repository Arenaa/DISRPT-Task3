# DISRPT 2025

Tools for supervised multi-class discourse-relation classification over a unified 17-label inventory: paired EDU-level units go through a data pipeline (`.rels` → cleaned TSV), then encoder-based classifiers (frozen linear head, full fine-tuning, framework-conditioned, metadata-prefixed variants) or decoder-only instruction tuning (Qwen). Outputs land under `results/` by default.

## Getting started

### Requirements

- **Python 3.10+** (code uses modern typing and standard library patterns from 3.10 onward).
- **GPU strongly recommended** for XLM-R fine-tuning and Qwen runs; CPU is possible but slow. Scripts default to CUDA when `torch.cuda.is_available()`.
- **Hugging Face Hub access** for downloading pretrained checkpoints (`FacebookAI/xlm-roberta-base`, Qwen variants, etc.).
- **DISRPT `.rels` data** under `data_subset/` before running the pipeline (layout follows the repo’s expected nested corpus paths).
V
### Installation

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
```

Install **PyTorch** for your platform (pick the CUDA build if you use a GPU): [PyTorch Get Started](https://pytorch.org/get-started/locally/).

Then install Python dependencies used by the scripts:

```bash
pip install transformers scikit-learn numpy matplotlib tqdm
```

Optional:

- **`peft`** — only if you use Qwen instruction SFT with `--lora` in `scripts/finetune_qwen_instruction_sft.py`.

### Run location

Always run commands from the **repository root** so defaults like `data_subset/`, `results/processed_tsv/`, and `results/…_results/` resolve correctly.

## User interface

This repository is **command-line only**. There is no web app, notebook server, or graphical UI bundled here; plots are written to PNG files under `results/`.

Every runnable script uses Python’s **`argparse`** CLI. To see flags and defaults:

```bash
python scripts/<script_name>.py --help
```

Training and evaluation print progress and metrics to **stdout**; metrics and artifacts are saved as **JSON**, **TSV**, and **PNG** files under `results/` (override paths with each script’s CLI).

## Typical workflow

1. **Prepare data**: place corpus `.rels` files under `data_subset/`.
2. **Build processed TSVs**: `python scripts/data_pipeline.py` (optional: `--input-dir`, `--output-dir`).
3. **Verify labels**: `python scripts/label_verification.py`.
4. **EDA plots** (optional): `python scripts/class_distribution_plots.py`.
5. **Train** one or more models (see table below); each writes to its `--output-dir`.
6. **Evaluate or compare**: `evaluate_per_dataset.py`, `evaluate_with_official_script.py`, `export_model_comparison_tables.py`, visualization scripts, or `compare_all_models.py` to regenerate comparison bundles.

## Script reference

| Script | Role |
| --- | --- |
| `data_pipeline.py` | Convert DISRPT `.rels` → cleaned TSV (`by_split/`, `by_source_file/`). |
| `label_verification.py` | Train/dev/test label coverage and consistency reports. |
| `class_distribution_plots.py` | Label distribution and heatmap figures. |
| `benchmark_frozen_xlm_roberta_linear.py` | Frozen XLM-R encoder + trainable linear head on pooled TSV splits. |
| `finetune_xlm_roberta.py` | Full fine-tuning of sequence classification head + encoder (shared CLI defaults for data paths). |
| `finetune_xlm_roberta_framework_conditioned.py` | Framework-conditioned encoder variant. |
| `finetune_xlm_roberta_tsv_features.py` | Encoder inputs prefixed with lightweight `dir` / `rel_type` features from TSV rows. |
| `finetune_qwen_instruction_sft.py` | Qwen causal LM instruction SFT (`--qwen3`, `--qwen3-4b`, optional `--lora`). |
| `qwen_prompt_baseline.py` | Same instruction-tuning objective as above, but reads train/dev/test only from `--input-dir` (typically `results/processed_tsv/by_source_file/`); use when you want the legacy entrypoint without pooled TSV flags. |
| `evaluate_per_dataset.py` | Load saved checkpoints; per-corpus / framework / language / label breakdown (no training). |
| `evaluate_with_official_script.py` | Bridge to bundled official DISRPT eval scripts + mapped `.rels` outputs. |
| `analyze_qualitative_angles.py` | Qualitative analysis from `test_gold_vs_pred.tsv`. |
| `export_model_comparison_tables.py` | Aggregate metrics into comparison CSVs under `results/comparison/`. |
| `visualize_per_dataset.py`, `visualize_f1_language_framework.py`, `visualize_xlmr_frozen_vs_finetune.py` | Figures from saved metrics JSON. |
| `compare_all_models.py` | Runs export + visualization scripts in sequence (no CLI beyond invoking the file). |

Shared **data-location flags** for XLM-R and Qwen trainers include `--data-dir`, `--train-file`, `--dev-file`, `--test-file`, `--by-source-dir`, and `--output-dir` (see `python scripts/finetune_xlm_roberta.py --help`).

**Qwen instruction SFT example** (after processed TSVs exist):

```bash
python scripts/finetune_qwen_instruction_sft.py --qwen3-4b --max-length 1024 \
  --train-batch-size 1 --gradient-accumulation-steps 16 --epochs 1
```

## Repository layout

```text
data_subset/                 # input .rels (subset) for the data pipeline
results/
  processed_tsv/             # default output of data_pipeline.py (by_split, by_source_file)
  comparison/                  # pooled/per-corpus comparison CSVs + qualitative summary tables (corpus_*.csv, corpus_test_tables.*)
  *_results/                   # each training run (metrics, best_model/, test_predictions/, qualitative_analysis/ …)
  per_dataset_plots/, eda_plots/, per_dataset_eval/, verification/, …
scripts/                     # Python training, eval, pipeline, and analysis entrypoints
```

All Python entrypoints live under **`scripts/`**.

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
