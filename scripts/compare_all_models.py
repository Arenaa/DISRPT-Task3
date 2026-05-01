"""Regenerate the 4-way comparison: CSVs, per-dataset plots, and pooled metrics PNGs.

Runs (in order) from repo root (``cwd``), with scripts under ``scripts/``:
  1. scripts/export_model_comparison_tables.py  -> results/comparison/
  2. scripts/visualize_per_dataset.py           -> results/per_dataset_plots/
  3. scripts/visualize_xlmr_frozen_vs_finetune.py -> results/xlmr_comparison_plots/

Per-dataset, all four models: `results/comparison/per_corpus_four_models_test.csv`. TSV rows
use `finetune_tsv_test.json` or, if that is absent, `per_dataset_test_metrics.json` when
`model` indicates the TSV run (see `tsv_feature_eval_path` in `export_model_comparison_tables.py`).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent

_PIPELINE_SCRIPTS = [
    "export_model_comparison_tables.py",
    "visualize_per_dataset.py",
    "visualize_xlmr_frozen_vs_finetune.py",
]


def main() -> None:
    for name in _PIPELINE_SCRIPTS:
        path = _SCRIPTS_DIR / name
        print(f"\n=== {name} ===\n")
        r = subprocess.run([sys.executable, str(path)], cwd=_REPO_ROOT, check=False)
        if r.returncode != 0:
            print(f"Exit {r.returncode} from {name}", file=sys.stderr)
            sys.exit(r.returncode)
    print("\nDone. See results/comparison/, results/per_dataset_plots/, results/xlmr_comparison_plots/")


if __name__ == "__main__":
    main()
