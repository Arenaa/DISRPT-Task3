"""Fine-tune XLM-RoBERTa on DISRPT with `dir` and `rel_type` prepended to unit1 (not `orig_label`).

Same training loop as `finetune_xlm_roberta.py`, but the first sequence to the encoder is
`[metadata line] + unit1_txt` so the model sees the extra TSV fields. Plain fine-tuning
(text-only) remains `finetune_xlm_roberta.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from disrpt_tsv_features import disrpt_feature_prefix_from_tsv_row
from finetune_xlm_roberta import (
    SplitData,
    build_arg_parser,
    limit_examples,
    read_tsv,
    run_finetune,
)


def load_split_tsv_features(split_path: Path, max_examples: int | None) -> SplitData:
    rows = read_tsv(split_path)
    rows = limit_examples(rows, max_examples)
    unit1: List[str] = []
    unit2: List[str] = []
    labels: List[str] = []
    for r in rows:
        lab = r.get("label", "")
        u1 = r.get("unit1_txt", "")
        u2 = r.get("unit2_txt", "")
        if not lab or not u1 or not u2:
            continue
        fp = disrpt_feature_prefix_from_tsv_row(r)
        if fp:
            u1 = f"{fp}{u1}"
        unit1.append(u1)
        unit2.append(u2)
        labels.append(lab)
    return SplitData(unit1=unit1, unit2=unit2, labels=labels)


def parse_args() -> argparse.Namespace:
    p = build_arg_parser()
    p.set_defaults(output_dir="results/xlmr_finetune_tsv_features_results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_finetune(
        args,
        load_split_fn=load_split_tsv_features,
        setup_extras={
            "input": "unit1 = dir, rel_type (prefix) + unit1_txt; unit2 = unit2_txt",
        },
        per_dataset_model_label="Fine-tuned XLM-RoBERTa + TSV features (dir, rel_type)",
    )


if __name__ == "__main__":
    main()
