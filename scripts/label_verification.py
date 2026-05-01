from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from data_pipeline import iter_rels_files, load_and_clean_rels_with_metadata


def load_clean_rows_with_metadata(input_dir: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    rows: list[dict[str, str]] = []
    totals = {
        "num_files": 0,
        "num_rows_before_cleaning": 0,
        "num_rows_after_cleaning": 0,
        "num_duplicate_rows_removed": 0,
    }

    for rels_file in iter_rels_files(input_dir):
        cleaned_rows, file_summary = load_and_clean_rels_with_metadata(rels_file, include_metadata=True)
        totals["num_files"] += 1
        totals["num_rows_before_cleaning"] += file_summary.num_rows_before_cleaning
        totals["num_rows_after_cleaning"] += file_summary.num_rows_after_cleaning
        totals["num_duplicate_rows_removed"] += file_summary.num_duplicate_rows_removed
        rows.extend(cleaned_rows)

    return rows, totals


def write_tsv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in columns} for row in rows)


def write_tsv_if_nonempty(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    if not rows:
        if path.exists():
            path.unlink()
        return
    write_tsv(path, rows, columns)


def verify_labels(rows: list[dict[str, str]]) -> dict[str, object]:
    global_train_labels = {row["label"] for row in rows if row["split"] == "train"}
    label_counts: Counter[str] = Counter()
    rows_by_label: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    orig_to_labels: defaultdict[str, set[str]] = defaultdict(set)
    orig_label_examples: defaultdict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        label_counts[row["label"]] += 1
        orig_to_labels[row["orig_label"]].add(row["label"])
        if len(rows_by_label[row["label"]]) < 5:
            rows_by_label[row["label"]].append(row)
        if len(orig_label_examples[row["orig_label"]]) < 5:
            orig_label_examples[row["orig_label"]].append(row)

    unseen_global_rows: list[dict[str, str]] = []
    for row in rows:
        if row["split"] not in {"dev", "test"}:
            continue
        if row["label"] not in global_train_labels:
            unseen_global_rows.append(row)

    orig_label_conflicts: list[dict[str, str]] = []
    for orig_label, labels in sorted(orig_to_labels.items()):
        if len(labels) <= 1:
            continue
        for example in orig_label_examples[orig_label]:
            orig_label_conflicts.append(
                {
                    "orig_label": orig_label,
                    "label_options": " | ".join(sorted(labels)),
                    "dataset": example["dataset"],
                    "split": example["split"],
                    "file": example["file"],
                    "line_no": example["line_no"],
                    "label": example["label"],
                    "unit1_txt": example["unit1_txt"],
                    "unit2_txt": example["unit2_txt"],
                }
            )

    singleton_rows: list[dict[str, str]] = []
    for label, count in sorted(label_counts.items()):
        if count != 1:
            continue
        example = rows_by_label[label][0]
        singleton_rows.append(
            {
                "label": label,
                "count": str(count),
                "dataset": example["dataset"],
                "split": example["split"],
                "file": example["file"],
                "line_no": example["line_no"],
                "orig_label": example["orig_label"],
                "unit1_txt": example["unit1_txt"],
                "unit2_txt": example["unit2_txt"],
            }
        )

    split_label_vocab = {
        split: sorted({row["label"] for row in rows if row["split"] == split})
        for split in ("train", "dev", "test", "unknown")
        if any(row["split"] == split for row in rows)
    }
    return {
        "label_counts": dict(sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))),
        "split_label_vocab": split_label_vocab,
        "global_train_labels": sorted(global_train_labels),
        "unseen_global_rows": unseen_global_rows,
        "orig_label_conflicts": orig_label_conflicts,
        "singleton_rows": singleton_rows,
    }


def build_verification_report(input_dir: Path, output_dir: Path) -> dict[str, object]:
    rows, totals = load_clean_rows_with_metadata(input_dir)
    verification = verify_labels(rows)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "totals": totals,
        "num_unique_labels": len(verification["label_counts"]),
        "train_label_set": verification["global_train_labels"],
        "issues": {
            "num_unseen_labels_from_train": len(verification["unseen_global_rows"]),
            "num_orig_label_conflicts": len({row["orig_label"] for row in verification["orig_label_conflicts"]}),
            "num_singleton_labels": len(verification["singleton_rows"]),
        },
        "labels_by_split": verification["split_label_vocab"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "label_quality_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    unseen_columns = ["dataset", "split", "file", "line_no", "orig_label", "label", "unit1_txt", "unit2_txt"]
    conflict_columns = ["orig_label", "label_options", "dataset", "split", "file", "line_no", "label", "unit1_txt", "unit2_txt"]
    singleton_columns = ["label", "count", "dataset", "split", "file", "line_no", "orig_label", "unit1_txt", "unit2_txt"]

    write_tsv_if_nonempty(output_dir / "labels_missing_from_train.tsv", verification["unseen_global_rows"], unseen_columns)
    write_tsv_if_nonempty(output_dir / "original_to_final_label_conflicts.tsv", verification["orig_label_conflicts"], conflict_columns)
    write_tsv_if_nonempty(output_dir / "labels_seen_once.tsv", verification["singleton_rows"], singleton_columns)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify label quality and consistency in DISRPT .rels files.")
    parser.add_argument("--input-dir", default="data_subset", help="Directory containing DISRPT .rels files.")
    parser.add_argument("--output-dir", default="results/verification", help="Directory for verification reports.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_verification_report(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
