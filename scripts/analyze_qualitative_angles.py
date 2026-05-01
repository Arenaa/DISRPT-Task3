"""Comprehensive qualitative analysis for discourse-relation predictions.

This script is designed to fully cover three common qualitative-analysis angles:

1. Error cases
   - Find the most frequent confusion pairs overall and within each language,
     framework, and dataset/corpus.
   - Attach original `unit1_txt` / `unit2_txt` examples to each confusion.

2. Cross-framework inconsistencies
   - For each unified label, compare precision / recall / F1 across frameworks.
   - Highlight labels whose performance varies the most across frameworks.
   - Report the dominant framework-specific confusion patterns for each label.

3. Rare labels
   - Automatically identify low-frequency labels using a support threshold.
   - Report overall rare labels across the whole test set.
   - Also report framework-internal rare labels, so a label can be treated as
     rare inside a specific framework even if it is not globally rare.
   - Show which broader labels they are most often confused with.

The script also reports slice-level metrics and top confusions for:
   - overall
   - language
   - framework
   - dataset / corpus

Expected input
--------------
You should point `--predictions` to a file such as:

    new_results/<model_name>/test_predictions/test_gold_vs_pred.tsv

That file must contain at least these columns:
    - corpus
    - row_in_corpus
    - gold_label
    - pred_label

The script joins these rows back to the cleaned per-corpus test TSV files under
`results/processed_tsv/by_source_file` so it can recover:
    - dir
    - rel_type
    - orig_label
    - unit1_txt
    - unit2_txt

Main outputs
------------
The script writes the following files to `--output-dir`:

    qualitative_analysis.json
        Full structured output for all analyses.

    qualitative_analysis.md
        Readable markdown report with the key findings.

    slice_metrics.tsv
        Overall / language / framework / dataset metrics.

    slice_top_confusions.tsv
        Top confusion pairs for each slice.

    all_errors.tsv
        One row per misclassified example, with original text attached.

    framework_label_variation.tsv
        Per-label performance by framework, plus cross-framework F1 spread.

    rare_label_summary.tsv
        Overall summary of labels whose support is at or below the rare-label
        threshold across the full test set.

    rare_label_confusions.tsv
        For each rare label, the most common wrong predicted labels.

    framework_rare_label_summary.tsv
        Summary of labels whose support is at or below the rare-label threshold
        within a specific framework.

    framework_rare_label_confusions.tsv
        Top wrong predicted labels for framework-internal rare labels.

Typical usage
-------------
Analyze XLM-R fine-tune + TSV features:

    python3 analyze_qualitative_angles.py \
      --predictions new_results/xlmr_finetune_tsv_features_results/test_predictions/test_gold_vs_pred.tsv \
      --output-dir new_results/xlmr_finetune_tsv_features_results/test_predictions/qualitative_analysis \
      --top-k 10 \
      --max-examples-per-pair 3 \
      --rare-threshold 50 \
      --min-support-per-framework 5

Analyze plain XLM-R fine-tuning:

    python3 analyze_qualitative_angles.py \
      --predictions new_results/xlmr_finetune_results/test_predictions/test_gold_vs_pred.tsv \
      --output-dir new_results/xlmr_finetune_results/test_predictions/qualitative_analysis

Interpretation tips
-------------------
* Error cases:
    Read `slice_top_confusions.tsv` and inspect examples in
    `qualitative_analysis.json` or `all_errors.tsv`.

* Cross-framework inconsistencies:
    Start with `framework_label_variation.tsv`, sort by `f1_range`, and look at
    labels whose F1 changes sharply across frameworks.

* Rare labels:
    Start with `rare_label_summary.tsv` and `rare_label_confusions.tsv` for
    global rarity, then inspect `framework_rare_label_summary.tsv` and
    `framework_rare_label_confusions.tsv` for framework-internal rarity.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import pstdev

from sklearn.metrics import accuracy_score, classification_report, f1_score


CORPUS_FRAMEWORK_OVERRIDES = {
    "eng.erst.gum": "rst",
}


@dataclass
class PredictionRow:
    corpus: str
    row_in_corpus: int
    framework: str
    language: str
    gold_label: str
    pred_label: str
    dir: str
    rel_type: str
    orig_label: str
    unit1_txt: str
    unit2_txt: str


def corpus_framework(corpus: str) -> str:
    if corpus in CORPUS_FRAMEWORK_OVERRIDES:
        return CORPUS_FRAMEWORK_OVERRIDES[corpus]
    parts = corpus.split(".")
    return parts[1] if len(parts) >= 2 else "unknown"


def corpus_language(corpus: str) -> str:
    parts = corpus.split(".")
    return parts[0] if parts else "unknown"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t", restval=""))


def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_prediction_rows(path: Path) -> list[dict[str, str]]:
    rows = read_tsv(path)
    required = {"corpus", "row_in_corpus", "gold_label", "pred_label"}
    missing = required - set(rows[0].keys()) if rows else required
    if missing:
        raise ValueError(f"Missing required prediction columns in {path}: {sorted(missing)}")
    return rows


def build_source_cache(by_source_dir: Path) -> dict[str, list[dict[str, str]]]:
    cache: dict[str, list[dict[str, str]]] = {}
    for corpus_dir in sorted(path for path in by_source_dir.iterdir() if path.is_dir()):
        corpus = corpus_dir.name
        test_path = corpus_dir / f"{corpus}_test.tsv"
        if test_path.exists():
            cache[corpus] = read_tsv(test_path)
    return cache


def attach_source_rows(
    prediction_rows: list[dict[str, str]],
    source_cache: dict[str, list[dict[str, str]]],
) -> list[PredictionRow]:
    rows: list[PredictionRow] = []
    for row in prediction_rows:
        corpus = row["corpus"]
        row_idx = int(row["row_in_corpus"])
        source_rows = source_cache.get(corpus)
        if source_rows is None:
            raise FileNotFoundError(f"No source TSV found for corpus: {corpus}")
        if row_idx < 0 or row_idx >= len(source_rows):
            raise IndexError(f"row_in_corpus={row_idx} is out of range for {corpus}")
        source = source_rows[row_idx]
        rows.append(
            PredictionRow(
                corpus=corpus,
                row_in_corpus=row_idx,
                framework=corpus_framework(corpus),
                language=corpus_language(corpus),
                gold_label=row["gold_label"],
                pred_label=row["pred_label"],
                dir=source.get("dir", ""),
                rel_type=source.get("rel_type", ""),
                orig_label=source.get("orig_label", ""),
                unit1_txt=source.get("unit1_txt", ""),
                unit2_txt=source.get("unit2_txt", ""),
            )
        )
    return rows


def per_label_metrics(rows: list[PredictionRow]) -> dict[str, dict[str, float | int]]:
    y_true = [row.gold_label for row in rows]
    y_pred = [row.pred_label for row in rows]
    labels = sorted(set(y_true) | set(y_pred))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        label: {
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1": float(report[label]["f1-score"]),
            "support": int(report[label]["support"]),
        }
        for label in labels
    }


def compute_slice_metrics(rows: list[PredictionRow]) -> dict[str, object]:
    y_true = [row.gold_label for row in rows]
    y_pred = [row.pred_label for row in rows]
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "support": len(rows),
        "num_gold_labels": len(set(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "num_errors": sum(row.gold_label != row.pred_label for row in rows),
        "per_label": per_label_metrics(rows),
    }


def confusion_counter(rows: list[PredictionRow]) -> Counter[tuple[str, str]]:
    return Counter(
        (row.gold_label, row.pred_label)
        for row in rows
        if row.gold_label != row.pred_label
    )


def collect_examples(
    rows: list[PredictionRow],
    *,
    max_examples_per_pair: int,
) -> dict[tuple[str, str], list[PredictionRow]]:
    out: dict[tuple[str, str], list[PredictionRow]] = defaultdict(list)
    for row in rows:
        if row.gold_label == row.pred_label:
            continue
        key = (row.gold_label, row.pred_label)
        if len(out[key]) < max_examples_per_pair:
            out[key].append(row)
    return dict(out)


def summarize_slice(
    slice_name: str,
    rows: list[PredictionRow],
    *,
    top_k: int,
    max_examples_per_pair: int,
) -> dict[str, object]:
    metrics = compute_slice_metrics(rows)
    pair_counts = confusion_counter(rows)
    pair_examples = collect_examples(rows, max_examples_per_pair=max_examples_per_pair)
    top_confusions = []
    for (gold, pred), count in pair_counts.most_common(top_k):
        top_confusions.append(
            {
                "gold_label": gold,
                "pred_label": pred,
                "count": count,
                "examples": [asdict(example) for example in pair_examples[(gold, pred)]],
            }
        )
    return {
        "slice": slice_name,
        **metrics,
        "top_confusions": top_confusions,
    }


def group_rows(rows: list[PredictionRow], attr: str) -> dict[str, list[PredictionRow]]:
    grouped: dict[str, list[PredictionRow]] = defaultdict(list)
    for row in rows:
        grouped[getattr(row, attr)].append(row)
    return dict(sorted(grouped.items()))


def build_slice_analysis(
    rows: list[PredictionRow],
    *,
    top_k: int,
    max_examples_per_pair: int,
) -> dict[str, object]:
    by_language = group_rows(rows, "language")
    by_framework = group_rows(rows, "framework")
    by_dataset = group_rows(rows, "corpus")
    return {
        "overall": summarize_slice(
            "overall",
            rows,
            top_k=top_k,
            max_examples_per_pair=max_examples_per_pair,
        ),
        "by_language": {
            key: summarize_slice(
                key,
                value,
                top_k=top_k,
                max_examples_per_pair=max_examples_per_pair,
            )
            for key, value in by_language.items()
        },
        "by_framework": {
            key: summarize_slice(
                key,
                value,
                top_k=top_k,
                max_examples_per_pair=max_examples_per_pair,
            )
            for key, value in by_framework.items()
        },
        "by_dataset": {
            key: summarize_slice(
                key,
                value,
                top_k=top_k,
                max_examples_per_pair=max_examples_per_pair,
            )
            for key, value in by_dataset.items()
        },
    }


def top_confusions_for_label(
    rows: list[PredictionRow],
    label: str,
    *,
    top_k: int,
    max_examples_per_pair: int,
) -> list[dict[str, object]]:
    label_rows = [row for row in rows if row.gold_label == label and row.pred_label != label]
    counter = Counter(row.pred_label for row in label_rows)
    examples_by_pred: dict[str, list[PredictionRow]] = defaultdict(list)
    for row in label_rows:
        if len(examples_by_pred[row.pred_label]) < max_examples_per_pair:
            examples_by_pred[row.pred_label].append(row)
    results = []
    for pred_label, count in counter.most_common(top_k):
        results.append(
            {
                "gold_label": label,
                "pred_label": pred_label,
                "count": count,
                "examples": [asdict(example) for example in examples_by_pred[pred_label]],
            }
        )
    return results


def analyze_framework_inconsistencies(
    rows: list[PredictionRow],
    *,
    top_k: int,
    max_examples_per_pair: int,
    min_support_per_framework: int,
) -> dict[str, object]:
    framework_groups = group_rows(rows, "framework")
    labels = sorted(set(row.gold_label for row in rows))
    label_summaries = []
    for label in labels:
        per_framework = []
        valid_f1s = []
        rows_for_label = [row for row in rows if row.gold_label == label]
        overall_support = len(rows_for_label)

        for framework, framework_rows in framework_groups.items():
            metrics = per_label_metrics(framework_rows)
            label_metrics = metrics.get(
                label,
                {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
            )
            support = int(label_metrics["support"])
            entry = {
                "framework": framework,
                "precision": float(label_metrics["precision"]),
                "recall": float(label_metrics["recall"]),
                "f1": float(label_metrics["f1"]),
                "support": support,
                "top_confusions": top_confusions_for_label(
                    framework_rows,
                    label,
                    top_k=min(top_k, 5),
                    max_examples_per_pair=max_examples_per_pair,
                ),
            }
            per_framework.append(entry)
            if support >= min_support_per_framework:
                valid_f1s.append(float(label_metrics["f1"]))

        if valid_f1s:
            f1_range = max(valid_f1s) - min(valid_f1s)
            f1_std = pstdev(valid_f1s) if len(valid_f1s) > 1 else 0.0
        else:
            f1_range = 0.0
            f1_std = 0.0

        label_summaries.append(
            {
                "label": label,
                "overall_support": overall_support,
                "framework_count_meeting_threshold": len(valid_f1s),
                "f1_range": f1_range,
                "f1_std": f1_std,
                "per_framework": sorted(per_framework, key=lambda item: item["framework"]),
            }
        )

    label_summaries.sort(
        key=lambda item: (
            item["framework_count_meeting_threshold"] > 0,
            item["f1_range"],
            item["overall_support"],
        ),
        reverse=True,
    )

    return {
        "min_support_per_framework": min_support_per_framework,
        "labels_sorted_by_f1_range": label_summaries,
        "top_inconsistent_labels": label_summaries[:top_k],
    }


def analyze_rare_labels(
    rows: list[PredictionRow],
    *,
    rare_threshold: int,
    top_k: int,
    max_examples_per_pair: int,
) -> dict[str, object]:
    overall_per_label = per_label_metrics(rows)
    framework_groups = group_rows(rows, "framework")
    rare_labels = sorted(
        label
        for label, metrics in overall_per_label.items()
        if int(metrics["support"]) <= rare_threshold
    )

    summaries = []
    for label in rare_labels:
        label_rows = [row for row in rows if row.gold_label == label]
        confusion_items = top_confusions_for_label(
            rows,
            label,
            top_k=top_k,
            max_examples_per_pair=max_examples_per_pair,
        )
        per_framework = []
        for framework, framework_rows in framework_groups.items():
            metrics = per_label_metrics(framework_rows).get(
                label,
                {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
            )
            per_framework.append(
                {
                    "framework": framework,
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "f1": float(metrics["f1"]),
                    "support": int(metrics["support"]),
                }
            )
        summaries.append(
            {
                "label": label,
                "overall_precision": float(overall_per_label[label]["precision"]),
                "overall_recall": float(overall_per_label[label]["recall"]),
                "overall_f1": float(overall_per_label[label]["f1"]),
                "overall_support": int(overall_per_label[label]["support"]),
                "top_confusions": confusion_items,
                "per_framework": sorted(per_framework, key=lambda item: item["framework"]),
                "example_gold_rows": [asdict(row) for row in label_rows[:max_examples_per_pair]],
            }
        )

    summaries.sort(key=lambda item: (item["overall_support"], item["overall_f1"]))
    return {
        "rare_threshold": rare_threshold,
        "num_rare_labels": len(summaries),
        "labels": summaries,
    }


def analyze_framework_rare_labels(
    rows: list[PredictionRow],
    *,
    rare_threshold: int,
    top_k: int,
    max_examples_per_pair: int,
) -> dict[str, object]:
    framework_groups = group_rows(rows, "framework")
    framework_summaries = []
    total_rare_pairs = 0

    for framework, framework_rows in framework_groups.items():
        framework_metrics = per_label_metrics(framework_rows)
        rare_labels = sorted(
            label
            for label, metrics in framework_metrics.items()
            if int(metrics["support"]) <= rare_threshold and int(metrics["support"]) > 0
        )
        label_summaries = []
        for label in rare_labels:
            confusion_items = top_confusions_for_label(
                framework_rows,
                label,
                top_k=top_k,
                max_examples_per_pair=max_examples_per_pair,
            )
            label_summaries.append(
                {
                    "framework": framework,
                    "label": label,
                    "precision": float(framework_metrics[label]["precision"]),
                    "recall": float(framework_metrics[label]["recall"]),
                    "f1": float(framework_metrics[label]["f1"]),
                    "support": int(framework_metrics[label]["support"]),
                    "top_confusions": confusion_items,
                }
            )
        label_summaries.sort(key=lambda item: (item["support"], item["f1"]))
        total_rare_pairs += len(label_summaries)
        framework_summaries.append(
            {
                "framework": framework,
                "num_rare_labels": len(label_summaries),
                "labels": label_summaries,
            }
        )

    framework_summaries.sort(key=lambda item: item["framework"])
    return {
        "rare_threshold": rare_threshold,
        "num_framework_rare_label_entries": total_rare_pairs,
        "frameworks": framework_summaries,
    }


def write_markdown_report(path: Path, payload: dict[str, object]) -> None:
    lines: list[str] = []
    lines.append("# Qualitative Analysis")
    lines.append("")
    lines.append(f"- Predictions: `{payload['predictions_path']}`")
    lines.append(f"- Total examples: `{payload['num_predictions']}`")
    lines.append("")

    overall = payload["slice_analysis"]["overall"]
    lines.append("## Overall")
    lines.append("")
    lines.append(
        f"- Accuracy: `{overall['accuracy']:.4f}`  "
        f"Macro-F1: `{overall['macro_f1']:.4f}`  "
        f"Weighted-F1: `{overall['weighted_f1']:.4f}`  "
        f"Errors: `{overall['num_errors']}`"
    )
    lines.append("")

    lines.append("## Error Cases")
    lines.append("")
    lines.append("### Top Overall Confusions")
    lines.append("")
    lines.append("| Gold | Pred | Count |")
    lines.append("|---|---|---:|")
    for item in overall["top_confusions"]:
        lines.append(f"| {item['gold_label']} | {item['pred_label']} | {item['count']} |")
    lines.append("")

    for section_title, section_key in [
        ("By Language", "by_language"),
        ("By Framework", "by_framework"),
        ("By Dataset", "by_dataset"),
    ]:
        lines.append(f"## {section_title}")
        lines.append("")
        section = payload["slice_analysis"][section_key]
        for group_key, summary in section.items():
            lines.append(f"### {group_key}")
            lines.append("")
            lines.append(
                f"- Support: `{summary['support']}`  "
                f"Accuracy: `{summary['accuracy']:.4f}`  "
                f"Macro-F1: `{summary['macro_f1']:.4f}`  "
                f"Errors: `{summary['num_errors']}`"
            )
            if summary["top_confusions"]:
                lines.append("")
                lines.append("| Gold | Pred | Count |")
                lines.append("|---|---|---:|")
                for item in summary["top_confusions"]:
                    lines.append(f"| {item['gold_label']} | {item['pred_label']} | {item['count']} |")
            else:
                lines.append("")
                lines.append("- No errors.")
            lines.append("")

    lines.append("## Cross-Framework Inconsistencies")
    lines.append("")
    lines.append("| Label | Overall Support | F1 Range | F1 Std | Frameworks >= Threshold |")
    lines.append("|---|---:|---:|---:|---:|")
    for item in payload["framework_inconsistencies"]["top_inconsistent_labels"]:
        lines.append(
            f"| {item['label']} | {item['overall_support']} | {item['f1_range']:.4f} | "
            f"{item['f1_std']:.4f} | {item['framework_count_meeting_threshold']} |"
        )
    lines.append("")

    lines.append("## Rare Labels")
    lines.append("")
    lines.append(
        f"- Rare threshold: labels with support <= `{payload['rare_label_analysis']['rare_threshold']}`"
    )
    lines.append("")
    lines.append("| Label | Support | Precision | Recall | F1 |")
    lines.append("|---|---:|---:|---:|---:|")
    for item in payload["rare_label_analysis"]["labels"]:
        lines.append(
            f"| {item['label']} | {item['overall_support']} | {item['overall_precision']:.4f} | "
            f"{item['overall_recall']:.4f} | {item['overall_f1']:.4f} |"
        )
    lines.append("")

    lines.append("## Framework-Internal Rare Labels")
    lines.append("")
    lines.append(
        f"- Rare threshold inside each framework: support <= `{payload['framework_rare_label_analysis']['rare_threshold']}`"
    )
    lines.append("")
    for framework_block in payload["framework_rare_label_analysis"]["frameworks"]:
        lines.append(f"### {framework_block['framework']}")
        lines.append("")
        if not framework_block["labels"]:
            lines.append("- No framework-internal rare labels under the current threshold.")
            lines.append("")
            continue
        lines.append("| Label | Support | Precision | Recall | F1 |")
        lines.append("|---|---:|---:|---:|---:|")
        for item in framework_block["labels"]:
            lines.append(
                f"| {item['label']} | {item['support']} | {item['precision']:.4f} | "
                f"{item['recall']:.4f} | {item['f1']:.4f} |"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run comprehensive qualitative analysis on a model's test_gold_vs_pred.tsv. "
            "Covers error cases, cross-framework inconsistencies, and rare-label behavior."
        )
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to a model prediction file such as `new_results/<model>/test_predictions/test_gold_vs_pred.tsv`.",
    )
    parser.add_argument(
        "--by-source-dir",
        default="results/processed_tsv/by_source_file",
        help="Directory containing per-corpus cleaned test TSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output files. Defaults to `<predictions_parent>/qualitative_analysis`.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top confusion items / labels to keep in summaries.",
    )
    parser.add_argument(
        "--max-examples-per-pair",
        type=int,
        default=3,
        help="Maximum number of attached examples for a confusion pair.",
    )
    parser.add_argument(
        "--rare-threshold",
        type=int,
        default=50,
        help=(
            "Labels with gold support <= this value are treated as rare labels. "
            "The threshold is applied both globally and within each framework."
        ),
    )
    parser.add_argument(
        "--min-support-per-framework",
        type=int,
        default=5,
        help=(
            "Minimum gold support a label must have inside a framework for that "
            "framework's F1 to count toward cross-framework inconsistency stats."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions)
    by_source_dir = Path(args.by_source_dir)
    output_dir = Path(args.output_dir) if args.output_dir else predictions_path.parent / "qualitative_analysis"

    prediction_rows = load_prediction_rows(predictions_path)
    source_cache = build_source_cache(by_source_dir)
    rows = attach_source_rows(prediction_rows, source_cache)

    slice_analysis = build_slice_analysis(
        rows,
        top_k=args.top_k,
        max_examples_per_pair=args.max_examples_per_pair,
    )
    framework_inconsistencies = analyze_framework_inconsistencies(
        rows,
        top_k=args.top_k,
        max_examples_per_pair=args.max_examples_per_pair,
        min_support_per_framework=args.min_support_per_framework,
    )
    rare_label_analysis = analyze_rare_labels(
        rows,
        rare_threshold=args.rare_threshold,
        top_k=args.top_k,
        max_examples_per_pair=args.max_examples_per_pair,
    )
    framework_rare_label_analysis = analyze_framework_rare_labels(
        rows,
        rare_threshold=args.rare_threshold,
        top_k=args.top_k,
        max_examples_per_pair=args.max_examples_per_pair,
    )

    payload = {
        "predictions_path": str(predictions_path),
        "num_predictions": len(rows),
        "slice_analysis": slice_analysis,
        "framework_inconsistencies": framework_inconsistencies,
        "rare_label_analysis": rare_label_analysis,
        "framework_rare_label_analysis": framework_rare_label_analysis,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "qualitative_analysis.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    metrics_rows = []
    for section_name, section_payload in [
        ("overall", {"overall": slice_analysis["overall"]}),
        ("language", slice_analysis["by_language"]),
        ("framework", slice_analysis["by_framework"]),
        ("dataset", slice_analysis["by_dataset"]),
    ]:
        for key, summary in section_payload.items():
            metrics_rows.append(
                {
                    "section": section_name,
                    "group_key": key,
                    "support": summary["support"],
                    "num_gold_labels": summary["num_gold_labels"],
                    "accuracy": summary["accuracy"],
                    "macro_f1": summary["macro_f1"],
                    "weighted_f1": summary["weighted_f1"],
                    "num_errors": summary["num_errors"],
                }
            )
    write_tsv(
        output_dir / "slice_metrics.tsv",
        metrics_rows,
        fieldnames=[
            "section",
            "group_key",
            "support",
            "num_gold_labels",
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "num_errors",
        ],
    )

    confusion_rows = []
    for section_name, section_payload in [
        ("overall", {"overall": slice_analysis["overall"]}),
        ("language", slice_analysis["by_language"]),
        ("framework", slice_analysis["by_framework"]),
        ("dataset", slice_analysis["by_dataset"]),
    ]:
        for key, summary in section_payload.items():
            for item in summary["top_confusions"]:
                confusion_rows.append(
                    {
                        "section": section_name,
                        "group_key": key,
                        "gold_label": item["gold_label"],
                        "pred_label": item["pred_label"],
                        "count": item["count"],
                    }
                )
    write_tsv(
        output_dir / "slice_top_confusions.tsv",
        confusion_rows,
        fieldnames=["section", "group_key", "gold_label", "pred_label", "count"],
    )

    all_error_rows = [asdict(row) for row in rows if row.gold_label != row.pred_label]
    write_tsv(
        output_dir / "all_errors.tsv",
        all_error_rows,
        fieldnames=[
            "corpus",
            "row_in_corpus",
            "framework",
            "language",
            "gold_label",
            "pred_label",
            "dir",
            "rel_type",
            "orig_label",
            "unit1_txt",
            "unit2_txt",
        ],
    )

    variation_rows = []
    for item in framework_inconsistencies["labels_sorted_by_f1_range"]:
        base = {
            "label": item["label"],
            "overall_support": item["overall_support"],
            "f1_range": item["f1_range"],
            "f1_std": item["f1_std"],
            "framework_count_meeting_threshold": item["framework_count_meeting_threshold"],
        }
        per_framework = {f"f1_{entry['framework']}": entry["f1"] for entry in item["per_framework"]}
        per_framework.update({f"support_{entry['framework']}": entry["support"] for entry in item["per_framework"]})
        variation_rows.append(base | per_framework)

    framework_names = sorted({row.framework for row in rows})
    variation_fieldnames = [
        "label",
        "overall_support",
        "f1_range",
        "f1_std",
        "framework_count_meeting_threshold",
    ]
    for framework in framework_names:
        variation_fieldnames.append(f"f1_{framework}")
    for framework in framework_names:
        variation_fieldnames.append(f"support_{framework}")
    write_tsv(
        output_dir / "framework_label_variation.tsv",
        variation_rows,
        fieldnames=variation_fieldnames,
    )

    rare_rows = []
    rare_confusion_rows = []
    for item in rare_label_analysis["labels"]:
        rare_rows.append(
            {
                "label": item["label"],
                "overall_support": item["overall_support"],
                "overall_precision": item["overall_precision"],
                "overall_recall": item["overall_recall"],
                "overall_f1": item["overall_f1"],
            }
        )
        for confusion in item["top_confusions"]:
            rare_confusion_rows.append(
                {
                    "gold_label": confusion["gold_label"],
                    "pred_label": confusion["pred_label"],
                    "count": confusion["count"],
                }
            )
    write_tsv(
        output_dir / "rare_label_summary.tsv",
        rare_rows,
        fieldnames=[
            "label",
            "overall_support",
            "overall_precision",
            "overall_recall",
            "overall_f1",
        ],
    )
    write_tsv(
        output_dir / "rare_label_confusions.tsv",
        rare_confusion_rows,
        fieldnames=["gold_label", "pred_label", "count"],
    )

    framework_rare_rows = []
    framework_rare_confusion_rows = []
    for framework_block in framework_rare_label_analysis["frameworks"]:
        for item in framework_block["labels"]:
            framework_rare_rows.append(
                {
                    "framework": framework_block["framework"],
                    "label": item["label"],
                    "support": item["support"],
                    "precision": item["precision"],
                    "recall": item["recall"],
                    "f1": item["f1"],
                }
            )
            for confusion in item["top_confusions"]:
                framework_rare_confusion_rows.append(
                    {
                        "framework": framework_block["framework"],
                        "gold_label": confusion["gold_label"],
                        "pred_label": confusion["pred_label"],
                        "count": confusion["count"],
                    }
                )
    write_tsv(
        output_dir / "framework_rare_label_summary.tsv",
        framework_rare_rows,
        fieldnames=["framework", "label", "support", "precision", "recall", "f1"],
    )
    write_tsv(
        output_dir / "framework_rare_label_confusions.tsv",
        framework_rare_confusion_rows,
        fieldnames=["framework", "gold_label", "pred_label", "count"],
    )

    write_markdown_report(output_dir / "qualitative_analysis.md", payload)

    print(
        json.dumps(
            {
                "num_predictions": len(rows),
                "overall_accuracy": slice_analysis["overall"]["accuracy"],
                "overall_macro_f1": slice_analysis["overall"]["macro_f1"],
                "overall_num_errors": slice_analysis["overall"]["num_errors"],
                "num_rare_labels": rare_label_analysis["num_rare_labels"],
                "num_framework_rare_label_entries": framework_rare_label_analysis["num_framework_rare_label_entries"],
                "output_dir": str(output_dir),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
