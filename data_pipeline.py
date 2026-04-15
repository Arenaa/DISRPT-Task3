from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


KEEP_COLUMNS = [
    "unit1_txt",
    "unit2_txt",
    "dir",
    "rel_type",
    "orig_label",
    "label",
]

WHITESPACE_RE = re.compile(r"\s+")
LABEL_NORMALIZATION = {
    # In this dataset, `contingency.cause.result` is mapped to `causal`
    # almost everywhere, and `contingency.cause` appears as a single
    # inconsistent outlier. Normalize it during cleaning to keep a strict
    # 17-label setup across TSV export, verification, and EDA.
    "contingency.cause": "causal",
}


@dataclass
class FileSummary:
    source_file: str
    dataset: str
    split: str
    num_rows_before_cleaning: int
    num_rows_after_cleaning: int
    num_malformed_rows: int
    num_rows_with_empty_fields: int
    num_duplicate_rows_removed: int


def normalize_text(value: str) -> str:
    return WHITESPACE_RE.sub(" ", value.strip())


def clean_row(row: dict[str, str]) -> dict[str, str]:
    cleaned = {column: normalize_text(row[column]) for column in KEEP_COLUMNS}
    cleaned["label"] = LABEL_NORMALIZATION.get(cleaned["label"], cleaned["label"])
    return cleaned


def validate_required_columns(header: list[str], path: Path) -> None:
    missing_columns = [column for column in KEEP_COLUMNS if column not in header]
    if missing_columns:
        raise ValueError(f"Missing required columns in {path}: {missing_columns}")


def infer_split(path: Path) -> str:
    stem = path.stem
    for split in ("train", "dev", "test"):
        if stem.endswith(f"_{split}"):
            return split
    return "unknown"


def iter_rels_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.rglob("*.rels"))


def load_and_clean_rels(path: Path) -> tuple[list[dict[str, str]], FileSummary]:
    return load_and_clean_rels_with_metadata(path, include_metadata=False)


def load_and_clean_rels_with_metadata(
    path: Path,
    include_metadata: bool = True,
) -> tuple[list[dict[str, str]], FileSummary]:
    cleaned_rows: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    raw_rows = 0
    dropped_malformed = 0
    dropped_missing = 0
    dropped_duplicates = 0

    with path.open("r", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        validate_required_columns(header, path)

        for line_no, line in enumerate(handle, start=2):
            raw_rows += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(header):
                dropped_malformed += 1
                continue

            raw_row = dict(zip(header, parts))
            row = clean_row(raw_row)

            if any(row[column] == "" for column in KEEP_COLUMNS):
                dropped_missing += 1
                continue

            key = tuple(row[column] for column in KEEP_COLUMNS)
            if key in seen:
                dropped_duplicates += 1
                continue

            seen.add(key)
            if include_metadata:
                cleaned_rows.append(
                    {
                        "file": str(path),
                        "dataset": path.parent.name,
                        "split": infer_split(path),
                        "line_no": str(line_no),
                        **row,
                    }
                )
            else:
                cleaned_rows.append(row)

    summary = FileSummary(
        source_file=str(path),
        dataset=path.parent.name,
        split=infer_split(path),
        num_rows_before_cleaning=raw_rows,
        num_rows_after_cleaning=len(cleaned_rows),
        num_malformed_rows=dropped_malformed,
        num_rows_with_empty_fields=dropped_missing,
        num_duplicate_rows_removed=dropped_duplicates,
    )
    return cleaned_rows, summary


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(KEEP_COLUMNS) + "\n")
        for row in rows:
            handle.write("\t".join(row[column] for column in KEEP_COLUMNS) + "\n")


def build_pipeline(input_dir: Path, output_dir: Path) -> dict[str, object]:
    rels_files = iter_rels_files(input_dir)
    combined_by_split: dict[str, list[dict[str, str]]] = {"train": [], "dev": [], "test": [], "unknown": []}
    file_summaries: list[FileSummary] = []

    for rels_file in rels_files:
        rows, summary = load_and_clean_rels(rels_file)
        file_summaries.append(summary)
        combined_by_split.setdefault(summary.split, []).extend(rows)

        relative = rels_file.relative_to(input_dir)
        per_source_output = output_dir / "by_source_file" / relative.parent / f"{rels_file.stem}.tsv"
        write_tsv(per_source_output, rows)

    combined_counts: dict[str, int] = {}
    by_split_dir = output_dir / "by_split"
    for split, rows in combined_by_split.items():
        if not rows:
            continue
        combined_counts[split] = len(rows)
        write_tsv(by_split_dir / f"{split}.tsv", rows)

    totals = {
        "num_files": len(file_summaries),
        "num_rows_before_cleaning": sum(item.num_rows_before_cleaning for item in file_summaries),
        "num_rows_after_cleaning": sum(item.num_rows_after_cleaning for item in file_summaries),
        "num_malformed_rows": sum(item.num_malformed_rows for item in file_summaries),
        "num_rows_with_empty_fields": sum(item.num_rows_with_empty_fields for item in file_summaries),
        "num_duplicate_rows_removed": sum(item.num_duplicate_rows_removed for item in file_summaries),
    }

    summary_payload = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "keep_columns": KEEP_COLUMNS,
        "totals": totals,
        "num_rows_by_split": combined_counts,
        "files": [asdict(item) for item in file_summaries],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return summary_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DISRPT .rels files into cleaned TSV files.")
    parser.add_argument("--input-dir", default="data_subset", help="Directory containing DISRPT .rels files.")
    parser.add_argument("--output-dir", default="processed_tsv", help="Directory for cleaned TSV outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    summary = build_pipeline(input_dir=input_dir, output_dir=output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
