"""Prompt-based Qwen baseline for DISRPT Task 3.

Loads Qwen/Qwen3-4B, prompts it on the cleaned DISRPT TSV data,
and evaluates the predicted discourse relation labels.

The script supports:

    * zero-shot prompting
    * few-shot prompting from train.tsv examples
    * optional metadata in the prompt (`dir`, `rel_type`)

The per-dataset JSON follows the same general structure as the existing
`results/per_dataset_eval/*.json` files and includes:

    * pooled metrics
    * per corpus
    * per framework
    * per language
    * per-label global metrics

This script does **not** fine-tune the model; it uses prompting only.

Examples:
    1. zero-shot
    python qwen_prompt_baseline.py --split test
    or (Small-scale debugging)
    python qwen_prompt_baseline.py --split test --max-examples 100 --log-freq 50
    2. few-shot (3-shot)
    python qwen_prompt_baseline.py --split test --few-shot-k 3
    3. few-shot (3-shot) + metadata
    python qwen_prompt_baseline.py --split test --few-shot-k 3 --include-dir --include-rel-type
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

LABEL_SET = [
    "alternation",
    "attribution",
    "causal",
    "comment",
    "concession",
    "condition",
    "conjunction",
    "contrast",
    "elaboration",
    "explanation",
    "frame",
    "mode",
    "organization",
    "purpose",
    "query",
    "reformulation",
    "temporal",
]
INVALID_LABEL = "__invalid__"
CORPUS_FRAMEWORK_OVERRIDES = {
    "eng.erst.gum": "rst",
}
DEFAULT_DATA_DIR = Path("results/processed_tsv/by_split")
DEFAULT_BY_SOURCE_DIR = Path("results/processed_tsv/by_source_file")
DEFAULT_PER_DATASET_OUT_DIR = Path("results/per_dataset_eval")
DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B"
DEFAULT_OUTPUT_DIR = Path("results/qwen_prompt_results")
DEFAULT_SEED = 42
DEFAULT_DTYPE = "auto"
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)
ROW_KEY_COLUMNS = ["unit1_txt", "unit2_txt", "dir", "rel_type", "orig_label", "label"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_tsv(path: Path) -> list[dict[str, str]]:
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(10 ** 9)

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", restval="")
        return list(reader)


def load_rows(path: Path, max_examples: int | None = None) -> list[dict[str, str]]:
    rows = read_tsv(path)
    clean_rows = []
    for row in rows:
        if not row.get("unit1_txt") or not row.get("unit2_txt") or not row.get("label"):
            continue
        clean_rows.append(row)
    if max_examples is not None:
        clean_rows = clean_rows[:max_examples]
    return clean_rows


def corpus_framework(corpus: str) -> str:
    if corpus in CORPUS_FRAMEWORK_OVERRIDES:
        return CORPUS_FRAMEWORK_OVERRIDES[corpus]
    parts = corpus.split(".")
    return parts[1] if len(parts) >= 2 else "unknown"


def corpus_language(corpus: str) -> str:
    parts = corpus.split(".")
    return parts[0] if parts else "unknown"


def row_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple((row.get(column) or "").strip() for column in ROW_KEY_COLUMNS)


def build_corpus_lookup(split: str) -> dict[tuple[str, ...], deque[str]]:
    lookup: dict[tuple[str, ...], deque[str]] = defaultdict(deque)
    if not DEFAULT_BY_SOURCE_DIR.exists():
        return lookup

    for corpus_dir in sorted(path for path in DEFAULT_BY_SOURCE_DIR.iterdir() if path.is_dir()):
        corpus = corpus_dir.name
        split_path = corpus_dir / f"{corpus}_{split}.tsv"
        if not split_path.exists():
            continue
        for row in load_rows(split_path):
            lookup[row_key(row)].append(corpus)
    return lookup


def attach_corpus_labels(rows: list[dict[str, str]], split: str) -> int:
    lookup = build_corpus_lookup(split)
    missing = 0
    for row in rows:
        queue = lookup.get(row_key(row))
        if queue:
            row["corpus"] = queue.popleft()
        else:
            row["corpus"] = "unknown"
            missing += 1
    return missing


def sample_few_shot_rows(
        rows: list[dict[str, str]],
        k: int,
        seed: int,
) -> list[dict[str, str]]:
    if k <= 0 or not rows:
        return []
    rng = random.Random(seed)
    if k >= len(rows):
        return list(rows)
    return rng.sample(rows, k)


def build_instruction(label_set: list[str]) -> str:
    labels = ", ".join(label_set)
    return (
        "You are a discourse relation classification system.\n"
        "Given two discourse units, predict the discourse relation label between them.\n"
        f"Choose exactly one label from this list:\n{labels}\n"
        "Return only one label and nothing else."
    )


def format_instance(
        row: dict[str, str],
        *,
        include_dir: bool,
        include_rel_type: bool,
) -> str:
    parts = [
        f"Unit 1: {row['unit1_txt']}",
        f"Unit 2: {row['unit2_txt']}",
    ]
    if include_dir:
        parts.append(f"Direction: {row.get('dir', '').strip()}")
    if include_rel_type:
        parts.append(f"Relation type: {row.get('rel_type', '').strip()}")
    return "\n".join(parts)


def format_few_shot_example(
        row: dict[str, str],
        *,
        include_dir: bool,
        include_rel_type: bool,
) -> str:
    return (
        f"{format_instance(row, include_dir=include_dir, include_rel_type=include_rel_type)}\n"
        f"Answer: {row['label']}"
    )


def build_prompt(
        row: dict[str, str],
        *,
        few_shot_rows: list[dict[str, str]],
        include_dir: bool,
        include_rel_type: bool,
) -> tuple[str, str]:
    system_prompt = build_instruction(LABEL_SET)
    user_parts: list[str] = []

    if few_shot_rows:
        user_parts.append("Examples:")
        for idx, ex in enumerate(few_shot_rows, start=1):
            user_parts.append(
                f"Example {idx}:\n{format_few_shot_example(ex, include_dir=include_dir, include_rel_type=include_rel_type)}")

    user_parts.append("Now classify this example:")
    user_parts.append(format_instance(row, include_dir=include_dir, include_rel_type=include_rel_type))
    user_parts.append("Answer:")
    user_prompt = "\n\n".join(user_parts)
    return system_prompt, user_prompt


def render_prompt(
        tokenizer: AutoTokenizer,
        system_prompt: str,
        user_prompt: str,
) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return f"{system_prompt}\n\n{user_prompt}"


def normalize_prediction(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return INVALID_LABEL

    cleaned = THINK_BLOCK_RE.sub(" ", cleaned).strip()
    cleaned = THINK_TAG_RE.sub(" ", cleaned).strip()
    if not cleaned:
        return INVALID_LABEL

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    candidates: list[str] = []
    if lines:
        candidates.extend(reversed(lines))
    candidates.append(cleaned)

    for candidate in candidates:
        stripped = candidate.strip("`*_\"'.,:;()[]{} ")
        lower = stripped.lower()

        if lower in LABEL_SET:
            return lower

        label_after_colon = re.search(r"(?:answer|label)\s*:\s*([a-zA-Z\-]+)", lower)
        if label_after_colon:
            possible = label_after_colon.group(1).strip()
            if possible in LABEL_SET:
                return possible

        matches = [label for label in LABEL_SET if re.search(rf"\b{re.escape(label)}\b", lower)]
        if len(matches) == 1:
            return matches[0]

    all_matches = [label for label in LABEL_SET if label in cleaned.lower()]
    if len(set(all_matches)) == 1:
        return all_matches[0]

    return INVALID_LABEL


def load_model_and_tokenizer() -> tuple[PreTrainedTokenizerBase, PreTrainedModel, torch.device]:
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME, trust_remote_code=False)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"trust_remote_code": False}
    if DEFAULT_DTYPE != "auto":
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        model_kwargs["torch_dtype"] = dtype_map[DEFAULT_DTYPE]

    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL_NAME, **model_kwargs)
        model_device = next(model.parameters()).device
    else:
        model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL_NAME, **model_kwargs)
        model_device = torch.device("cpu")
        model.to(model_device)

    model.eval()
    return tokenizer, model, model_device


def generate_label(
        *,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        model_device: torch.device,
        prompt: str,
        max_new_tokens: int,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": False,
    }

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    generated = outputs[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def evaluate_predictions(
        y_true: list[str],
        y_pred: list[str],
) -> dict[str, Any]:
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=LABEL_SET, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=LABEL_SET, average="weighted", zero_division=0))
    report = classification_report(
        y_true,
        y_pred,
        labels=LABEL_SET,
        output_dict=True,
        zero_division=0,
    )
    cm_labels = LABEL_SET + ([INVALID_LABEL] if INVALID_LABEL in y_pred else [])
    cm = confusion_matrix(y_true, y_pred, labels=cm_labels).tolist()
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report,
        "confusion_matrix": cm,
        "confusion_matrix_labels": cm_labels,
        "invalid_predictions": sum(pred == INVALID_LABEL for pred in y_pred),
    }


def slice_report_str_labels(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    if not y_true:
        return {
            "support": 0,
            "num_gold_labels": 0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "per_label": {},
        }

    present_labels = sorted(set(y_true) | set(y_pred))
    report = classification_report(
        y_true,
        y_pred,
        labels=present_labels,
        target_names=present_labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "support": len(y_true),
        "num_gold_labels": len(set(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "per_label": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in present_labels
        },
    }


def aggregate_predictions_by_corpus(preds_by_corpus: dict[str, dict[str, list[str]]]) -> dict[str, Any]:
    per_corpus: dict[str, dict[str, Any]] = {}
    by_framework: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"gold": [], "pred": []})
    by_language: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"gold": [], "pred": []})
    all_gold: list[str] = []
    all_pred: list[str] = []

    for corpus, payload in preds_by_corpus.items():
        gold = payload["gold_label"]
        pred = payload["pred_label"]
        fw = corpus_framework(corpus)
        lg = corpus_language(corpus)
        per_corpus[corpus] = {
            "framework": fw,
            "language": lg,
            **slice_report_str_labels(gold, pred),
        }
        by_framework[fw]["gold"].extend(gold)
        by_framework[fw]["pred"].extend(pred)
        by_language[lg]["gold"].extend(gold)
        by_language[lg]["pred"].extend(pred)
        all_gold.extend(gold)
        all_pred.extend(pred)

    per_framework = {
        fw: {
            "corpora": sorted(c for c in preds_by_corpus if corpus_framework(c) == fw),
            **slice_report_str_labels(payload["gold"], payload["pred"]),
        }
        for fw, payload in sorted(by_framework.items())
    }
    per_language = {
        lg: {
            "corpora": sorted(c for c in preds_by_corpus if corpus_language(c) == lg),
            **slice_report_str_labels(payload["gold"], payload["pred"]),
        }
        for lg, payload in sorted(by_language.items())
    }
    pooled = slice_report_str_labels(all_gold, all_pred)
    return {
        "pooled": pooled,
        "per_corpus": per_corpus,
        "per_framework": per_framework,
        "per_language": per_language,
        "per_label_global": pooled["per_label"],
    }


def write_predictions(
        path: Path,
        rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "index",
        "corpus",
        "gold_label",
        "pred_label",
        "is_valid_prediction",
        "raw_generation",
        "unit1_txt",
        "unit2_txt",
        "dir",
        "rel_type",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n",
                                quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def predict_rows(
    rows: list[dict[str, str]],
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    model_device: torch.device,
    few_shot_rows: list[dict[str, str]],
    include_dir: bool,
    include_rel_type: bool,
    max_new_tokens: int,
    log_every: int,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    predictions: list[dict[str, Any]] = []
    y_true: list[str] = []
    y_pred: list[str] = []
    started = time.time()

    for idx, row in enumerate(rows):
        system_prompt, user_prompt = build_prompt(
            row,
            few_shot_rows=few_shot_rows,
            include_dir=include_dir,
            include_rel_type=include_rel_type,
        )
        prompt = render_prompt(tokenizer, system_prompt, user_prompt)
        raw_generation = generate_label(
            model=model,
            tokenizer=tokenizer,
            model_device=model_device,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        pred_label = normalize_prediction(raw_generation)

        y_true.append(row["label"])
        y_pred.append(pred_label)
        predictions.append(
            {
                "index": idx,
                "corpus": row.get("corpus", ""),
                "gold_label": row["label"],
                "pred_label": pred_label,
                "is_valid_prediction": pred_label != INVALID_LABEL,
                "raw_generation": raw_generation,
                "unit1_txt": row["unit1_txt"],
                "unit2_txt": row["unit2_txt"],
                "dir": row.get("dir", ""),
                "rel_type": row.get("rel_type", ""),
            }
        )

        if (idx + 1) % log_every == 0:
            elapsed = time.time() - started
            print(f"[{idx + 1}/{len(rows)}] elapsed={elapsed:.1f}s invalid={sum(p == INVALID_LABEL for p in y_pred)}")

    return predictions, y_true, y_pred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prompt Qwen3-4B on DISRPT TSV data.")
    parser.add_argument("--split", default="test", choices=["dev", "test"], help="Which split to evaluate.")
    parser.add_argument("--max-examples", type=int, default=None, help="Optional cap on evaluated examples.")
    parser.add_argument("--few-shot-k", type=int, default=0, help="Number of few-shot examples to prepend.")
    parser.add_argument("--include-dir", action="store_true", help="Include dir in the prompt.")
    parser.add_argument("--include-rel-type", action="store_true", help="Include rel_type in the prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--log-freq", dest="log_every", type=int, default=200, help="Print progress every N examples.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(DEFAULT_SEED)

    data_dir = DEFAULT_DATA_DIR
    eval_path = data_dir / f"{args.split}.tsv"
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing split TSV: {eval_path}")

    few_shot_path = data_dir / "train.tsv"
    if args.few_shot_k > 0 and not few_shot_path.exists():
        raise FileNotFoundError(f"Missing few-shot TSV: {few_shot_path}")

    eval_rows = load_rows(eval_path, max_examples=args.max_examples)
    unmatched_corpus_rows = attach_corpus_labels(eval_rows, args.split)
    few_shot_source = load_rows(few_shot_path) if args.few_shot_k > 0 else []
    few_shot_rows = sample_few_shot_rows(few_shot_source, args.few_shot_k, DEFAULT_SEED)

    tokenizer, model, model_device = load_model_and_tokenizer()
    predictions, y_true, y_pred = predict_rows(
        eval_rows,
        model=model,
        tokenizer=tokenizer,
        model_device=model_device,
        few_shot_rows=few_shot_rows,
        include_dir=args.include_dir,
        include_rel_type=args.include_rel_type,
        max_new_tokens=args.max_new_tokens,
        log_every=args.log_every,
    )

    metrics = evaluate_predictions(y_true, y_pred)
    summary = {
        "model": "Qwen Prompt Baseline",
        "model_name": DEFAULT_MODEL_NAME,
        "split": args.split,
        "num_examples": len(eval_rows),
        "label_set": LABEL_SET,
        "few_shot_k": args.few_shot_k,
        "include_dir": args.include_dir,
        "include_rel_type": args.include_rel_type,
        "input": "prompt(unit1_txt, unit2_txt, optional dir/rel_type)",
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
        "seed": DEFAULT_SEED,
        "dtype": DEFAULT_DTYPE,
        "data_dir": str(data_dir),
        "num_unmatched_corpus_rows": unmatched_corpus_rows,
        **metrics,
    }

    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_predictions(output_dir / f"{args.split}_predictions.tsv", predictions)

    preds_by_corpus: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"gold_label": [], "pred_label": []})
    for prediction in predictions:
        corpus = prediction.get("corpus") or "unknown"
        preds_by_corpus[corpus]["gold_label"].append(prediction["gold_label"])
        preds_by_corpus[corpus]["pred_label"].append(prediction["pred_label"])

    if preds_by_corpus:
        agg = aggregate_predictions_by_corpus(preds_by_corpus)
        per_dataset_payload = {
            "model": "Qwen Prompt Baseline",
            "model_name": DEFAULT_MODEL_NAME,
            "max_length": None,
            "label_set": LABEL_SET,
            "few_shot_k": args.few_shot_k,
            "include_dir": args.include_dir,
            "include_rel_type": args.include_rel_type,
            "input": "prompt(unit1_txt, unit2_txt, optional dir/rel_type)",
            "seed": DEFAULT_SEED,
            "dtype": DEFAULT_DTYPE,
            **agg,
        }
        per_dataset_out_dir = DEFAULT_PER_DATASET_OUT_DIR
        per_dataset_out_dir.mkdir(parents=True, exist_ok=True)
        (per_dataset_out_dir / f"qwen_prompt_{args.split}.json").write_text(
            json.dumps(per_dataset_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(json.dumps(
        {
            "accuracy": summary["accuracy"],
            "macro_f1": summary["macro_f1"],
            "weighted_f1": summary["weighted_f1"],
            "invalid_predictions": summary["invalid_predictions"],
            "num_examples": summary["num_examples"],
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
