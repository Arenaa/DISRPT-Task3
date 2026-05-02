"""Supervised instruction fine-tuning (SFT) of Qwen causal LMs for DISRPT Task 3.

Training matches standard instruction tuning: user content is built from
``build_prompt`` (minus the trailing ``## Answer:`` cue), optional system
message, and the tokenizer's ``chat_template`` when present (Qwen2 / Qwen3);
loss is computed only on the assistant turn (the gold label + EOS). When no
``chat_template`` exists, tokenization falls back to the same masked-target
layout as before (prompt tokens ignored in the loss).

Implemented here:

    * full-parameter (or checkpointed) supervised instruction SFT
    * verbose instruction-style user content with:
        - language / corpus / framework (LCF)
        - direction
        - relation type
        - only the fields preserved in the cleaned processed TSV files
    * greedy chat-template decoding on dev/test when supported
    * pooled and per-dataset evaluation artifacts

Not reproduced exactly from the paper:

    * model pruning to stay under a shared-task parameter cap
    * translation-based data augmentation
    * richer raw-document features that are no longer available after TSV export,
      such as same-speaker annotations, sentence context windows, and exact
      token-distance features

Examples:

    python scripts/qwen_prompt_baseline.py

    python scripts/qwen_prompt_baseline.py --max-train-examples 2000 --max-dev-examples 200

    python scripts/qwen_prompt_baseline.py --epochs 1 --train-batch-size 1 \
        --gradient-accumulation-steps 16 --eval-max-new-tokens 16
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None

DEFAULT_INSTRUCTION_SYSTEM = (
    "You solve discourse relation classification. Reply with exactly one lowercase "
    "label from the task label set and no other text."
)

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
LABEL_GLOSSES = {
    "alternation": "alternating options or choices",
    "attribution": "reported speech, thought, or attributed content",
    "causal": "cause, result, or reason relation",
    "comment": "speaker or writer comment on content",
    "concession": "unexpected outcome despite expectation",
    "condition": "conditional relation",
    "conjunction": "joint or additive relation",
    "contrast": "contrast or opposition",
    "elaboration": "additional detail, expansion, or specification",
    "explanation": "explanatory justification or clarification",
    "frame": "background or framing information",
    "mode": "manner or way of doing something",
    "organization": "text-organizing or structural relation",
    "purpose": "goal or intended outcome",
    "query": "question-answer or request relation",
    "reformulation": "restatement, paraphrase, or reformulation",
    "temporal": "temporal sequence or timing relation",
}
INVALID_LABEL = "__invalid__"
CORPUS_FRAMEWORK_OVERRIDES = {"eng.erst.gum": "rst"}
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)

DEFAULT_INPUT_DIR = Path("results/processed_tsv/by_source_file")
DEFAULT_OUTPUT_DIR = Path("results/qwen_sft_results")
DEFAULT_PER_DATASET_OUT_DIR = Path("results/per_dataset_eval")
DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B"
REQUIRED_TSV_COLUMNS = ["unit1_txt", "unit2_txt", "dir", "rel_type", "orig_label", "label"]


@dataclass(frozen=True)
class Example:
    split: str
    corpus_id: str
    language: str
    corpus: str
    framework_prompt: str
    framework_group: str
    unit1_txt: str
    unit2_txt: str
    direction: str
    rel_type: str
    label: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def corpus_framework(corpus_id: str) -> str:
    if corpus_id in CORPUS_FRAMEWORK_OVERRIDES:
        return CORPUS_FRAMEWORK_OVERRIDES[corpus_id]
    parts = corpus_id.split(".")
    return parts[1] if len(parts) >= 2 else "unknown"


def corpus_language(corpus_id: str) -> str:
    parts = corpus_id.split(".")
    return parts[0] if parts else "unknown"


def corpus_name(corpus_id: str) -> str:
    parts = corpus_id.split(".")
    return parts[-1] if parts else corpus_id


def prompt_framework(corpus_id: str) -> str:
    parts = corpus_id.split(".")
    return parts[1] if len(parts) >= 2 else "unknown"


def validate_required_columns(header: list[str], path: Path) -> None:
    missing = [column for column in REQUIRED_TSV_COLUMNS if column not in header]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")


def direction_text(raw_dir: str) -> str:
    if raw_dir == "1>2":
        return "From Unit1 to Unit2."
    if raw_dir == "1<2":
        return "From Unit2 to Unit1."
    return raw_dir or "Unknown."


def build_prompt(example: Example) -> str:
    labels = ", ".join(LABEL_SET)
    prompt = (
        "## Role and Goal:\n"
        "You are an expert in discourse analysis, tasked with identifying the discourse relation "
        "between two sentence units based on the provided label. Your goal is to accurately "
        "determine the relationship between these two units.\n"
        "## Guidelines:\n"
        "1. You will receive Unit1 and Unit2. Unit1 appears before Unit2 in the original text.\n"
        "2. You will also be informed about the language of these units.\n"
        "3. You will also be informed of the corpus from which the data is drawn, which may help "
        "guide your analysis.\n"
        "4. The framework for analysis will be provided, outlining the structure used for discourse analysis.\n"
        "5. The direction of the relationship between these two units will be given.\n"
        "6. The relation type metadata will also be provided.\n"
        "7. You will be provided with a set of labels representing possible discourse relations. "
        "Choose one label that best fits the relationship between Unit1 and Unit2, and output only the chosen label.\n"
        "8. Do not explain your answer.\n"
        "9. Output exactly one label and nothing else.\n"
        "## Labels:\n"
        f"{labels}\n"
        "## Label Hints:\n"
        f"{format_label_hints()}\n"
        "## Language:\n"
        f"{example.language}\n"
        "## Corpus:\n"
        f"{example.corpus}\n"
        "## Framework:\n"
        f"{example.framework_prompt}\n"
        "## Direction:\n"
        f"{direction_text(example.direction)}\n"
        "## Relation Type:\n"
        f"{example.rel_type or 'Unknown'}\n"
        "## Unit1:\n"
        f"{example.unit1_txt}\n"
        "## Unit2:\n"
        f"{example.unit2_txt}\n"
        "## Answer:\n"
    )
    return prompt


def chat_template_extra_kwargs(tokenizer: PreTrainedTokenizerBase) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    try:
        sig = inspect.signature(tokenizer.apply_chat_template)
        if "enable_thinking" in sig.parameters:
            kwargs["enable_thinking"] = False
    except (TypeError, ValueError):
        pass
    return kwargs


def as_token_id_list(ids: Any) -> list[int]:
    """Normalize ``apply_chat_template(..., tokenize=True)`` output."""
    if ids is None:
        raise ValueError("apply_chat_template returned no token ids")
    if hasattr(ids, "input_ids"):
        return as_token_id_list(ids["input_ids"])
    if isinstance(ids, dict) and "input_ids" in ids:
        return as_token_id_list(ids["input_ids"])
    if isinstance(ids, list):
        return [int(t) for t in ids]
    if isinstance(ids, torch.Tensor):
        flat = ids.detach().cpu().squeeze()
        if flat.ndim != 1:
            raise ValueError(f"Expected 1d token ids, got shape {tuple(flat.shape)}")
        return [int(x) for x in flat.tolist()]
    try:
        if isinstance(ids, np.ndarray):
            flat = np.squeeze(ids)
            return [int(x) for x in flat.reshape(-1).tolist()]
    except (ImportError, AttributeError):
        pass
    raise TypeError(f"Unexpected chat template tokenize output type: {type(ids)}")


def user_block_for_chat(example: Example) -> str:
    """Strip trailing ``## Answer:`` so the assistant turn carries only the label."""
    text = build_prompt(example)
    suffix = "## Answer:\n"
    if text.endswith(suffix):
        text = text[: -len(suffix)].rstrip()
    return text + "\n\nRespond with exactly one lowercase label from the label list and nothing else."


def tokenize_chat_supervised(
    tokenizer: PreTrainedTokenizerBase,
    example: Example,
    max_length: int,
    *,
    system: str | None,
    chat_kw: dict[str, Any],
) -> dict[str, torch.Tensor]:
    user_content = user_block_for_chat(example)
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": example.label})

    if getattr(tokenizer, "chat_template", None):
        prompt_messages = messages[:-1]
        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors=None,
            **chat_kw,
        )
        full_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors=None,
            **chat_kw,
        )
        prompt_ids = as_token_id_list(prompt_ids)
        full_ids = as_token_id_list(full_ids)
        plen = len(prompt_ids)
        if full_ids[:plen] != prompt_ids:
            plen = min(plen, len(full_ids))
        labels_list = [-100] * plen + list(full_ids[plen:])
        input_ids = full_ids[:max_length]
        labels = labels_list[:max_length]
        if len(input_ids) < len(full_ids):
            labels = labels[: len(input_ids)]
    else:
        target_ids = tokenizer(example.label, add_special_tokens=False).input_ids
        eos_id = tokenizer.eos_token_id
        if eos_id is not None:
            target_ids = target_ids + [int(eos_id)]
        prompt_ids = tokenizer(
            user_content,
            add_special_tokens=True,
            truncation=True,
            max_length=max(1, max_length - len(target_ids)),
        ).input_ids
        input_ids = (prompt_ids + target_ids)[:max_length]
        labels = ([-100] * len(prompt_ids) + target_ids)[:max_length]

    attention_mask = [1] * len(input_ids)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Tokenizer needs pad_token_id for batching.")
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class ChatInstructionSFTDataset(Dataset):
    def __init__(
        self,
        examples: list[Example],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        *,
        system_prompt: str | None,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        self._chat_kw = chat_template_extra_kwargs(tokenizer)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return tokenize_chat_supervised(
            self.tokenizer,
            self.examples[idx],
            self.max_length,
            system=self.system_prompt,
            chat_kw=self._chat_kw,
        )


def format_label_hints() -> str:
    return "\n".join(f"- {label}: {LABEL_GLOSSES[label]}" for label in LABEL_SET)


def read_tsv(path: Path) -> list[dict[str, str]]:
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(10 ** 9)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t", restval=""))


def read_examples_from_processed_tsv(
        by_source_dir: Path,
        max_examples_by_split: dict[str, int | None] | None = None,
) -> dict[str, list[Example]]:
    max_examples_by_split = max_examples_by_split or {}
    examples_by_split: dict[str, list[Example]] = {"train": [], "dev": [], "test": []}

    for corpus_dir in sorted(path for path in by_source_dir.iterdir() if path.is_dir()):
        corpus_id = corpus_dir.name
        lang = corpus_language(corpus_id)
        corp = corpus_name(corpus_id)
        framework_prompt_name = prompt_framework(corpus_id)
        framework_group = corpus_framework(corpus_id)

        for split in ("train", "dev", "test"):
            split_path = corpus_dir / f"{corpus_id}_{split}.tsv"
            if not split_path.exists():
                continue
            rows = read_tsv(split_path)
            if rows:
                validate_required_columns(list(rows[0].keys()), split_path)

            for row in rows:
                if any(not (row.get(column) or "").strip() for column in ("unit1_txt", "unit2_txt", "label")):
                    continue
                examples_by_split[split].append(
                    Example(
                        split=split,
                        corpus_id=corpus_id,
                        language=lang,
                        corpus=corp,
                        framework_prompt=framework_prompt_name,
                        framework_group=framework_group,
                        unit1_txt=(row.get("unit1_txt") or "").strip(),
                        unit2_txt=(row.get("unit2_txt") or "").strip(),
                        direction=(row.get("dir") or "").strip(),
                        rel_type=(row.get("rel_type") or "").strip(),
                        label=(row.get("label") or "").strip(),
                    )
                )

            limit = max_examples_by_split.get(split)
            if limit is not None and len(examples_by_split[split]) >= limit:
                examples_by_split[split] = examples_by_split[split][:limit]

    for split, limit in max_examples_by_split.items():
        if limit is not None:
            examples_by_split[split] = examples_by_split[split][:limit]

    return examples_by_split


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--per-dataset-out-dir", default=str(DEFAULT_PER_DATASET_OUT_DIR))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--eval-max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_INSTRUCTION_SYSTEM,
        help="System message for chat-template instruction tuning; empty string disables.",
    )
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-dev-examples", type=int, default=None)
    parser.add_argument("--max-test-examples", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--skip-per-dataset", action="store_true")
    parser.add_argument("--no-save-best", dest="save_best", action="store_false")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    return parser


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def pad_batch(
        batch: list[dict[str, torch.Tensor]],
        *,
        pad_token_id: int,
) -> dict[str, torch.Tensor]:
    input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=pad_token_id)
    attention_mask = pad_sequence([item["attention_mask"] for item in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=-100)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


def load_model_and_tokenizer(
        model_name: str,
        dtype_name: str,
        device: torch.device,
        gradient_checkpointing: bool,
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.pad_token is None:
        raise ValueError("Tokenizer must define either pad_token, eos_token, or unk_token.")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=False,
        torch_dtype=resolve_dtype(dtype_name),
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.to(device)
    return tokenizer, model


def build_loaders(
        examples_by_split: dict[str, list[Example]],
        tokenizer: PreTrainedTokenizerBase,
        args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    system_prompt = (getattr(args, "system_prompt", None) or "").strip() or None
    train_dataset = ChatInstructionSFTDataset(
        examples_by_split["train"], tokenizer, args.max_length, system_prompt=system_prompt
    )
    dev_dataset = ChatInstructionSFTDataset(
        examples_by_split["dev"], tokenizer, args.max_length, system_prompt=system_prompt
    )
    test_dataset = ChatInstructionSFTDataset(
        examples_by_split["test"], tokenizer, args.max_length, system_prompt=system_prompt
    )

    collate = lambda batch: pad_batch(batch, pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)
    return train_loader, dev_loader, test_loader


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


def encode_prompt_for_generation(
    tokenizer: PreTrainedTokenizerBase,
    example: Example,
    *,
    system: str | None,
    chat_kw: dict[str, Any],
) -> torch.Tensor:
    user_content = user_block_for_chat(example)
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_content})
    if getattr(tokenizer, "chat_template", None):
        raw = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors=None,
            **chat_kw,
        )
        flat = as_token_id_list(raw)
        return torch.tensor([flat], dtype=torch.long)
    return tokenizer(user_content, return_tensors="pt", truncation=True).input_ids


def generate_predictions_chat(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    examples: list[Example],
    device: torch.device,
    *,
    max_new_tokens: int,
    system: str | None,
    log_every: int,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    chat_kw = chat_template_extra_kwargs(tokenizer)
    preds_out: list[dict[str, Any]] = []
    gold: list[str] = []
    pred: list[str] = []
    started = time.time()
    model.eval()
    pad_id = tokenizer.pad_token_id
    with torch.no_grad():
        it = examples
        if _tqdm is not None:
            it = _tqdm(examples, desc="eval_generate")
        for idx, ex in enumerate(it):
            input_ids = encode_prompt_for_generation(tokenizer, ex, system=system, chat_kw=chat_kw).to(device)
            out = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            gen_ids = out[0, input_ids.shape[1] :]
            raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            lab = normalize_prediction(raw)
            gold.append(ex.label)
            pred.append(lab)
            preds_out.append(
                {
                    "index": idx,
                    "corpus": ex.corpus_id,
                    "gold_label": ex.label,
                    "pred_label": lab,
                    "is_valid_prediction": lab != INVALID_LABEL,
                    "raw_generation": raw,
                    "unit1_txt": ex.unit1_txt,
                    "unit2_txt": ex.unit2_txt,
                    "dir": ex.direction,
                    "rel_type": ex.rel_type,
                }
            )
            if log_every > 0 and ((idx + 1) % log_every == 0 or idx + 1 == len(examples)):
                elapsed = time.time() - started
                invalid = sum(item == INVALID_LABEL for item in pred)
                print(f"[{idx + 1}/{len(examples)}] elapsed={elapsed:.1f}s invalid={invalid}")

    return preds_out, gold, pred


def evaluate_predictions(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
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
        framework = corpus_framework(corpus)
        language = corpus_language(corpus)
        per_corpus[corpus] = {
            "framework": framework,
            "language": language,
            **slice_report_str_labels(gold, pred),
        }
        by_framework[framework]["gold"].extend(gold)
        by_framework[framework]["pred"].extend(pred)
        by_language[language]["gold"].extend(gold)
        by_language[language]["pred"].extend(pred)
        all_gold.extend(gold)
        all_pred.extend(pred)

    per_framework = {
        framework: {
            "corpora": sorted(c for c in preds_by_corpus if corpus_framework(c) == framework),
            **slice_report_str_labels(payload["gold"], payload["pred"]),
        }
        for framework, payload in sorted(by_framework.items())
    }
    per_language = {
        language: {
            "corpora": sorted(c for c in preds_by_corpus if corpus_language(c) == language),
            **slice_report_str_labels(payload["gold"], payload["pred"]),
        }
        for language, payload in sorted(by_language.items())
    }
    pooled = slice_report_str_labels(all_gold, all_pred)
    return {
        "pooled": pooled,
        "per_corpus": per_corpus,
        "per_framework": per_framework,
        "per_language": per_language,
        "per_label_global": pooled["per_label"],
    }


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
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


def evaluate_split(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        examples: list[Example],
        args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    system_prompt = (getattr(args, "system_prompt", None) or "").strip() or None
    predictions, y_true, y_pred = generate_predictions_chat(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        device=torch.device(args.device),
        max_new_tokens=args.eval_max_new_tokens,
        system=system_prompt,
        log_every=args.log_every,
    )
    return predictions, evaluate_predictions(y_true, y_pred)


def train_one_epoch(
        model: PreTrainedModel,
        loader: DataLoader,
        optimizer: AdamW,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        device: torch.device,
        args: argparse.Namespace,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    autocast_dtype = resolve_dtype(args.dtype)
    autocast_enabled = device.type == "cuda" and autocast_dtype in {torch.float16, torch.bfloat16}
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and autocast_dtype == torch.float16)

    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss += float(loss.item()) * args.gradient_accumulation_steps
        if step % args.gradient_accumulation_steps == 0 or step == len(loader):
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    return total_loss / max(1, len(loader))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    per_dataset_out_dir = Path(args.per_dataset_out_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Missing input dir: {input_dir}")

    examples_by_split = read_examples_from_processed_tsv(
        input_dir,
        max_examples_by_split={
            "train": args.max_train_examples,
            "dev": args.max_dev_examples,
            "test": args.max_test_examples,
        },
    )
    if not examples_by_split["train"]:
        raise ValueError("No training examples were loaded.")

    device = torch.device(args.device)
    tokenizer, model = load_model_and_tokenizer(
        args.model_name,
        args.dtype,
        device,
        args.gradient_checkpointing,
    )
    train_loader, _, _ = build_loaders(examples_by_split, tokenizer, args)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_update_steps = max(1, math.ceil(len(train_loader) / args.gradient_accumulation_steps) * args.epochs)
    warmup_steps = int(args.warmup_ratio * total_update_steps)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=max(1, warmup_steps)),
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=max(1, total_update_steps - warmup_steps),
            ),
        ],
        milestones=[max(1, warmup_steps)],
    ) if warmup_steps < total_update_steps else torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=0.0,
        total_iters=max(1, total_update_steps),
    )

    history: list[dict[str, Any]] = []
    best_dev_macro_f1 = -1.0
    best_state_dict: dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, args)
        _, dev_metrics = evaluate_split(model, tokenizer, examples_by_split["dev"], args)
        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_loss,
                "dev_accuracy": dev_metrics["accuracy"],
                "dev_macro_f1": dev_metrics["macro_f1"],
                "dev_weighted_f1": dev_metrics["weighted_f1"],
                "dev_invalid_predictions": dev_metrics["invalid_predictions"],
            }
        )
        print(
            f"Epoch {epoch}/{args.epochs}  "
            f"loss={avg_loss:.4f}  "
            f"dev_acc={dev_metrics['accuracy']:.4f}  "
            f"dev_macro_f1={dev_metrics['macro_f1']:.4f}  "
            f"dev_invalid={dev_metrics['invalid_predictions']}"
        )

        if dev_metrics["macro_f1"] > best_dev_macro_f1:
            best_dev_macro_f1 = dev_metrics["macro_f1"]
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state_dict is not None:
        model.load_state_dict({key: value.to(device) for key, value in best_state_dict.items()})

    _, dev_metrics = evaluate_split(model, tokenizer, examples_by_split["dev"], args)
    test_predictions, test_metrics = evaluate_split(model, tokenizer, examples_by_split["test"], args)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "setup": {
            "model_name": args.model_name,
            "max_length": args.max_length,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "seed": args.seed,
            "dtype": args.dtype,
            "input_dir": str(input_dir),
            "train_examples": len(examples_by_split["train"]),
            "dev_examples": len(examples_by_split["dev"]),
            "test_examples": len(examples_by_split["test"]),
            "label_set": LABEL_SET,
            "training_style": "instruction_sft_causal_lm",
            "chat_template": bool(getattr(tokenizer, "chat_template", None)),
            "system_prompt": (args.system_prompt or "").strip() or None,
            "decoder_features": [
                "language",
                "corpus",
                "framework",
                "direction",
                "rel_type",
            ],
            "data_source": "results/processed_tsv/by_source_file",
        },
        "dev_metrics": dev_metrics,
        "test_metrics": test_metrics,
        "training_history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    write_predictions(output_dir / "test_predictions.tsv", test_predictions)

    if args.save_best and best_state_dict is not None:
        best_dir = output_dir / "best_model"
        best_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(best_dir)
        tokenizer.save_pretrained(best_dir)

    if not args.skip_per_dataset:
        preds_by_corpus: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"gold_label": [], "pred_label": []})
        for row in test_predictions:
            corpus = row.get("corpus") or "unknown"
            preds_by_corpus[corpus]["gold_label"].append(row["gold_label"])
            preds_by_corpus[corpus]["pred_label"].append(row["pred_label"])
        agg = aggregate_predictions_by_corpus(preds_by_corpus)
        per_dataset_payload = {
            "model": "Qwen instruction SFT (chat template when available)",
            "model_name": args.model_name,
            "max_length": args.max_length,
            "label_set": LABEL_SET,
            **agg,
        }
        per_dataset_out_dir.mkdir(parents=True, exist_ok=True)
        (per_dataset_out_dir / "qwen_sft_test.json").write_text(
            json.dumps(per_dataset_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "dev_accuracy": dev_metrics["accuracy"],
                "dev_macro_f1": dev_metrics["macro_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_invalid_predictions": test_metrics["invalid_predictions"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
