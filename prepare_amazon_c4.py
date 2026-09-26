from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

from SimRec.simulator import clean_text
from SimRec.user_preference_profile import load_user_histories, render_user_preference


DEFAULT_SOURCE = None
DEFAULT_PROFILE_SOURCES = ""


def _load_rows(source: str, split: str) -> list[dict[str, Any]]:
    if source.endswith(".jsonl"):
        with open(source, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if source.endswith(".csv"):
        with open(source, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install `datasets` or pass a local .csv/.jsonl Amazon-C4 export.") from exc

    ds = load_dataset(source, split=split)
    return [ds[i] for i in range(len(ds))]


def _split_profile_sources(profile_sources: str | None) -> list[str]:
    if not profile_sources:
        return []
    return [source.strip() for source in profile_sources.split(",") if source.strip()]


def _build_example(
    row: dict[str, Any],
    user_histories: dict[str, list[dict[str, Any]]] | None = None,
    profile_history_k: int = 5,
) -> dict[str, Any] | None:
    query = clean_text(row.get("query"))
    review = clean_text(row.get("ori_review"))
    item_id = clean_text(row.get("item_id"))
    if not query or not review or not item_id:
        return None
    user_id = clean_text(row.get("user_id"))
    example = {
        "qid": str(row.get("qid", "")),
        "source": "McAuley-Lab/Amazon-C4",
        "user_id": user_id,
        "target_item_id": item_id,
        "reference_review": review,
        "reference_query": query,
    }
    if user_histories is not None:
        history = [
            item
            for item in user_histories.get(user_id, [])
            if clean_text(item.get("item_id")).upper() != item_id.upper()
        ]
        example["user_profile"] = {"purchase_history": history}
        rendered, found = render_user_preference(
            user_id=user_id,
            history=history,
            history_k=profile_history_k,
            exclude_item_id=item_id,
        )
        example["user_preference"] = rendered
        example["user_profile_status"] = "found" if found else "missing"
    return example


def main(
    source: str = DEFAULT_SOURCE,
    split: str = "train",
    output_dir: str = "data",
    train_ratio: float = 0.95,
    seed: int = 42,
    max_samples: int | None = None,
    profile_sources: str | None = DEFAULT_PROFILE_SOURCES,
    profile_max_history: int = 50,
    profile_history_k: int = 5,
) -> None:
    rows = _load_rows(source, split)
    if max_samples is not None:
        rows = rows[:max_samples]

    profile_source_list = _split_profile_sources(profile_sources)
    user_histories = None
    if profile_source_list:
        user_histories = load_user_histories(profile_source_list, max_history=profile_max_history)
        print(f"Loaded user histories for {len(user_histories)} users from {len(profile_source_list)} profile source(s)")

    random.Random(seed).shuffle(rows)
    examples = [
        item
        for row in rows
        if (
            item := _build_example(
                row,
                user_histories=user_histories,
                profile_history_k=profile_history_k,
            )
        )
        is not None
    ]
    cut = int(len(examples) * train_ratio)
    train_examples = examples[:cut]
    val_examples = examples[cut:]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"

    with train_path.open("w", encoding="utf-8") as f:
        for item in train_examples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with val_path.open("w", encoding="utf-8") as f:
        for item in val_examples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote {len(train_examples)} train rows to {train_path}")
    print(f"Wrote {len(val_examples)} val rows to {val_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Amazon-C4 data for SimRec.")
    parser.add_argument("--source", required=True, help="Path to an Amazon-C4 export or datasets identifier.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--train_ratio", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--profile_sources",
        default=DEFAULT_PROFILE_SOURCES,
        help="Comma-separated Amazon/profile JSONL or JSON files used to precompute each sample's user_profile.",
    )
    parser.add_argument(
        "--profile_max_history",
        type=int,
        default=50,
        help="Maximum history entries to store per user profile; use 0 to keep all available history.",
    )
    parser.add_argument("--profile_history_k", type=int, default=5, help="History entries summarized in user_preference.")
    args = parser.parse_args()
    main(
        source=args.source,
        split=args.split,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        max_samples=args.max_samples,
        profile_sources=args.profile_sources,
        profile_max_history=args.profile_max_history,
        profile_history_k=args.profile_history_k,
    )
