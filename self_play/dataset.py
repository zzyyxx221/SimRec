from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from SimRec.simulator import clean_text


REC_SYSTEM_PROMPT = (
    "/no_think\n"
    "You are the recommender in a shopping dialogue.\n"
    "Your goal is to find the single best catalog item for the shopper.\n"
    "Use retrieval evidence before recommending. Do not guess product facts.\n"
    "\n"
    "Valid output modes:\n"
    "1. Tool-call mode\n"
    "- Use this mode when you need search results or item details.\n"
    "- Output exactly one JSON tool call and nothing else.\n"
    "- Do not add explanation, markdown, role labels, or natural language around the tool call.\n"
    '- search format: {"name":"search_products","arguments":{"query":"..."}}\n'
    '- item-details format: {"name":"get_item_details","arguments":{"item_id":"B0XXXXXXXX"}}\n'
    "\n"
    "2. Shopper-facing reply mode\n"
    "- Use this mode only after you have enough retrieved evidence to answer.\n"
    "- Write plain English, including the required field labels below.\n"
    "- Never expose tool names, JSON, code, commands, role tokens, XML tags, or hidden reasoning.\n"
    "- Never end with an unfinished sentence, ellipsis, or placeholder.\n"
    "\n"
    "Recommendation output contract:\n"
    "- Recommend exactly one item unless the shopper explicitly asks for alternatives.\n"
    "- Every recommendation must use exactly this field structure:\n"
    "  recommended_item_id: <ITEM_ID>\n"
    "  recommended_item_description: <2-4 complete English sentences>\n"
    "- The description is the most important field. It must explain why this item matches the shopper's requested attributes.\n"
    "- The description will be reused as a retrieval query for Recall@K expansion. Include discriminative words: genre/category, author/brand when known, tone/style, core attributes, and evidence from search or item details.\n"
    "- Mention only claims supported by retrieved evidence.\n"
    "\n"
    "Decision policy:\n"
    "- At the first shopper request, call search_products with a concise query built from the user's need.\n"
    "- Recommend only when the retrieved title/content clearly supports the shopper's requested product type and attributes.\n"
    "- If the first search results do not clearly match the requested product type, run one revised search with more specific keywords instead of inventing unsupported product facts.\n"
    "- If the shopper gives corrective feedback after a recommendation, run one revised search query that incorporates that feedback, then make the final recommendation.\n"
    "- You may inspect the single most promising item with get_item_details, then you must recommend immediately.\n"
    "- Do not use user preference history in this run.\n"
    "- Ask one short clarification question only when no reasonable search query can be formed.\n"
)


def build_rollout_messages() -> list[dict[str, str]]:
    return [{"role": "system", "content": REC_SYSTEM_PROMPT}]


class SimRecTaskDataset(Dataset):
    def __init__(self, entries, tokenizer, config):
        self.entries = list(entries)
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        row = dict(self.entries[idx])
        extra_info = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
        row["index"] = idx
        row["tools_kwargs"] = {}
        row["interaction_kwargs"] = dict(extra_info.get("interaction_kwargs") or {})
        row["is_validate"] = bool(extra_info.get("is_validate", False))
        row["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)
        return row


class SimRecDataset(Dataset):
    def __init__(self, data_files, tokenizer, processor, config, is_train: bool = True):
        if isinstance(data_files, str):
            data_files = [data_files]
        self.samples = []
        self.entries = []
        validate = not is_train
        for file_path in data_files:
            path = Path(file_path)
            if path.suffix != ".jsonl":
                raise ValueError(f"Unsupported data file format: {file_path}")
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        sample = json.loads(line)
                        reference_query = clean_text(sample.get("reference_query") or sample.get("problem"))
                        reference_review = clean_text(
                            sample.get("reference_review") or sample.get("target_product") or reference_query
                        )
                        target_item_ids = self._normalize_target_item_ids(sample)
                        target_item_id = target_item_ids[0] if target_item_ids else ""
                        if reference_review and target_item_id:
                            sample["reference_query"] = reference_query
                            sample["reference_review"] = reference_review
                            sample["target_item_id"] = target_item_id
                            sample["target_item_ids"] = target_item_ids
                            sample_curriculum = sample.get("curriculum") if isinstance(sample.get("curriculum"), dict) else {}
                            self.samples.append(sample)
                            dialogue: list[dict[str, str]] = []
                            prompt = f"{REC_SYSTEM_PROMPT}\nThe shopper will speak first.\n"
                            self.entries.append(
                                {
                                    "prompt": prompt,
                                    "raw_prompt": build_rollout_messages(),
                                    "agent_name": "simrec_self_play_tool_agent",
                                    "env_context": {"dialogue": dialogue, "initial_role": "rec"},
                                    "data_source": "SimRec.self_play/recommender",
                                    "ability": "recommendation",
                                    "reward_model": {
                                        "style": "rule",
                                        "ground_truth": {
                                            "qid": sample.get("qid"),
                                            "source": sample.get("source", "McAuley-Lab/Amazon-C4"),
                                            "user_id": sample.get("user_id", ""),
                                            "target_item_id": target_item_id,
                                            "target_item_ids": target_item_ids,
                                            "target_ids": target_item_ids,
                                            "reference_review": reference_review,
                                            "reference_query": reference_query,
                                            "curriculum": sample_curriculum,
                                            "initial_state": {"dialogue": dialogue, "current_role": "rec"},
                                        },
                                    },
                                    "extra_info": {
                                        "qid": sample.get("qid"),
                                        "user_id": sample.get("user_id", ""),
                                        "curriculum": sample_curriculum,
                                        "role": "rec",
                                        "is_validate": validate,
                                        "interaction_kwargs": {
                                            "name": "simrec_self_play_user",
                                            "qid": sample.get("qid"),
                                            "user_id": sample.get("user_id", ""),
                                            "target_item_id": target_item_id,
                                            "target_item_ids": target_item_ids,
                                            "target_ids": target_item_ids,
                                            "reference_query": reference_query,
                                            "is_validate": validate,
                                        },
                                    },
                                }
                            )
        self.task_dataset = SimRecTaskDataset(self.entries, tokenizer, config)

    def __len__(self):
        return len(self.task_dataset)

    def __getitem__(self, idx):
        return self.task_dataset[idx]

    @staticmethod
    def _normalize_target_item_ids(sample: dict[str, Any]) -> list[str]:
        raw_targets = sample.get("target_item_ids")
        if raw_targets is None:
            raw_targets = sample.get("target_ids")
        if raw_targets is None:
            raw_targets = [sample.get("target_item_id") or sample.get("target_id")]
        if not isinstance(raw_targets, (list, tuple, set)):
            raw_targets = [raw_targets]

        targets: list[str] = []
        for item in raw_targets:
            item_id = clean_text(item).upper()
            if item_id and item_id not in targets:
                targets.append(item_id)
        return targets
