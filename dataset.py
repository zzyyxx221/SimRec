from __future__ import annotations

import json
import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from SimRec.skill_evolution import category_from_source, infer_category_from_text, normalize_category
from SimRec.simulator import clean_text


DEFAULT_METADATA_PATH = Path(__file__).resolve().parent / "data/indexes/qwen3_embedding_amazon_1M/metadata.pkl"


REC_SYSTEM_PROMPT_WITH_USER_PREFERENCE = (
    "/no_think\n"
    "You are the recommender in a shopping dialogue.\n"
    "Your goal is to find the single best catalog item for the shopper.\n"
    "Use retrieval evidence before recommending. Do not guess product facts.\n"
    "Do not output <think> tags or hidden reasoning. Keep each tool decision concise and reserve enough space for the final structured recommendation.\n"
    "\n"
    "Valid output modes:\n"
    "1. Tool-call mode\n"
    "- Use this mode when you need search results or item details.\n"
    "- Output exactly one JSON tool call and nothing else.\n"
    "- Do not add explanation, markdown, role labels, or natural language around the tool call.\n"
    '- search format: {"name":"search_products","arguments":{"query":"..."}}\n'
    '- item-details format: {"name":"get_item_details","arguments":{"item_id":"B0XXXXXXXX"}}\n'
    '- user-preference format: {"name":"get_user_preference","arguments":{}}\n'
    "\n"
    "2. Shopper-facing reply mode\n"
    "- Use this mode only after you have enough retrieved evidence to answer.\n"
    "- Write plain English, including the required field labels below.\n"
    "- Never expose tool names, JSON, code, commands, role tokens, XML tags, or hidden reasoning.\n"
    "- Never end with an unfinished sentence, ellipsis, or placeholder.\n"
    "\n"
    "3. Clarification/dialogue mode\n"
    "- Use this mode only when the shopper's request lacks a decisive constraint and no reasonable search query can be formed.\n"
    "- Ask exactly one short question and nothing else.\n"
    "- Do not emit a tool call, recommendation fields, JSON, or internal reasoning in this mode.\n"
    "- After the shopper answers, return to tool-call mode and search with the clarified constraint.\n"
    "\n"
    "Recommendation output contract:\n"
    "- Recommend exactly one item unless the shopper explicitly asks for alternatives.\n"
    "- Every recommendation must use exactly this field structure:\n"
    "  recommended_item_id: <ITEM_ID>\n"
    "  recommended_item_description: <2-4 complete English sentences>\n"
    "- The description is the most important field. It must explain why this item matches the shopper's requested attributes.\n"
    "- The description will be reused as a retrieval query for Recall@K expansion. Include discriminative words: genre/category, author/brand when known, tone/style, core attributes, and evidence from search or item details.\n"
    "- Mention only claims supported by retrieved evidence.\n"
    "- Any shopper-facing answer after tool use must still use the recommendation field structure; do not answer follow-up questions as free-form prose.\n"
    "\n"
    "Decision and tool policy:\n"
    "- Build search queries from the current request's product type and discriminative constraints. Keep them concise; use at most 3 searches total. Each search must use a meaningfully different query and should add new evidence.\n"
    "- Use get_item_details selectively to verify plausible candidates and hard constraints; do not inspect the same item twice.\n"
    "- get_user_preference is available once per episode. You may call it at any point before the final recommendation and use its returned profile as ranking context.\n"
    "- After calling it, use the returned preference together with the shopper's explicit request and retrieved evidence.\n"
    "- get_user_preference has no arguments; call it exactly as {\"name\":\"get_user_preference\",\"arguments\":{}} and use the returned preprocessed preference text.\n"
    "- The tool name is exactly get_user_preference; do not write get_usr_preference or any other alias.\n"
    "- User preference history is context only; product facts and final item IDs must still be grounded in search results or item details.\n"
    "- When top search results are similar, or shopper feedback adds specific attributes such as brand, model, size, edition, material, compatibility, or included accessories, inspect one plausible candidate with get_item_details before recommending.\n"
    "- Recommend only when retrieved evidence supports the shopper's requested product type and attributes.\n"
    "- After shopper feedback, do not answer in prose; search with the corrected constraints, inspect useful evidence if needed, then recommend using the required fields.\n"
    "- If results already match the shopper's corrected brand, model, size, format, or other hard attribute, inspect or recommend that item instead of searching again.\n"
    "- Never repeat a previous search query. If no meaningfully different query is possible or the tool budget is nearly exhausted, recommend the best grounded retrieved item.\n"
    "- Ask one short clarification question only when no reasonable search query can be formed.\n"
)


_USER_PREFERENCE_PROMPT_LINES = (
    '- user-preference format: {"name":"get_user_preference","arguments":{}}\n',
    "- get_user_preference is available once per episode. You may call it at any point before the final recommendation and use its returned profile as ranking context.\n",
    "- After calling it, use the returned preference together with the shopper's explicit request and retrieved evidence.\n",
    '- get_user_preference has no arguments; call it exactly as {"name":"get_user_preference","arguments":{}} and use the returned preprocessed preference text.\n',
    "- The tool name is exactly get_user_preference; do not write get_usr_preference or any other alias.\n",
    "- User preference history is context only; product facts and final item IDs must still be grounded in search results or item details.\n",
)


_ITEM_DETAILS_PROMPT_LINES = (
    '- item-details format: {"name":"get_item_details","arguments":{"item_id":"B0XXXXXXXX"}}\n',
    "- Use get_item_details selectively to verify plausible candidates and hard constraints; do not inspect the same item twice.\n",
    "- When top search results are similar, or shopper feedback adds specific attributes such as brand, model, size, edition, material, compatibility, or included accessories, inspect one plausible candidate with get_item_details before recommending.\n",
)

_REQUIRED_USER_PREFERENCE_PROMPT_LINES = (
    "- You must call get_user_preference exactly once in every episode before making the final recommendation. Call it with no arguments and use its returned profile as ranking context. Do not call it again.\n",
)

_REQUIRED_ITEM_DETAILS_PROMPT_LINES = (
    "- After search_products returns candidates, you must call get_item_details exactly once on one plausible candidate before making the final recommendation. Do not call it again.\n",
)


def _without_user_preference_prompt(prompt: str) -> str:
    for line in _USER_PREFERENCE_PROMPT_LINES:
        prompt = prompt.replace(line, "")
    return prompt


def _without_item_details_prompt(prompt: str) -> str:
    for line in _ITEM_DETAILS_PROMPT_LINES:
        prompt = prompt.replace(line, "")
    return prompt


def _require_tool_once_prompts(prompt: str) -> str:
    if _env_flag("SIMREC_REQUIRE_USER_PREFERENCE_ONCE", False):
        prompt = prompt.replace(
            "- get_user_preference is available once per episode. You may call it at any point before the final recommendation and use its returned profile as ranking context.\n",
            _REQUIRED_USER_PREFERENCE_PROMPT_LINES[0],
        )
        prompt = prompt.replace(
            "- After calling it, use the returned preference together with the shopper's explicit request and retrieved evidence.\n",
            "- Call get_user_preference once before searching, then use only category-relevant preferences that are compatible with the current request.\n",
        )

    if _env_flag("SIMREC_REQUIRE_ITEM_DETAILS_ONCE", False):
        prompt = prompt.replace(
            "- Use get_item_details selectively to verify plausible candidates and hard constraints; do not inspect the same item twice.\n",
            _REQUIRED_ITEM_DETAILS_PROMPT_LINES[0],
        )
        prompt = prompt.replace(
            "- When top search results are similar, or shopper feedback adds specific attributes such as brand, model, size, edition, material, compatibility, or included accessories, inspect one plausible candidate with get_item_details before recommending.\n",
            "- Use the single required item-details call to verify one candidate returned by search_products before recommending.\n",
        )

    return prompt


REC_SYSTEM_PROMPT_WITHOUT_USER_PREFERENCE = _without_user_preference_prompt(REC_SYSTEM_PROMPT_WITH_USER_PREFERENCE)
REC_SYSTEM_PROMPT = REC_SYSTEM_PROMPT_WITH_USER_PREFERENCE


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def build_rec_system_prompt() -> str:
    if _env_flag("SIMREC_ENABLE_USER_PREFERENCE_PROMPT", True):
        prompt = REC_SYSTEM_PROMPT_WITH_USER_PREFERENCE
    else:
        prompt = REC_SYSTEM_PROMPT_WITHOUT_USER_PREFERENCE
    if not _env_flag("SIMREC_ENABLE_ITEM_DETAILS_PROMPT", True):
        prompt = _without_item_details_prompt(prompt)
    return _require_tool_once_prompts(prompt)


def build_rollout_messages() -> list[dict[str, str]]:
    return [{"role": "system", "content": build_rec_system_prompt()}]


def _metadata_path() -> Path:
    configured = os.environ.get("SIMREC_METADATA_PKL")
    return Path(configured).expanduser() if configured else DEFAULT_METADATA_PATH


@lru_cache(maxsize=2)
def _load_category_by_item(metadata_path: str) -> dict[str, str]:
    path = Path(metadata_path).expanduser()
    if not path.exists():
        return {}
    with path.open("rb") as f:
        payload = pickle.load(f)
    rows = payload.values() if isinstance(payload, dict) else payload
    category_by_item: dict[str, str] = {}
    if not isinstance(rows, list) and not hasattr(rows, "__iter__"):
        return category_by_item
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw = row.get("raw") if isinstance(row.get("raw"), dict) else {}
        item_id = clean_text(row.get("id") or row.get("item_id") or raw.get("item_id")).upper()
        category = normalize_category(row.get("category") or row.get("main_category") or raw.get("category"))
        if item_id and category:
            category_by_item[item_id] = category
    return category_by_item


def target_category_from_metadata(target_item_id: Any) -> str:
    item_id = clean_text(target_item_id).upper()
    if not item_id:
        return ""
    return _load_category_by_item(str(_metadata_path())).get(item_id, "")


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


class SimRecTaskDataset(Dataset):
    def __init__(self, entries, tokenizer, config):
        self.entries = list(entries)
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        row = dict(self.entries[idx])
        row["index"] = idx
        row["tools_kwargs"] = {}
        row["interaction_kwargs"] = {}
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
                        target_item_ids = _normalize_target_item_ids(sample)
                        target_item_id = target_item_ids[0] if target_item_ids else ""
                        if reference_review and target_item_id:
                            sample["reference_query"] = reference_query
                            sample["reference_review"] = reference_review
                            sample["target_item_id"] = target_item_id
                            sample["target_item_ids"] = target_item_ids
                            target_category = target_category_from_metadata(target_item_id)
                            target_category = target_category or normalize_category(sample.get("target_category"))
                            target_category = target_category or category_from_source(sample.get("source"))
                            if not target_category:
                                target_category, _ = infer_category_from_text(reference_query)
                            sample_curriculum = sample.get("curriculum") if isinstance(sample.get("curriculum"), dict) else {}
                            user_profile = sample.get("user_profile")
                            user_preference = sample.get("user_preference", "")
                            user_profile_status = sample.get("user_profile_status", "")
                            self.samples.append(sample)
                            dialogue: list[dict[str, str]] = []
                            rec_system_prompt = build_rec_system_prompt()
                            prompt = f"{rec_system_prompt}\nThe shopper will speak first.\n"
                            self.entries.append(
                                {
                                    "prompt": prompt,
                                    "raw_prompt": build_rollout_messages(),
                                    "agent_name": "simrec_tool_agent",
                                    "env_context": {"dialogue": dialogue, "initial_role": "rec"},
                                    "data_source": "SimRec/recommender",
                                    "ability": "recommendation",
                                    "reward_model": {
                                        "style": "rule",
                                        "ground_truth": {
                                            "qid": sample.get("qid"),
                                            "source": sample.get("source", "McAuley-Lab/Amazon-C4"),
                                            "user_id": sample.get("user_id", ""),
                                            "target_item_id": target_item_id,
                                            "target_item_ids": target_item_ids,
                                            "target_category": target_category,
                                            "reference_review": reference_review,
                                            "reference_query": reference_query,
                                            "user_profile": user_profile,
                                            "user_preference": user_preference,
                                            "user_profile_status": user_profile_status,
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
                                                "name": "simrec_user",
                                                "qid": sample.get("qid"),
                                                "user_id": sample.get("user_id", ""),
                                                "target_item_id": target_item_id,
                                                "target_item_ids": target_item_ids,
                                                "reference_query": reference_query,
                                                "user_profile": user_profile,
                                                "user_preference": user_preference,
                                                "user_profile_status": user_profile_status,
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
