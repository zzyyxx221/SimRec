from __future__ import annotations

import json
import inspect
import math
import os
from typing import Any

from transformers.utils import get_json_schema

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from SimRec.self_play.reward_constants import (
    VALID_ITEM_DETAILS_BONUS,
    VALID_SEARCH_BONUS,
    VALID_USER_PREFERENCE_BONUS,
)
from SimRec.self_play.search_backend import SearchBackend, UserPreferenceBackend
from SimRec.self_play.tool_call_runtime import append_trace, build_metrics, ensure_simrec_self_play_state, format_dialogue

DEFAULT_SEARCH_MODE = os.environ.get("SIMREC_SEARCH_MODE", "hybrid")
DEFAULT_SEARCH_TOP_K = int(os.environ.get("SIMREC_SEARCH_TOP_K", "10"))
DEFAULT_VALIDATION_SEARCH_TOP_K = int(os.environ.get("SIMREC_VALIDATION_SEARCH_TOP_K", "100"))
DEFAULT_SEARCH_DISPLAY_TOP_K = int(os.environ.get("SIMREC_SEARCH_DISPLAY_TOP_K", "10"))
DEFAULT_REWARD_NDCG_MODE = (
    os.environ.get("SIMREC_REWARD_NDCG_MODE")
    or os.environ.get("SIMREC_REWARD_NDCG_MODE")
    or "best_rank"
).strip().lower()


def _ndcg_at_rank(rank: int | None) -> float:
    if rank is None or rank <= 0:
        return 0.0

    return float(1.0 / math.log2(rank + 1))


def _reward_ndcg_mode() -> str:
    mode = (
        os.environ.get("SIMREC_REWARD_NDCG_MODE")
        or os.environ.get("SIMREC_REWARD_NDCG_MODE")
        or DEFAULT_REWARD_NDCG_MODE
    ).strip().lower()
    return mode if mode in {"best_rank", "paper_multi_target"} else "best_rank"


def _normalize_target_item_ids(extra_info: dict[str, Any]) -> list[str]:
    raw_targets = extra_info.get("target_item_ids")
    if raw_targets is None:
        raw_targets = extra_info.get("target_ids")
    if raw_targets is None:
        raw_targets = [extra_info.get("target_item_id")]
    if not isinstance(raw_targets, (list, tuple, set)):
        raw_targets = [raw_targets]

    targets: list[str] = []
    for item in raw_targets:
        item_id = str(item or "").strip().upper()
        if item_id and item_id not in targets:
            targets.append(item_id)
    return targets


def _best_rank(result_ids: list[str], target_item_ids: list[str]) -> int | None:
    ranks = [result_ids.index(item_id) + 1 for item_id in target_item_ids if item_id in result_ids]
    return min(ranks) if ranks else None


def _paper_multi_target_ndcg(result_ids: list[str], target_item_ids: list[str], k: int) -> float:
    targets = [str(item or "").strip().upper() for item in target_item_ids if str(item or "").strip()]
    if not targets:
        return 0.0
    target_set = set(targets)
    dcg = 0.0
    for idx, item_id in enumerate(result_ids[:k], start=1):
        if str(item_id or "").strip().upper() in target_set:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_len = min(k, len(targets))
    ideal_dcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_len + 1))
    return float(dcg / ideal_dcg) if ideal_dcg > 0.0 else 0.0


def _retrieval_ndcg(result_ids: list[str], target_item_ids: list[str], k: int) -> float:
    if _reward_ndcg_mode() == "paper_multi_target":
        return _paper_multi_target_ndcg(result_ids, target_item_ids, k)
    rank = _best_rank(result_ids, target_item_ids)
    return _ndcg_at_rank(rank) if rank is not None and rank <= k else 0.0


def _get_dialogue(agent_data) -> list[dict[str, Any]]:
    messages = getattr(agent_data, "messages", [])
    return format_dialogue(list(messages), max_turns=8)


def _is_validation_rollout(agent_data) -> bool:
    for source in (
        getattr(agent_data, "extra_fields", None),
        getattr(agent_data, "interaction_kwargs", None),
        getattr(agent_data, "tool_kwargs", None),
    ):
        if not isinstance(source, dict):
            continue
        for key in ("validate", "is_validate", "validation"):
            if key in source:
                value = source.get(key)
                if isinstance(value, bool):
                    return value
                return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
        tool_extra_fields = source.get("tool_extra_fields")
        if isinstance(tool_extra_fields, dict) and "validate" in tool_extra_fields:
            value = tool_extra_fields.get("validate")
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _build_tool_schema(func) -> OpenAIFunctionToolSchema:
    schema = get_json_schema(func)
    parameters = schema.setdefault("function", {}).setdefault("parameters", {"type": "object", "properties": {}})
    parameters.setdefault("type", "object")
    parameters.setdefault("properties", {})
    if "required" not in parameters:
        signature = inspect.signature(func)
        required = []
        for name, param in signature.parameters.items():
            if name == "self":
                continue
            if param.default is inspect.Signature.empty:
                required.append(name)
        parameters["required"] = required
    return OpenAIFunctionToolSchema(**schema)


class SimRecSearchTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.backend = SearchBackend()

    def search_products(self, query: str) -> dict[str, Any]:
        """Search the local product catalog for candidate items.

        Args:
            query: Natural-language product need or refined query.
        """

        return {"query": query}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return _build_tool_schema(self.search_products)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs.get("agent_data")
        query = str(parameters.get("query", "")).strip()
        extra_info = getattr(agent_data, "interaction_kwargs", {}) or {}
        target_item_ids = _normalize_target_item_ids(extra_info)
        mode = DEFAULT_SEARCH_MODE
        validate = _is_validation_rollout(agent_data)
        top_k = DEFAULT_VALIDATION_SEARCH_TOP_K if validate else DEFAULT_SEARCH_TOP_K
        results = self.backend.search_records(query=query, top_k=top_k, mode=mode)
        rendered = self.backend.render_search_results(results[:DEFAULT_SEARCH_DISPLAY_TOP_K])
        result_ids = [str(item.get("id", "")).upper() for item in results if item.get("id")]
        target_rank = _best_rank(result_ids, target_item_ids)
        hit_rank_at_10 = target_rank if target_rank is not None and target_rank <= 10 else None
        retrieval_ndcg_at_10 = _retrieval_ndcg(result_ids, target_item_ids, 10)
        retrieval_ndcg_at_100 = _retrieval_ndcg(result_ids, target_item_ids, 100)
        search_hit = hit_rank_at_10 is not None

        state = ensure_simrec_self_play_state(agent_data.extra_fields)
        normalized_query = " ".join(query.lower().split())
        previous_queries = {
            " ".join(str(item or "").lower().split())
            for item in state.get("search_history", [])
            if str(item or "").strip()
        }
        duplicate_query = bool(normalized_query and normalized_query in previous_queries)
        tool_bonus = 0.0 if duplicate_query else VALID_SEARCH_BONUS
        recommendation_count = int(state.get("recommendation_count", 0) or 0)
        state["search_history"].append(query)
        state["search_invoked_count"] = int(state.get("search_invoked_count", 0)) + 1
        if duplicate_query:
            state["duplicate_search_count"] = int(state.get("duplicate_search_count", 0) or 0) + 1
        if recommendation_count > 0:
            state["revised_search_invoked_count"] = int(state.get("revised_search_invoked_count", 0) or 0) + 1
        state["last_search_results"] = results
        state["last_search_result_ids"] = result_ids
        state["last_search_observation"] = rendered
        state["must_recommend_next"] = bool(results)
        state["search_hit_at_k"] = search_hit
        state["retrieval_ndcg_at_10"] = retrieval_ndcg_at_10
        state["retrieval_ndcg_at_100"] = retrieval_ndcg_at_100
        state["reward_ndcg_mode"] = _reward_ndcg_mode()

        reward = tool_bonus
        metrics = build_metrics(
            action_type="search",
            success=False,
            valid_action=True,
            tool_reward=tool_bonus,
            search_invoked=True,
            search_hit_at_k=search_hit,
            retrieval_ndcg_at_10=retrieval_ndcg_at_10,
            retrieval_ndcg_at_100=retrieval_ndcg_at_100,
        )
        metrics["reward_ndcg_mode"] = _reward_ndcg_mode()
        append_trace(
            agent_data.extra_fields,
            source="tool",
            action=json.dumps({"tool_name": self.name, "arguments": parameters}, ensure_ascii=False),
            obs={
                "obs": rendered,
                "dialogue": _get_dialogue(agent_data),
                "result_ids": result_ids,
                "target_rank": target_rank,
                "target_item_ids": target_item_ids,
                "reward_ndcg_mode": _reward_ndcg_mode(),
            },
            reward=reward,
            metrics=metrics,
            done=False,
            valid_action=True,
            raw_action=agent_data.extra_fields.get("recovered_tool_raw_action"),
            action_extraction=agent_data.extra_fields.get("recovered_tool_action_extraction"),
        )
        return ToolResponse(text=rendered), tool_bonus, {"trace_metrics": metrics}


class SimRecItemDetailsTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.backend = SearchBackend()

    def get_item_details(self, item_id: str) -> dict[str, str]:
        """Inspect a specific product from the local catalog.

        Args:
            item_id: Product ID returned by a previous search result.
        """

        return {"item_id": item_id}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return _build_tool_schema(self.get_item_details)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs.get("agent_data")
        item_id = str(parameters.get("item_id", "")).strip().upper()
        rendered, found = self.backend.get_item_details(item_id)

        state = ensure_simrec_self_play_state(agent_data.extra_fields)
        state["details_invoked_count"] = int(state.get("details_invoked_count", 0)) + 1
        state["last_viewed_item_id"] = item_id if found else None
        state["must_recommend_next"] = True

        tool_bonus = VALID_ITEM_DETAILS_BONUS if found else 0.0
        reward = tool_bonus
        metrics = build_metrics(
            action_type="get_item_details",
            success=False,
            valid_action=True,
            tool_reward=tool_bonus,
            item_details_invoked=True,
        )
        append_trace(
            agent_data.extra_fields,
            source="tool",
            action=json.dumps({"tool_name": self.name, "arguments": parameters}, ensure_ascii=False),
            obs={"obs": rendered, "dialogue": _get_dialogue(agent_data)},
            reward=reward,
            metrics=metrics,
            done=False,
            valid_action=True,
            raw_action=agent_data.extra_fields.get("recovered_tool_raw_action"),
            action_extraction=agent_data.extra_fields.get("recovered_tool_action_extraction"),
        )
        return ToolResponse(text=rendered), tool_bonus, {"trace_metrics": metrics}


class SimRecUserPreferenceTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.backend = UserPreferenceBackend()

    def get_user_preference(self, history_k: int = 5) -> dict[str, int]:
        """Summarize the shopper's prior purchase preferences from local history.

        Args:
            history_k: Number of recent history entries to summarize.
        """

        return {"history_k": history_k}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return _build_tool_schema(self.get_user_preference)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs.get("agent_data")
        extra_info = getattr(agent_data, "interaction_kwargs", {}) or {}
        history_k = max(1, int(parameters.get("history_k", 5)))
        rendered, found = self.backend.get_user_preference(
            user_id=str(extra_info.get("user_id", "")),
            history_k=history_k,
            exclude_qid=str(extra_info.get("qid", "")),
        )

        state = ensure_simrec_self_play_state(agent_data.extra_fields)
        state["user_preference_observation"] = rendered

        tool_bonus = VALID_USER_PREFERENCE_BONUS if found else 0.0
        reward = tool_bonus
        metrics = build_metrics(
            action_type="get_user_preference",
            success=False,
            valid_action=True,
            tool_reward=tool_bonus,
            user_preference_invoked=True,
        )
        append_trace(
            agent_data.extra_fields,
            source="tool",
            action=json.dumps({"tool_name": self.name, "arguments": parameters}, ensure_ascii=False),
            obs={"obs": rendered, "dialogue": _get_dialogue(agent_data)},
            reward=reward,
            metrics=metrics,
            done=False,
            valid_action=True,
        )
        return ToolResponse(text=rendered), tool_bonus, {"trace_metrics": metrics}
