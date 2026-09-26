from __future__ import annotations

import json
import inspect
import os
import re
from typing import Any

from transformers.utils import get_json_schema

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from SimRec.reward_constants import (
    VALID_ITEM_DETAILS_BONUS,
    VALID_SEARCH_BONUS,
    VALID_USER_PREFERENCE_BONUS,
)
from SimRec.search_backend import SearchBackend, UserPreferenceBackend
from SimRec.skill_evolution import infer_categories_from_results, infer_category_from_text
from SimRec.tool_call_runtime import (
    append_trace,
    build_metrics,
    ensure_simrec_state,
    format_dialogue,
    inject_target_result_randomly,
)
from SimRec.user_preference_profile import render_user_profile_field

DEFAULT_SEARCH_MODE = os.environ.get("SIMREC_SEARCH_MODE", "dense")
DEFAULT_SEARCH_TOP_K = int(os.environ.get("SIMREC_SEARCH_TOP_K", "10"))
DEFAULT_VALIDATION_SEARCH_TOP_K = int(os.environ.get("SIMREC_VALIDATION_SEARCH_TOP_K", "100"))
DEFAULT_SEARCH_DISPLAY_TOP_K = int(os.environ.get("SIMREC_SEARCH_DISPLAY_TOP_K", "10"))
DEFAULT_INJECT_TARGET_ON_MISS = os.environ.get("SIMREC_TRAIN_INJECT_TARGET_ON_MISS", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
DEFAULT_VALIDATION_INJECT_TARGET_ON_MISS = os.environ.get(
    "SIMREC_VALIDATION_INJECT_TARGET_ON_MISS",
    "false",
).strip().lower() in {"1", "true", "yes", "y", "on"}
DEFAULT_MIN_USER_TURNS_BEFORE_TARGET_INJECTION = int(
    os.environ.get("SIMREC_MIN_USER_TURNS_BEFORE_TARGET_INJECTION", "2")
)
DEFAULT_REWARD_NDCG_MODE = os.environ.get("SIMREC_REWARD_NDCG_MODE", "best_rank").strip().lower() or "best_rank"


def _ndcg_at_rank(rank: int | None) -> float:
    if rank is None or rank <= 0:
        return 0.0
    import math

    return float(1.0 / math.log2(rank + 1))


def _reward_ndcg_mode() -> str:
    mode = os.environ.get("SIMREC_REWARD_NDCG_MODE", DEFAULT_REWARD_NDCG_MODE).strip().lower()
    return mode if mode in {"best_rank", "paper_multi_target"} else "best_rank"


def _paper_multi_target_ndcg(result_ids: list[str], target_item_ids: list[str], k: int) -> float:
    import math

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


def _normalize_query(query: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(query or "").lower())
    return " ".join(normalized.split())


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
        if item_id:
            targets.append(item_id)
    return targets


def _best_rank(result_ids: list[str], target_item_ids: list[str]) -> int | None:
    ranks = [result_ids.index(item_id) + 1 for item_id in target_item_ids if item_id in result_ids]
    return min(ranks) if ranks else None


def _get_dialogue(agent_data) -> list[dict[str, Any]]:
    messages = getattr(agent_data, "messages", [])
    return format_dialogue(list(messages), max_turns=8)


def _user_turn_count(agent_data) -> int:
    return sum(1 for message in getattr(agent_data, "messages", []) if message.get("role") == "user")


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
        state = ensure_simrec_state(agent_data.extra_fields)
        normalized_query = _normalize_query(query)
        previous_queries = {
            _normalize_query(str(item or ""))
            for item in state.get("search_history", [])
            if str(item or "").strip()
        }
        duplicate_query = bool(normalized_query and normalized_query in previous_queries)
        if duplicate_query:
            result_ids = [
                str(item or "").upper()
                for item in state.get("last_search_result_ids", [])
                if str(item or "").strip()
            ]
            rendered = "Duplicate search query detected. No new retrieval was run; use the previous search results."
            retrieval_ndcg_at_10 = float(state.get("retrieval_ndcg_at_10", 0.0) or 0.0)
            retrieval_ndcg_at_100 = float(state.get("retrieval_ndcg_at_100", 0.0) or 0.0)
            search_hit = bool(retrieval_ndcg_at_10 > 0.0)
            new_result_ids: list[str] = []
            duplicate_result_set = bool(result_ids)
            tool_bonus = 0.0
            recommendation_count = int(state.get("recommendation_count", 0) or 0)
            state["search_history"].append(query)
            state["search_invoked_count"] = int(state.get("search_invoked_count", 0)) + 1
            state["duplicate_search_count"] = int(state.get("duplicate_search_count", 0) or 0) + 1
            if duplicate_result_set:
                state["duplicate_result_search_count"] = int(state.get("duplicate_result_search_count", 0) or 0) + 1
            if recommendation_count > 0:
                state["revised_search_invoked_count"] = int(state.get("revised_search_invoked_count", 0) or 0) + 1

            metrics = build_metrics(
                action_type="search",
                success=False,
                valid_action=True,
                tool_reward=0.0,
                search_invoked=True,
                search_hit_at_k=search_hit,
                retrieval_ndcg_at_10=retrieval_ndcg_at_10,
                retrieval_ndcg_at_100=retrieval_ndcg_at_100,
            )
            metrics["duplicate_query"] = 1.0
            metrics["duplicate_result_search"] = float(duplicate_result_set)
            metrics["new_result_count"] = 0.0
            metrics["reward_ndcg_mode"] = _reward_ndcg_mode()
            append_trace(
                agent_data.extra_fields,
                source="tool",
                action=json.dumps({"tool_name": self.name, "arguments": parameters}, ensure_ascii=False),
                obs={
                    "obs": rendered,
                    "dialogue": _get_dialogue(agent_data),
                    "result_ids": result_ids,
                    "new_result_ids": new_result_ids,
                    "duplicate_query": True,
                    "duplicate_result_search": duplicate_result_set,
                    "skipped_retrieval": True,
                    "reward_ndcg_mode": _reward_ndcg_mode(),
                },
                reward=0.0,
                metrics=metrics,
                done=False,
                valid_action=True,
                raw_action=agent_data.extra_fields.get("recovered_tool_raw_action"),
                action_extraction=agent_data.extra_fields.get("recovered_tool_action_extraction"),
            )
            return ToolResponse(text=rendered), 0.0, {"trace_metrics": metrics}

        results = self.backend.search_records(query=query, top_k=top_k, mode=mode)
        natural_result_ids = [str(item.get("id", "")).upper() for item in results if item.get("id")]
        target_rank = _best_rank(natural_result_ids, target_item_ids)
        retrieval_ndcg_at_10 = _retrieval_ndcg(natural_result_ids, target_item_ids, 10)
        retrieval_ndcg_at_100 = _retrieval_ndcg(natural_result_ids, target_item_ids, 100)
        natural_search_hit = target_rank is not None and target_rank <= 10

        rendered_results = list(results[:DEFAULT_SEARCH_DISPLAY_TOP_K])
        rendered_result_ids = [str(item.get("id", "")).upper() for item in rendered_results if item.get("id")]
        user_turn_count = _user_turn_count(agent_data)
        target_visible_in_natural_results = bool(set(target_item_ids).intersection(rendered_result_ids))
        search_missed_visible_target = bool(target_item_ids) and not target_visible_in_natural_results
        search_miss_count = int(state.get("search_miss_count", 0) or 0)
        if search_missed_visible_target:
            search_miss_count += 1
        injection_eligible = (
            bool(target_item_ids)
            and user_turn_count >= DEFAULT_MIN_USER_TURNS_BEFORE_TARGET_INJECTION
            and not target_visible_in_natural_results
        )
        inject_target_on_miss = DEFAULT_VALIDATION_INJECT_TARGET_ON_MISS if validate else DEFAULT_INJECT_TARGET_ON_MISS
        target_injected = False
        if (
            inject_target_on_miss
            and injection_eligible
            and not state.get("target_injected")
        ):
            for candidate_target_id in target_item_ids:
                target_record = self.backend.searcher.get_item_record(candidate_target_id)
                if inject_target_result_randomly(
                    rendered_results,
                    candidate_target_id,
                    target_record,
                    fallback_type=mode,
                ):
                    target_injected = True
                    break
        result_ids = [str(item.get("id", "")).upper() for item in rendered_results if item.get("id")]
        rendered = self.backend.render_search_results(rendered_results)
        search_hit = bool(natural_search_hit)
        target_available_in_results = bool(target_visible_in_natural_results or target_injected)

        seen_result_ids = {
            str(item or "").upper()
            for item in state.get("seen_search_result_ids", [])
            if str(item or "").strip()
        }
        new_natural_result_ids = [item_id for item_id in natural_result_ids if item_id not in seen_result_ids]
        duplicate_result_set = bool(natural_result_ids) and not new_natural_result_ids
        tool_bonus = 0.0 if duplicate_query or duplicate_result_set else VALID_SEARCH_BONUS
        recommendation_count = int(state.get("recommendation_count", 0) or 0)
        state["search_history"].append(query)
        state["search_invoked_count"] = int(state.get("search_invoked_count", 0)) + 1
        if duplicate_query:
            state["duplicate_search_count"] = int(state.get("duplicate_search_count", 0) or 0) + 1
        if duplicate_result_set:
            state["duplicate_result_search_count"] = int(state.get("duplicate_result_search_count", 0) or 0) + 1
        if recommendation_count > 0:
            state["revised_search_invoked_count"] = int(state.get("revised_search_invoked_count", 0) or 0) + 1
        state["seen_search_result_ids"] = sorted(seen_result_ids.union(natural_result_ids))
        state["search_miss_count"] = search_miss_count
        state["last_search_results"] = rendered_results
        state["last_search_result_ids"] = result_ids
        state["last_search_observation"] = rendered
        state["must_recommend_next"] = bool(rendered_results)
        state["search_hit_at_k"] = search_hit
        state["natural_search_hit_at_k"] = natural_search_hit
        state["target_available_in_results"] = target_available_in_results
        state["target_injected"] = bool(state.get("target_injected") or target_injected)
        state["retrieval_ndcg_at_10"] = retrieval_ndcg_at_10
        state["retrieval_ndcg_at_100"] = retrieval_ndcg_at_100
        state["reward_ndcg_mode"] = _reward_ndcg_mode()
        ranked_categories = infer_categories_from_results(rendered_results, max_categories=2)
        result_category, result_category_confidence = ranked_categories[0] if ranked_categories else ("", 0.0)
        if not result_category:
            result_category, result_category_confidence = infer_category_from_text(query)
        if result_category and result_category_confidence >= float(state.get("observed_category_confidence", 0.0) or 0.0):
            state["observed_category"] = result_category
            state["observed_categories"] = [category for category, _ in ranked_categories[:2]] or [result_category]
            state["observed_category_confidence"] = result_category_confidence

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
        metrics["duplicate_query"] = float(duplicate_query)
        metrics["duplicate_result_search"] = float(duplicate_result_set)
        metrics["new_result_count"] = float(len(new_natural_result_ids))
        metrics["reward_ndcg_mode"] = _reward_ndcg_mode()
        metrics["target_injected"] = float(target_injected)
        metrics["target_injection_eligible"] = float(injection_eligible)
        metrics["search_miss_count"] = float(search_miss_count)
        metrics["target_injection_after_user_turns"] = float(DEFAULT_MIN_USER_TURNS_BEFORE_TARGET_INJECTION)
        metrics["natural_search_hit_at_k"] = float(natural_search_hit)
        metrics["target_available_in_results"] = float(target_available_in_results)
        append_trace(
            agent_data.extra_fields,
            source="tool",
            action=json.dumps({"tool_name": self.name, "arguments": parameters}, ensure_ascii=False),
            obs={
                "obs": rendered,
                "dialogue": _get_dialogue(agent_data),
                "result_ids": result_ids,
                "natural_result_ids": natural_result_ids,
                "new_result_ids": new_natural_result_ids,
                "duplicate_query": duplicate_query,
                "duplicate_result_search": duplicate_result_set,
                "target_rank": target_rank,
                "target_item_ids": target_item_ids,
                "reward_ndcg_mode": _reward_ndcg_mode(),
                "target_injected": target_injected,
                "target_injection_eligible": injection_eligible,
                "search_miss_count": search_miss_count,
                "target_injection_after_user_turns": DEFAULT_MIN_USER_TURNS_BEFORE_TARGET_INJECTION,
                "user_turn_count": user_turn_count,
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

        state = ensure_simrec_state(agent_data.extra_fields)
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
        self.backend: UserPreferenceBackend | None = None

    def get_user_preference(self) -> dict[str, Any]:
        """Return the shopper's preprocessed user preference profile for this episode."""

        return {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return _build_tool_schema(self.get_user_preference)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs.get("agent_data")
        extra_info = getattr(agent_data, "interaction_kwargs", {}) or {}
        history_k = 5
        user_id = str(extra_info.get("user_id", ""))
        exclude_qid = str(extra_info.get("qid", ""))
        exclude_item_id = str(extra_info.get("target_item_id", ""))
        user_profile = extra_info.get("user_profile")
        user_preference = extra_info.get("user_preference")
        has_sample_profile = isinstance(user_profile, dict) or str(user_profile or "").strip()
        has_sample_preference = str(user_preference or "").strip()
        state = ensure_simrec_state(agent_data.extra_fields)
        if int(state.get("user_preference_invoked_count", 0) or 0) >= 1:
            rendered = "User preference: Already provided earlier in this episode."
            metrics = build_metrics(
                action_type="get_user_preference",
                success=False,
                valid_action=False,
                user_preference_invoked=True,
            )
            append_trace(
                agent_data.extra_fields,
                source="tool",
                action=json.dumps({"tool_name": self.name, "arguments": parameters}, ensure_ascii=False),
                obs={"obs": rendered, "dialogue": _get_dialogue(agent_data)},
                reward=0.0,
                metrics=metrics,
                done=False,
                valid_action=False,
            )
            return ToolResponse(text=rendered), 0.0, {"trace_metrics": metrics}

        if has_sample_profile:
            rendered, found = render_user_profile_field(
                user_id=user_id,
                user_profile=user_profile,
                history_k=history_k,
                exclude_qid=exclude_qid,
                exclude_item_id=exclude_item_id,
            )
        elif has_sample_preference:
            rendered, found = render_user_profile_field(
                user_id=user_id,
                user_profile=str(user_preference),
                history_k=history_k,
                exclude_qid=exclude_qid,
                exclude_item_id=exclude_item_id,
            )
        else:
            if self.backend is None:
                self.backend = UserPreferenceBackend()
            rendered, found = self.backend.get_user_preference(
                user_id=user_id,
                history_k=history_k,
                exclude_qid=exclude_qid,
                exclude_item_id=exclude_item_id,
            )

        state["user_preference_observation"] = rendered
        state["user_preference_invoked_count"] = int(state.get("user_preference_invoked_count", 0) or 0) + 1

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
