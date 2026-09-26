from __future__ import annotations

import fcntl
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

if os.environ.get("SIMREC_LIGHTWEIGHT_REWARD_MANAGER", "").strip().lower() in {"1", "true", "yes"}:
    torch = None
    DataProto = Any

    class RewardManagerBase:  # type: ignore[no-redef]
        def __init__(self, config=None, tokenizer=None, compute_score=None):
            self.config = config
            self.tokenizer = tokenizer
            self.compute_score = compute_score

else:
    import torch
    from verl import DataProto
    from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

from SimRec.self_play.reward_constants import (
    VALID_ITEM_DETAILS_BONUS,
    VALID_SEARCH_BONUS,
    VALID_USER_PREFERENCE_BONUS,
)
from SimRec.self_play.search_backend import SearchBackend
from SimRec.self_play.tool_call_runtime import extract_recommended_item_ids, extract_structured_recommendations, format_dialogue

MIN_REWARD = 0.0
MAX_REWARD = 2.5
FEEDBACK_QUERY_NO_CHANGE_PENALTY = float(os.environ.get("SIMREC_FEEDBACK_QUERY_NO_CHANGE_PENALTY", "0.05"))
LENGTH_PENALTY_FREE_TURNS = int(os.environ.get("SIMREC_LENGTH_PENALTY_FREE_TURNS", "4"))
LENGTH_PENALTY_PER_TURN = float(os.environ.get("SIMREC_LENGTH_PENALTY_PER_TURN", "0.02"))
MAX_LENGTH_PENALTY = float(os.environ.get("SIMREC_MAX_LENGTH_PENALTY", "0.20"))
SELF_PLAY_USER_REWARD_SHARE = float(os.environ.get("SIMREC_SELF_PLAY_USER_REWARD_SHARE", "1.0"))
HIT_AT_1_REWARD = float(os.environ.get("SIMREC_HIT_AT_1_REWARD", "0.8"))
REWARD_PHASE = os.environ.get("SIMREC_REWARD_PHASE", "legacy").strip() or "legacy"
GROUNDED_COMPLETION_BONUS = float(os.environ.get("SIMREC_GROUNDED_COMPLETION_BONUS", "0.0"))
MISSING_RECOMMENDATION_PENALTY = float(os.environ.get("SIMREC_MISSING_RECOMMENDATION_PENALTY", "0.0"))
UNGROUNDED_RECOMMENDATION_PENALTY = float(
    os.environ.get("SIMREC_UNGROUNDED_RECOMMENDATION_PENALTY", "0.0")
)
FORCED_RECOMMENDATION_RETRIEVAL_FACTOR = float(
    os.environ.get("SIMREC_FORCED_RECOMMENDATION_RETRIEVAL_FACTOR", "1.0")
)
MISSING_RECOMMENDATION_RETRIEVAL_FACTOR = float(
    os.environ.get("SIMREC_MISSING_RECOMMENDATION_RETRIEVAL_FACTOR", "1.0")
)
FORCED_RECOMMENDATION_HIT_FACTOR = float(os.environ.get("SIMREC_FORCED_RECOMMENDATION_HIT_FACTOR", "1.0"))
HIDDEN_REASONING_PENALTY = float(os.environ.get("SIMREC_HIDDEN_REASONING_PENALTY", "0.0"))
EFFECTIVE_USER_PREFERENCE_BONUS = float(
    os.environ.get("SIMREC_EFFECTIVE_USER_PREFERENCE_BONUS", "0.0")
)
TOOL_BONUS_PER_ACTION_CAP = {
    "search": VALID_SEARCH_BONUS,
    "get_item_details": VALID_ITEM_DETAILS_BONUS,
    "get_user_preference": VALID_USER_PREFERENCE_BONUS,
}
TOOL_BONUS_PER_TURN_CAP = 0.10
RECALL_EXPANSION_TOP_K = int(os.environ.get("SIMREC_RECALL_EXPANSION_TOP_K", "10"))
RECALL_EXPANSION_MODE = os.environ.get("SIMREC_RECALL_EXPANSION_MODE", "bm25_dense_weighted")
RECALL_EXPANSION_REWARD = float(os.environ.get("SIMREC_RECALL_EXPANSION_REWARD", "0.0"))
RETRIEVAL_NDCG_REWARD_SCALE = float(os.environ.get("SIMREC_RETRIEVAL_NDCG_REWARD_SCALE", "1.0"))
FINAL_RETRIEVAL_TOP_K = int(os.environ.get("SIMREC_FINAL_RETRIEVAL_TOP_K", "100"))
FINAL_RETRIEVAL_MODE = os.environ.get("SIMREC_FINAL_RETRIEVAL_MODE", "bm25_dense_weighted")
REWARD_NDCG_MODE = os.environ.get("SIMREC_REWARD_NDCG_MODE", "best_rank").strip().lower() or "best_rank"
REWARD_EXTRA_INFO_KEYS = (
    "format_reward",
    "tool_reward",
    "retrieval_ndcg_at_10",
    "retrieval_ndcg_at_100",
    "retrieval_ndcg_reward",
    "applied_retrieval_ndcg_reward",
    "retrieval_completion_factor",
    "feedback_query_improvement",
    "feedback_query_improvement_reward",
    "feedback_query_no_change_penalty",
    "length_penalty",
    "num_turns",
    "search_invoked",
    "item_details_invoked",
    "user_preference_invoked",
    "effective_user_preference",
    "user_preference_ndcg_improvement",
    "effective_user_preference_bonus",
    "success",
    "recommend_hit_at_1",
    "hit_at_1_reward",
    "structured_recommendation_present",
    "natural_grounded_recommendation",
    "grounded_completion_bonus",
    "missing_recommendation_penalty",
    "ungrounded_recommendation_penalty",
    "hidden_reasoning_detected",
    "hidden_reasoning_penalty",
    "clarification_count",
    "dialogue_action_count",
)


class SimRecRewardManager(RewardManagerBase):
    name = "simrec_self_play"

    def __init__(self, config=None, tokenizer=None, compute_score=None, num_examine=0, reward_fn_key=None, **kwargs):
        super().__init__(config=config, tokenizer=tokenizer, compute_score=compute_score)
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key or "data_source"
        self.step = 1
        run_id = os.environ.get("VERL_RUN_ID") or f"sim-rec-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
        record_root = Path(kwargs.get("record_dir", Path(__file__).resolve().parent / "verl_step_records"))
        self.record_dir = record_root / run_id
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self._recall_search_backend: SearchBackend | None = None

    @staticmethod
    def _normalize_object(value: object) -> object:
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return value.item()
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    @staticmethod
    def _metric_float(value: object, default: float = 0.0) -> float:
        if value is None:
            return default
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (bool, int, float)):
            out = float(value)
            return out if np.isfinite(out) else default
        return default

    @staticmethod
    def _target_item_ids(ground_truth: dict[str, object]) -> list[str]:
        raw_targets = ground_truth.get("target_item_ids")
        if raw_targets is None:
            raw_targets = ground_truth.get("target_ids")
        if raw_targets is None:
            raw_targets = [ground_truth.get("target_item_id")]
        if not isinstance(raw_targets, (list, tuple, set)):
            raw_targets = [raw_targets]

        targets: list[str] = []
        for item in raw_targets:
            item_id = str(item or "").strip().upper()
            if item_id:
                targets.append(item_id)
        return targets

    @staticmethod
    def _best_rank(result_ids: list[str], target_item_ids: list[str]) -> int | None:
        ranks = [result_ids.index(item_id) + 1 for item_id in target_item_ids if item_id in result_ids]
        return min(ranks) if ranks else None

    @staticmethod
    def _reward_ndcg_mode() -> str:
        mode = os.environ.get("SIMREC_REWARD_NDCG_MODE", REWARD_NDCG_MODE).strip().lower()
        return mode if mode in {"best_rank", "paper_multi_target"} else "best_rank"

    @classmethod
    def _paper_multi_target_ndcg(cls, result_ids: list[str], target_item_ids: list[str], k: int) -> float:
        targets = [str(item or "").strip().upper() for item in target_item_ids if str(item or "").strip()]
        if not targets:
            return 0.0
        target_set = set(targets)
        dcg = 0.0
        for idx, item_id in enumerate(result_ids[:k], start=1):
            if str(item_id or "").strip().upper() in target_set:
                dcg += 1.0 / np.log2(idx + 1)
        ideal_len = min(k, len(targets))
        ideal_dcg = sum(1.0 / np.log2(idx + 1) for idx in range(1, ideal_len + 1))
        return float(dcg / ideal_dcg) if ideal_dcg > 0.0 else 0.0

    @classmethod
    def _retrieval_ndcg(cls, result_ids: list[str], target_item_ids: list[str], k: int) -> float:
        if cls._reward_ndcg_mode() == "paper_multi_target":
            return cls._paper_multi_target_ndcg(result_ids, target_item_ids, k)
        rank = cls._best_rank(result_ids, target_item_ids)
        return cls._ndcg_at_rank(rank) if rank is not None and rank <= k else 0.0

    @classmethod
    def _reward_extra_info(cls, metrics_summary: dict[str, object]) -> dict[str, float]:
        return {f"simrec_self_play_{key}": cls._metric_float(metrics_summary.get(key)) for key in REWARD_EXTRA_INFO_KEYS}

    def _extract_step_info(self, item) -> tuple[int | None, bool]:
        step = self._normalize_object(item.non_tensor_batch.get("step"))
        validate = self._normalize_object(item.non_tensor_batch.get("validate", False))
        if step is None:
            tool_extra_fields = self._normalize_object(item.non_tensor_batch.get("tool_extra_fields"))
            if isinstance(tool_extra_fields, dict):
                step = self._normalize_object(tool_extra_fields.get("step"))
                validate = self._normalize_object(
                    tool_extra_fields.get("validate", tool_extra_fields.get("is_validate", validate))
                )
        if not validate:
            is_validate = self._normalize_object(item.non_tensor_batch.get("is_validate", False))
            validate = is_validate
        try:
            step = int(step) if step is not None else None
        except (TypeError, ValueError):
            step = None
        return step, bool(validate)

    def _extract_trajectory_ids(self, item) -> tuple[int | None, int | None]:
        sample_index = self._normalize_object(item.non_tensor_batch.get("sample_index"))
        rollout_n = self._normalize_object(item.non_tensor_batch.get("rollout_n"))
        if sample_index is None or rollout_n is None:
            tool_extra_fields = self._normalize_object(item.non_tensor_batch.get("tool_extra_fields"))
            if isinstance(tool_extra_fields, dict):
                if sample_index is None:
                    sample_index = self._normalize_object(tool_extra_fields.get("sample_index"))
                if rollout_n is None:
                    rollout_n = self._normalize_object(tool_extra_fields.get("rollout_n"))
        return sample_index, rollout_n

    def _extract_num_turns(self, item) -> float:
        num_turns = self._normalize_object(item.non_tensor_batch.get("__num_turns__"))
        if num_turns is None:
            tool_extra_fields = self._normalize_object(item.non_tensor_batch.get("tool_extra_fields"))
            if isinstance(tool_extra_fields, dict):
                num_turns = self._normalize_object(tool_extra_fields.get("__num_turns__"))
        if isinstance(num_turns, (list, tuple)) and len(num_turns) == 1:
            num_turns = num_turns[0]
        try:
            value = float(num_turns)
        except (TypeError, ValueError):
            return 0.0
        return value if np.isfinite(value) else 0.0

    @staticmethod
    def _length_penalty(num_turns: float) -> float:
        excess_turns = max(0.0, float(num_turns) - float(LENGTH_PENALTY_FREE_TURNS))
        penalty = excess_turns * LENGTH_PENALTY_PER_TURN
        return float(min(MAX_LENGTH_PENALTY, penalty))

    def _extract_tool_interact_info(self, item) -> list[dict]:
        tool_interact_info = self._normalize_object(item.non_tensor_batch.get("tool_interact_info"))
        if tool_interact_info is None:
            tool_extra_fields = self._normalize_object(item.non_tensor_batch.get("tool_extra_fields"))
            if isinstance(tool_extra_fields, dict):
                tool_interact_info = self._normalize_object(tool_extra_fields.get("tool_interact_info"))
        if tool_interact_info is None:
            return []
        if isinstance(tool_interact_info, tuple):
            tool_interact_info = list(tool_interact_info)
        return tool_interact_info if isinstance(tool_interact_info, list) else []

    def _extract_turn_token_spans(self, item) -> list[dict]:
        turn_token_spans = self._normalize_object(item.non_tensor_batch.get("turn_token_spans"))
        if turn_token_spans is None:
            tool_extra_fields = self._normalize_object(item.non_tensor_batch.get("tool_extra_fields"))
            if isinstance(tool_extra_fields, dict):
                turn_token_spans = self._normalize_object(tool_extra_fields.get("turn_token_spans"))
        if turn_token_spans is None:
            return []
        if isinstance(turn_token_spans, tuple):
            turn_token_spans = list(turn_token_spans)
        return turn_token_spans if isinstance(turn_token_spans, list) else []

    def _extract_skill_usage(self, item) -> list[dict]:
        skill_usage = self._normalize_object(item.non_tensor_batch.get("skill_usage"))
        if skill_usage is None:
            tool_extra_fields = self._normalize_object(item.non_tensor_batch.get("tool_extra_fields"))
            if isinstance(tool_extra_fields, dict):
                skill_usage = self._normalize_object(tool_extra_fields.get("skill_usage"))
        if skill_usage is None:
            return []
        if isinstance(skill_usage, tuple):
            skill_usage = list(skill_usage)
        return skill_usage if isinstance(skill_usage, list) else []

    def _extract_extra_field(self, item, key: str, default=None):
        value = self._normalize_object(item.non_tensor_batch.get(key))
        if value is not None:
            return value
        tool_extra_fields = self._normalize_object(item.non_tensor_batch.get("tool_extra_fields"))
        if isinstance(tool_extra_fields, dict):
            value = self._normalize_object(tool_extra_fields.get(key))
        return default if value is None else value

    @staticmethod
    def _parse_obs(obs: object) -> dict:
        if isinstance(obs, dict):
            return obs
        if isinstance(obs, str) and obs.strip().startswith("{"):
            try:
                return json.loads(obs)
            except json.JSONDecodeError:
                return {}
        return {}

    def _recover_visible_recommendation(self, item) -> tuple[list[str], list[dict[str, str]]]:
        """Recover a recommendation when an old rollout dropped interaction traces."""
        candidates: list[str] = []
        final_dialogue = self._extract_extra_field(item, "final_dialogue")
        if isinstance(final_dialogue, list):
            for message in reversed(final_dialogue):
                if not isinstance(message, dict):
                    continue
                if str(message.get("role") or "").lower() not in {"assistant", "recommender", "rec"}:
                    continue
                candidates.append(str(message.get("content") or ""))

        # ``raw_response`` is the decoded response sequence and is available
        # even for records produced before final_dialogue/tool-trace export.
        if not candidates:
            try:
                decoded = self._decode_response(item)
            except Exception:
                decoded = None
            if decoded:
                candidates.append(decoded)

        for text in candidates:
            structured = extract_structured_recommendations(text)
            item_ids = extract_recommended_item_ids(text)
            if item_ids:
                return item_ids, structured
        return [], []

    def _save_records(self, records: list[dict], *, step: int | None = None, validate: bool = False) -> None:
        step_id = step if step is not None and step >= 0 else self.step
        is_val = validate or self.num_examine == 1
        filename = f"{self.name}-step-val-{step_id}.json" if is_val else f"{self.name}-step-{step_id}.json"
        path = self.record_dir / filename
        with path.open("a+", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0)
                content = f.read().strip()
                existing = json.loads(content) if content else []
                existing.extend(records)
                f.seek(0)
                f.truncate()
                json.dump(existing, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        if step is None or step_id == self.step:
            self.step = max(self.step, step_id + 1)

    def _decode_response(self, item) -> str | None:
        response_ids = item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = int(item.batch["attention_mask"][-response_length:].sum().item())
        valid_response_ids = response_ids[:valid_response_length]
        if self.tokenizer is None:
            return None
        return self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)

    def _build_record(
        self,
        item,
        *,
        response_text: str | None,
        user_feedback_text: str | None,
        total_reward: float,
        metrics_summary: dict[str, float],
        parsed_trace: list[dict],
        dialogue_snapshot: list[dict] | None,
        turn_token_spans: list[dict] | None,
        turn_rewards: dict[int, float] | None,
        recommended_item_ids: list[str] | None,
        recommended_items: list[dict[str, str]] | None,
        expanded_recall: dict | None,
        final_retrieval: dict | None,
        search_trajectory: list[dict] | None,
        feedback_query_summary: dict | None,
        skill_usage: list[dict] | None,
    ) -> dict:
        ground_truth = item.non_tensor_batch.get("reward_model", {}).get("ground_truth", {})
        step, validate = self._extract_step_info(item)
        sample_index, rollout_n = self._extract_trajectory_ids(item)
        record = {
            "step": step,
            "validate": validate,
            "sample_index": sample_index,
            "rollout_n": rollout_n,
            "data_source": item.non_tensor_batch.get(self.reward_fn_key),
            "source": ground_truth.get("source"),
            "qid": ground_truth.get("qid"),
            "target_item_id": ground_truth.get("target_item_id"),
            "target_item_ids": self._target_item_ids(ground_truth),
            "reward_ndcg_mode": self._reward_ndcg_mode(),
            "target_category": ground_truth.get("target_category"),
            "reference_query": ground_truth.get("reference_query"),
            "raw_response": response_text,
            "user_feedback_after_response": user_feedback_text,
            "dialogue": dialogue_snapshot,
            "turn_token_spans": turn_token_spans,
            "turn_rewards": turn_rewards,
            "recommended_item_ids": recommended_item_ids,
            "recommended_items": recommended_items,
            "recommend_hit_at_1": metrics_summary["recommend_hit_at_1"],
            "hit_at_1_reward": metrics_summary["hit_at_1_reward"],
            "expanded_recall": expanded_recall,
            "final_retrieval": final_retrieval,
            "search_trajectory": search_trajectory,
            "feedback_query_summary": feedback_query_summary,
            "skill_usage": skill_usage,
            "reward": total_reward,
            "success": metrics_summary["success"],
            "format_reward": metrics_summary["format_reward"],
            "tool_reward": metrics_summary["tool_reward"],
            "result_reward": metrics_summary["result_reward"],
            "expanded_recall_reward": metrics_summary["expanded_recall_reward"],
            "turn_penalty": metrics_summary["turn_penalty"],
            "valid_actions": metrics_summary["valid_actions"],
            "search_invoked": metrics_summary["search_invoked"],
            "item_details_invoked": metrics_summary["item_details_invoked"],
            "user_preference_invoked": metrics_summary["user_preference_invoked"],
            "effective_user_preference": metrics_summary["effective_user_preference"],
            "user_preference_ndcg_improvement": metrics_summary["user_preference_ndcg_improvement"],
            "effective_user_preference_bonus": metrics_summary["effective_user_preference_bonus"],
            "grounded_recommendation": metrics_summary["grounded_recommendation"],
            "search_hit_at_k": metrics_summary["search_hit_at_k"],
            "retrieval_ndcg_at_10": metrics_summary["retrieval_ndcg_at_10"],
            "retrieval_ndcg_reward": metrics_summary["retrieval_ndcg_reward"],
            "applied_retrieval_ndcg_reward": metrics_summary["applied_retrieval_ndcg_reward"],
            "retrieval_completion_factor": metrics_summary["retrieval_completion_factor"],
            "feedback_query_improvement": metrics_summary["feedback_query_improvement"],
            "feedback_query_improvement_reward": metrics_summary["feedback_query_improvement_reward"],
            "feedback_query_no_change_penalty": metrics_summary["feedback_query_no_change_penalty"],
            "length_penalty": metrics_summary["length_penalty"],
            "num_turns": metrics_summary["num_turns"],
            "forced_recommendation": metrics_summary["forced_recommendation"],
            "structured_recommendation_present": metrics_summary["structured_recommendation_present"],
            "natural_grounded_recommendation": metrics_summary["natural_grounded_recommendation"],
            "grounded_completion_bonus": metrics_summary["grounded_completion_bonus"],
            "missing_recommendation_penalty": metrics_summary["missing_recommendation_penalty"],
            "ungrounded_recommendation_penalty": metrics_summary["ungrounded_recommendation_penalty"],
            "hidden_reasoning_detected": metrics_summary["hidden_reasoning_detected"],
            "hidden_reasoning_penalty": metrics_summary["hidden_reasoning_penalty"],
            "clarification_count": metrics_summary["clarification_count"],
            "dialogue_action_count": metrics_summary["dialogue_action_count"],
            "action_type_counts": metrics_summary["action_type_counts"],
            "assistant_action_type_counts": metrics_summary["assistant_action_type_counts"],
            "reward_phase": metrics_summary["reward_phase"],
            "expanded_recall_at_k": metrics_summary["expanded_recall_at_k"],
            "tool_trace": parsed_trace,
        }
        if validate:
            record["final_retrieval_ndcg_at_100"] = metrics_summary["final_retrieval_ndcg_at_100"]
            record["final_retrieval_hit_at_100"] = metrics_summary["final_retrieval_hit_at_100"]
        return record

    def _get_recall_search_backend(self) -> SearchBackend:
        if self._recall_search_backend is None:
            self._recall_search_backend = SearchBackend()
        return self._recall_search_backend

    def _expand_recall_from_descriptions(
        self, recommended_items: list[dict[str, str]], target_item_id: str | None
    ) -> dict:
        target_item_ids = [str(target_item_id or "").upper()] if target_item_id else []
        expanded_ids: list[str] = []
        queries: list[dict[str, object]] = []

        if not recommended_items:
            return {
                "mode": RECALL_EXPANSION_MODE,
                "top_k": RECALL_EXPANSION_TOP_K,
                "queries": queries,
                "item_ids": expanded_ids,
                "recall_at_k": 0.0,
            }

        try:
            backend = self._get_recall_search_backend()
            for item in recommended_items:
                query = str(item.get("description") or "").strip()
                if not query:
                    continue
                results = backend.search_records(query=query, top_k=RECALL_EXPANSION_TOP_K, mode=RECALL_EXPANSION_MODE)
                result_ids = [str(row.get("id", "")).upper() for row in results if row.get("id")]
                queries.append({"recommended_item_id": item.get("item_id"), "query": query, "result_ids": result_ids})
                for item_id in result_ids:
                    if item_id and item_id not in expanded_ids:
                        expanded_ids.append(item_id)
        except Exception as exc:
            return {
                "mode": RECALL_EXPANSION_MODE,
                "top_k": RECALL_EXPANSION_TOP_K,
                "queries": queries,
                "item_ids": expanded_ids,
                "error": str(exc),
                "recall_at_k": 0.0,
            }

        return {
            "mode": RECALL_EXPANSION_MODE,
            "top_k": RECALL_EXPANSION_TOP_K,
            "queries": queries,
            "item_ids": expanded_ids,
            "recall_at_k": float(bool(set(target_item_ids).intersection(expanded_ids[:RECALL_EXPANSION_TOP_K]))),
        }

    @staticmethod
    def _parse_tool_action(action: object) -> dict:
        if not isinstance(action, str) or not action.strip().startswith("{"):
            return {}
        try:
            payload = json.loads(action)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _ndcg_at_rank(rank: int | None) -> float:
        if rank is None or rank <= 0:
            return 0.0
        return float(1.0 / np.log2(rank + 1))

    def _evaluate_final_retrieval_query(
        self, query: str | None, target_item_id: str | None, target_item_ids: list[str] | None = None
    ) -> dict:
        query = str(query or "").strip()
        target_item_ids = target_item_ids or ([str(target_item_id or "").upper()] if target_item_id else [])
        if not query:
            return {
                "query": "",
                "mode": FINAL_RETRIEVAL_MODE,
                "top_k": FINAL_RETRIEVAL_TOP_K,
                "result_ids": [],
                "target_rank": None,
                "hit_at_100": 0.0,
                "ndcg_at_100": 0.0,
            }
        try:
            backend = self._get_recall_search_backend()
            results = backend.search_records(query=query, top_k=FINAL_RETRIEVAL_TOP_K, mode=FINAL_RETRIEVAL_MODE)
            result_ids = [str(row.get("id", "")).upper() for row in results if row.get("id")]
            rank = self._best_rank(result_ids, target_item_ids)
            return {
                "query": query,
                "mode": FINAL_RETRIEVAL_MODE,
                "top_k": FINAL_RETRIEVAL_TOP_K,
                "result_ids": result_ids,
                "target_rank": rank,
                "hit_at_100": float(rank is not None),
                "ndcg_at_100": self._retrieval_ndcg(result_ids, target_item_ids, 100),
                "reward_ndcg_mode": self._reward_ndcg_mode(),
            }
        except Exception as exc:
            return {
                "query": query,
                "mode": FINAL_RETRIEVAL_MODE,
                "top_k": FINAL_RETRIEVAL_TOP_K,
                "result_ids": [],
                "target_rank": None,
                "hit_at_100": 0.0,
                "ndcg_at_100": 0.0,
                "error": str(exc),
            }

    @staticmethod
    def _normalize_query(query: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", " ", str(query or "").lower())
        return " ".join(normalized.split())

    @staticmethod
    def _normalize_item_ids(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        normalized = []
        for item in value:
            item_id = str(item or "").strip().upper()
            if item_id:
                normalized.append(item_id)
        return normalized

    def _score_feedback_query_improvement(self, search_trajectory: list[dict]) -> tuple[float, float, dict | None]:
        pre_feedback = [item for item in search_trajectory if int(item.get("feedback_count", 0) or 0) <= 0]
        post_feedback = [item for item in search_trajectory if int(item.get("feedback_count", 0) or 0) > 0]
        if not pre_feedback or not post_feedback:
            return 0.0, 0.0, None

        baseline = max(pre_feedback, key=lambda item: float(item.get("ndcg_at_10", 0.0) or 0.0))
        revised = max(post_feedback, key=lambda item: float(item.get("ndcg_at_10", 0.0) or 0.0))
        baseline_ndcg = float(baseline.get("ndcg_at_10", 0.0) or 0.0)
        revised_ndcg = float(revised.get("ndcg_at_10", 0.0) or 0.0)
        improvement = max(0.0, revised_ndcg - baseline_ndcg)

        baseline_query = self._normalize_query(str(baseline.get("query") or ""))
        revised_query = self._normalize_query(str(revised.get("query") or ""))
        no_change_penalty = FEEDBACK_QUERY_NO_CHANGE_PENALTY if revised_query and revised_query == baseline_query else 0.0
        reward = 0.0
        summary = {
            "baseline_query": baseline.get("query"),
            "baseline_ndcg_at_10": baseline_ndcg,
            "revised_query": revised.get("query"),
            "revised_ndcg_at_10": revised_ndcg,
            "improvement": improvement,
            "no_change_penalty": no_change_penalty,
            "reward": reward,
        }
        return improvement, reward, summary

    def _score_item(self, item) -> tuple[float, dict[str, float], dict]:
        tool_interact_info = self._extract_tool_interact_info(item)
        _, validate_rollout = self._extract_step_info(item)
        turn_token_spans = self._extract_turn_token_spans(item)

        total_reward = 0.0
        success = 0.0
        format_reward = 0.0
        tool_reward = 0.0
        result_reward = 0.0
        expanded_recall_reward = 0.0
        turn_penalty = 0.0
        valid_actions = 0.0
        search_invoked = 0.0
        item_details_invoked = 0.0
        user_preference_invoked = 0.0
        effective_user_preference = 0.0
        user_preference_ndcg_improvement = 0.0
        effective_user_preference_bonus = 0.0
        grounded_recommendation = 0.0
        search_hit_at_k = 0.0
        natural_search_hit_at_k = 0.0
        target_injected = 0.0
        target_available_in_results = 0.0
        retrieval_ndcg_at_10 = 0.0
        retrieval_ndcg_at_100 = 0.0
        retrieval_ndcg_reward = 0.0
        feedback_query_improvement = 0.0
        feedback_query_improvement_reward = 0.0
        feedback_query_no_change_penalty = 0.0
        length_penalty = 0.0
        hit_at_1_reward = 0.0
        num_turns = self._extract_num_turns(item)
        forced_recommendation = 0.0
        expanded_recall_at_k = 0.0
        parsed_trace = []
        turn_rewards: dict[int, float] = {}
        pending_turn_reward = 0.0
        seen_tool_bonus_actions: set[str] = set()
        format_reward_by_turn: dict[int, float] = {}
        unspanned_format_reward = 0.0
        dialogue_snapshot = None
        last_interaction_feedback = None
        last_assistant_action = None
        recommended_item_ids: list[str] = []
        recommended_items: list[dict[str, str]] = []
        final_search_query = ""
        feedback_count = 0
        search_trajectory: list[dict] = []
        feedback_query_summary: dict | None = None
        seen_search_queries: set[str] = set()
        seen_search_result_ids: set[str] = set()
        best_unique_retrieval_reward = 0.0
        best_unique_retrieval_ndcg = 0.0
        user_preference_retrieval_baseline: float | None = None
        hidden_reasoning_detected = float(bool(self._extract_extra_field(item, "hidden_reasoning_detected", False)))
        clarification_count = 0.0
        dialogue_action_count = 0.0
        action_type_counts: defaultdict[str, int] = defaultdict(int)
        assistant_action_type_counts: defaultdict[str, int] = defaultdict(int)

        for interaction in tool_interact_info:
            if not isinstance(interaction, dict):
                continue
            interaction_reward = float(interaction.get("reward", 0.0) or 0.0)
            turn_span_index = interaction.get("turn_span_index")
            metrics = interaction.get("metrics", {}) or {}
            source = interaction.get("source", "unknown")
            action_type = str(metrics.get("action_type", "") or "")
            if action_type:
                action_type_counts[action_type] += 1
                if source == "tool":
                    assistant_action_type_counts["tool_call"] += 1
                elif action_type == "clarify":
                    assistant_action_type_counts["clarify"] += 1
                elif action_type == "recommend":
                    assistant_action_type_counts["recommend"] += 1
            raw_obs = self._parse_obs(interaction.get("obs"))
            # Do not pop from the trace object itself.  The same trace is the
            # source for both reward computation and the persisted audit
            # record, so mutating it here can erase the dialogue snapshot.
            parsed_obs = dict(raw_obs)
            trace_dialogue = parsed_obs.pop("dialogue", None)
            if isinstance(trace_dialogue, list):
                dialogue_snapshot = format_dialogue(trace_dialogue, max_turns=8)
            action = interaction.get("action")
            if isinstance(action, str) and action.strip():
                last_assistant_action = action

            if source == "tool":
                payload = self._parse_tool_action(action)
                arguments = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
                if action_type == "search" and arguments.get("query"):
                    final_search_query = str(arguments.get("query") or "").strip()
                    normalized_query = self._normalize_query(final_search_query)
                    result_ids = self._normalize_item_ids(parsed_obs.get("result_ids"))
                    new_result_ids = [item_id for item_id in result_ids if item_id not in seen_search_result_ids]
                    duplicate_query = bool(normalized_query and normalized_query in seen_search_queries)
                    duplicate_result_search = bool(result_ids) and not new_result_ids
                    unique_search = not duplicate_query and not duplicate_result_search
                    search_ndcg = float(metrics.get("retrieval_ndcg_at_100", 0.0) or 0.0)
                    candidate_search_reward = RETRIEVAL_NDCG_REWARD_SCALE * search_ndcg if unique_search else 0.0
                    search_reward = max(0.0, candidate_search_reward - best_unique_retrieval_reward)
                    if candidate_search_reward > best_unique_retrieval_reward:
                        best_unique_retrieval_reward = candidate_search_reward
                    if unique_search:
                        best_unique_retrieval_ndcg = max(best_unique_retrieval_ndcg, search_ndcg)
                    if (
                        unique_search
                        and user_preference_retrieval_baseline is not None
                        and search_ndcg > user_preference_retrieval_baseline
                    ):
                        user_preference_ndcg_improvement = max(
                            user_preference_ndcg_improvement,
                            search_ndcg - user_preference_retrieval_baseline,
                        )
                        if not effective_user_preference:
                            effective_user_preference = 1.0
                            if EFFECTIVE_USER_PREFERENCE_BONUS > 0.0:
                                effective_user_preference_bonus = EFFECTIVE_USER_PREFERENCE_BONUS
                                total_reward += effective_user_preference_bonus
                                pending_turn_reward += effective_user_preference_bonus
                    trajectory_entry = {
                        "query": final_search_query,
                        "feedback_count": feedback_count,
                        "ndcg_at_10": search_ndcg,
                        "hit_at_10": float(metrics.get("search_hit_at_k", 0.0) or 0.0),
                        "natural_hit_at_10": float(metrics.get("natural_search_hit_at_k", 0.0) or 0.0),
                        "target_injected": float(metrics.get("target_injected", 0.0) or 0.0),
                        "target_available_in_results": float(metrics.get("target_available_in_results", 0.0) or 0.0),
                        "duplicate_query": duplicate_query,
                        "duplicate_result_search": duplicate_result_search,
                        "new_result_count": len(new_result_ids),
                    }
                    trajectory_entry["ndcg_at_100"] = float(metrics.get("retrieval_ndcg_at_100", 0.0) or 0.0)
                    search_trajectory.append(trajectory_entry)
                    if normalized_query:
                        seen_search_queries.add(normalized_query)
                    seen_search_result_ids.update(result_ids)
                    retrieval_ndcg_reward = best_unique_retrieval_reward
                    if search_reward:
                        total_reward += search_reward
                        pending_turn_reward += search_reward
                if (
                    action_type == "get_user_preference"
                    and float(metrics.get("valid_action", 0.0) or 0.0) > 0.0
                    and user_preference_retrieval_baseline is None
                ):
                    user_preference_retrieval_baseline = best_unique_retrieval_ndcg
                capped_bonus = float(metrics.get("tool_reward", 0.0) or 0.0)
                capped_bonus = min(capped_bonus, TOOL_BONUS_PER_ACTION_CAP.get(action_type, capped_bonus))
                if action_type and action_type not in seen_tool_bonus_actions and capped_bonus > 0.0:
                    seen_tool_bonus_actions.add(action_type)
                    tool_reward += capped_bonus
                    total_reward += capped_bonus
                    pending_turn_reward += capped_bonus
                valid_actions += float(metrics.get("valid_action", 0.0) or 0.0)
                search_invoked = max(search_invoked, float(metrics.get("search_invoked", 0.0) or 0.0))
                item_details_invoked = max(
                    item_details_invoked, float(metrics.get("item_details_invoked", 0.0) or 0.0)
                )
                user_preference_invoked = max(
                    user_preference_invoked, float(metrics.get("user_preference_invoked", 0.0) or 0.0)
                )
                search_hit_at_k = max(search_hit_at_k, float(metrics.get("search_hit_at_k", 0.0) or 0.0))
                natural_search_hit_at_k = max(
                    natural_search_hit_at_k,
                    float(metrics.get("natural_search_hit_at_k", 0.0) or 0.0),
                )
                target_injected = max(target_injected, float(metrics.get("target_injected", 0.0) or 0.0))
                target_available_in_results = max(
                    target_available_in_results,
                    float(metrics.get("target_available_in_results", 0.0) or 0.0),
                )
                retrieval_ndcg_at_10 = max(
                    retrieval_ndcg_at_10, float(metrics.get("retrieval_ndcg_at_10", 0.0) or 0.0)
                )
                retrieval_ndcg_at_100 = max(
                    retrieval_ndcg_at_100, float(metrics.get("retrieval_ndcg_at_100", 0.0) or 0.0)
                )
                parsed_trace.append(
                    {
                        "turn_span_index": interaction.get("turn_span_index"),
                        "action": action,
                        "raw_action": interaction.get("raw_action"),
                        "action_extraction": interaction.get("action_extraction"),
                        "reward": interaction.get("reward"),
                        "done": interaction.get("done"),
                        "valid_action": interaction.get("valid_action"),
                        "obs": parsed_obs,
                    }
                )
                continue

            interaction_format_reward = float(metrics.get("format_reward", 0.0) or 0.0)
            applied_format_reward = interaction_format_reward
            if source == "interaction" and isinstance(turn_span_index, (int, np.integer)):
                span_index = int(turn_span_index)
                if span_index in format_reward_by_turn:
                    applied_format_reward = 0.0
                else:
                    format_reward_by_turn[span_index] = interaction_format_reward
                turn_reward = applied_format_reward + pending_turn_reward
                total_reward += applied_format_reward
                if turn_reward:
                    turn_rewards[span_index] = turn_rewards.get(span_index, 0.0) + turn_reward
                pending_turn_reward = 0.0
            else:
                if source == "interaction":
                    if unspanned_format_reward > 0.0:
                        applied_format_reward = 0.0
                    else:
                        unspanned_format_reward = interaction_format_reward
                total_reward += applied_format_reward
                pending_turn_reward = 0.0
            success = max(success, float(metrics.get("success", 0.0) or 0.0))
            valid_actions += float(metrics.get("valid_action", 0.0) or 0.0)
            format_reward += applied_format_reward
            turn_penalty += float(metrics.get("turn_penalty", 0.0) or 0.0)
            search_invoked = max(search_invoked, float(metrics.get("search_invoked", 0.0) or 0.0))
            item_details_invoked = max(item_details_invoked, float(metrics.get("item_details_invoked", 0.0) or 0.0))
            user_preference_invoked = max(
                user_preference_invoked, float(metrics.get("user_preference_invoked", 0.0) or 0.0)
            )
            grounded_recommendation = max(
                grounded_recommendation, float(metrics.get("grounded_recommendation", 0.0) or 0.0)
            )
            search_hit_at_k = max(search_hit_at_k, float(metrics.get("search_hit_at_k", 0.0) or 0.0))
            natural_search_hit_at_k = max(
                natural_search_hit_at_k,
                float(metrics.get("natural_search_hit_at_k", 0.0) or 0.0),
            )
            target_injected = max(target_injected, float(metrics.get("target_injected", 0.0) or 0.0))
            target_available_in_results = max(
                target_available_in_results,
                float(metrics.get("target_available_in_results", 0.0) or 0.0),
            )
            retrieval_ndcg_at_10 = max(
                retrieval_ndcg_at_10, float(metrics.get("retrieval_ndcg_at_10", 0.0) or 0.0)
            )
            retrieval_ndcg_at_100 = max(
                retrieval_ndcg_at_100, float(metrics.get("retrieval_ndcg_at_100", 0.0) or 0.0)
            )
            forced_recommendation = max(
                forced_recommendation, float(metrics.get("forced_recommendation", 0.0) or 0.0)
            )
            if source == "interaction":
                if action_type == "clarify":
                    clarification_count += 1.0
                    dialogue_action_count += 1.0
                if parsed_obs.get("obs"):
                    last_interaction_feedback = parsed_obs.get("obs")
                    if action_type == "recommend":
                        feedback_count += 1
                if isinstance(action, str) and action.strip():
                    recommended_item_ids = extract_recommended_item_ids(action)
                    recommended_items = extract_structured_recommendations(action)

        # Prefer the final conversation state published by the agent loop.
        # This is written after the self-play/API user reply is appended and
        # therefore is not affected by trace-copy timing.
        final_dialogue = self._extract_extra_field(item, "final_dialogue")
        if isinstance(final_dialogue, list):
            dialogue_snapshot = format_dialogue(final_dialogue, max_turns=8)

        final_feedback = self._extract_extra_field(item, "last_user_feedback")
        if final_feedback is not None and str(final_feedback).strip():
            last_interaction_feedback = str(final_feedback).strip()

        # Keep compatibility with records produced before final_dialogue was
        # added: recover the latest feedback directly from the interaction
        # trace when it is available.
        if not last_interaction_feedback:
            for interaction in reversed(tool_interact_info):
                if not isinstance(interaction, dict) or interaction.get("source") != "interaction":
                    continue
                parsed_obs = self._parse_obs(interaction.get("obs"))
                feedback = parsed_obs.get("obs")
                if feedback is not None and str(feedback).strip():
                    last_interaction_feedback = str(feedback).strip()
                    break

        # Older Self-Play validation records may contain the recommendation in
        # raw_response but lose the interaction trace during export.  Recover
        # it before computing Hit@1 and recommendation rewards.
        if not recommended_item_ids:
            recovered_ids, recovered_items = self._recover_visible_recommendation(item)
            if recovered_ids:
                recommended_item_ids = recovered_ids
                recommended_items = recovered_items
                target_ids = set(
                    self._target_item_ids(
                        item.non_tensor_batch.get("reward_model", {}).get("ground_truth", {})
                    )
                )
                if target_ids and any(item_id in target_ids for item_id in recovered_ids):
                    success = max(success, 1.0)

        failure_penalty = 0.0

        ground_truth = item.non_tensor_batch.get("reward_model", {}).get("ground_truth", {})
        target_item_ids = self._target_item_ids(ground_truth)
        recommend_hit_at_1 = float(bool(recommended_item_ids and recommended_item_ids[0] in target_item_ids))
        structured_recommendation_present = float(bool(recommended_item_ids and recommended_items))
        natural_grounded_recommendation = float(
            bool(structured_recommendation_present and grounded_recommendation and not forced_recommendation)
        )
        hit_reward_factor = FORCED_RECOMMENDATION_HIT_FACTOR if forced_recommendation else 1.0
        hit_at_1_reward = HIT_AT_1_REWARD * recommend_hit_at_1 * hit_reward_factor
        if hit_at_1_reward:
            total_reward += hit_at_1_reward
            if turn_rewards:
                last_turn_index = max(turn_rewards)
                turn_rewards[last_turn_index] = turn_rewards.get(last_turn_index, 0.0) + hit_at_1_reward
            else:
                turn_rewards[0] = turn_rewards.get(0, 0.0) + hit_at_1_reward
        feedback_query_improvement, feedback_query_improvement_reward, feedback_query_summary = (
            self._score_feedback_query_improvement(search_trajectory)
        )
        feedback_query_no_change_penalty = (
            float(feedback_query_summary.get("no_change_penalty", 0.0) or 0.0) if feedback_query_summary else 0.0
        )
        if pending_turn_reward:
            turn_rewards[0] = turn_rewards.get(0, 0.0) + pending_turn_reward

        if natural_grounded_recommendation:
            retrieval_completion_factor = 1.0
            grounded_completion_bonus = GROUNDED_COMPLETION_BONUS
            missing_recommendation_penalty = 0.0
            ungrounded_recommendation_penalty = 0.0
        elif forced_recommendation and grounded_recommendation:
            retrieval_completion_factor = FORCED_RECOMMENDATION_RETRIEVAL_FACTOR
            grounded_completion_bonus = 0.0
            missing_recommendation_penalty = MISSING_RECOMMENDATION_PENALTY if search_invoked else 0.0
            ungrounded_recommendation_penalty = 0.0
        else:
            retrieval_completion_factor = MISSING_RECOMMENDATION_RETRIEVAL_FACTOR
            grounded_completion_bonus = 0.0
            missing_recommendation_penalty = (
                MISSING_RECOMMENDATION_PENALTY if search_invoked and not structured_recommendation_present else 0.0
            )
            ungrounded_recommendation_penalty = (
                UNGROUNDED_RECOMMENDATION_PENALTY
                if search_invoked and structured_recommendation_present and not grounded_recommendation
                else 0.0
            )

        retrieval_completion_factor = float(np.clip(retrieval_completion_factor, 0.0, 1.0))
        retrieval_discount = retrieval_ndcg_reward * (1.0 - retrieval_completion_factor)
        applied_retrieval_ndcg_reward = retrieval_ndcg_reward - retrieval_discount
        hidden_reasoning_penalty = HIDDEN_REASONING_PENALTY if hidden_reasoning_detected else 0.0
        if grounded_completion_bonus:
            total_reward += grounded_completion_bonus
            last_turn_index = max(turn_rewards) if turn_rewards else 0
            turn_rewards[last_turn_index] = turn_rewards.get(last_turn_index, 0.0) + grounded_completion_bonus

        length_penalty = self._length_penalty(num_turns)
        total_penalty = (
            feedback_query_no_change_penalty
            + length_penalty
            + retrieval_discount
            + missing_recommendation_penalty
            + ungrounded_recommendation_penalty
            + hidden_reasoning_penalty
        )
        reward_before_penalty = total_reward
        if total_penalty:
            total_reward -= total_penalty
        final_retrieval = search_trajectory[-1] if search_trajectory else None
        final_retrieval_ndcg_at_100 = retrieval_ndcg_at_100
        final_retrieval_hit_at_100 = float(retrieval_ndcg_at_100 > 0.0)
        expanded_recall = None
        expanded_recall_at_k = 0.0
        expanded_recall_reward = 0.0
        total_reward = float(np.clip(total_reward, MIN_REWARD, MAX_REWARD))
        failure_penalty = max(0.0, reward_before_penalty - total_reward)

        # Self-play is an SimRec.self_play-only credit-assignment policy. Keep the
        # scalar reward computation identical to SimRec, then optionally
        # copy the episode reward to user-role spans.
        if SELF_PLAY_USER_REWARD_SHARE > 0.0:
            user_span_indices = [
                index
                for index, span in enumerate(turn_token_spans)
                if isinstance(span, dict)
                and span.get("role") == "user"
                and span.get("response_mask", 0) == 1
            ]
            if user_span_indices:
                shared_reward = float(total_reward) * SELF_PLAY_USER_REWARD_SHARE
                for span_index in user_span_indices:
                    turn_rewards[span_index] = turn_rewards.get(span_index, 0.0) + shared_reward

        metrics_summary = {
            "success": success,
            "format_reward": format_reward,
            "tool_reward": tool_reward,
            "result_reward": result_reward,
            "expanded_recall_reward": expanded_recall_reward,
            "turn_penalty": turn_penalty,
            "valid_actions": valid_actions,
            "search_invoked": search_invoked,
            "item_details_invoked": item_details_invoked,
            "user_preference_invoked": user_preference_invoked,
            "effective_user_preference": effective_user_preference,
            "user_preference_ndcg_improvement": user_preference_ndcg_improvement,
            "effective_user_preference_bonus": effective_user_preference_bonus,
            "grounded_recommendation": grounded_recommendation,
            "search_hit_at_k": search_hit_at_k,
            "natural_search_hit_at_k": natural_search_hit_at_k,
            "target_injected": target_injected,
            "target_available_in_results": target_available_in_results,
            "retrieval_ndcg_at_10": retrieval_ndcg_at_10,
            "retrieval_ndcg_at_100": retrieval_ndcg_at_100,
            "retrieval_ndcg_reward": retrieval_ndcg_reward,
            "applied_retrieval_ndcg_reward": applied_retrieval_ndcg_reward,
            "retrieval_completion_factor": retrieval_completion_factor,
            "feedback_query_improvement": feedback_query_improvement,
            "feedback_query_improvement_reward": feedback_query_improvement_reward,
            "feedback_query_no_change_penalty": feedback_query_no_change_penalty,
            "length_penalty": length_penalty,
            "num_turns": num_turns,
            "forced_recommendation": forced_recommendation,
            "structured_recommendation_present": structured_recommendation_present,
            "natural_grounded_recommendation": natural_grounded_recommendation,
            "grounded_completion_bonus": grounded_completion_bonus,
            "missing_recommendation_penalty": missing_recommendation_penalty,
            "ungrounded_recommendation_penalty": ungrounded_recommendation_penalty,
            "hidden_reasoning_detected": hidden_reasoning_detected,
            "hidden_reasoning_penalty": hidden_reasoning_penalty,
            "clarification_count": clarification_count,
            "dialogue_action_count": dialogue_action_count,
            "action_type_counts": dict(action_type_counts),
            "assistant_action_type_counts": dict(assistant_action_type_counts),
            "reward_phase": REWARD_PHASE,
            "recommend_hit_at_1": recommend_hit_at_1,
            "hit_at_1_reward": hit_at_1_reward,
            "final_retrieval_ndcg_at_100": final_retrieval_ndcg_at_100,
            "final_retrieval_hit_at_100": final_retrieval_hit_at_100,
            "expanded_recall_at_k": expanded_recall_at_k,
            "reward": total_reward,
        }
        return total_reward, metrics_summary, {
            "tool_trace": parsed_trace,
            "dialogue": dialogue_snapshot,
            "user_feedback_after_response": last_interaction_feedback,
            "assistant_response": last_assistant_action,
            "recommended_item_ids": recommended_item_ids,
            "recommended_items": recommended_items,
            "expanded_recall": expanded_recall,
            "final_retrieval": final_retrieval,
            "search_trajectory": search_trajectory,
            "feedback_query_summary": feedback_query_summary,
            "turn_rewards": turn_rewards,
            "failure_penalty": failure_penalty,
        }

    @staticmethod
    def _apply_turn_level_rewards(
        reward_row: torch.Tensor,
        *,
        turn_token_spans: list[dict],
        turn_rewards: dict[int, float],
        failure_penalty: float,
        total_reward: float,
        valid_resp_len: int,
    ) -> bool:
        applied = False
        last_reward_position: int | None = None
        response_len = int(reward_row.shape[-1])
        for raw_index, span in enumerate(turn_token_spans):
            if not isinstance(span, dict):
                continue
            reward = float(turn_rewards.get(raw_index, 0.0))
            start = span.get("start")
            end = span.get("end")
            try:
                start_i = int(start)
                end_i = int(end)
            except (TypeError, ValueError):
                continue
            if reward == 0.0 or end_i <= start_i:
                continue
            position = min(end_i, valid_resp_len, response_len) - 1
            if position < 0:
                continue
            reward_row[position] += reward
            last_reward_position = position if last_reward_position is None else max(last_reward_position, position)
            applied = True

        if applied and failure_penalty:
            position = last_reward_position if last_reward_position is not None else max(min(valid_resp_len, response_len) - 1, 0)
            reward_row[position] -= float(failure_penalty)
        elif not applied and valid_resp_len > 0:
            position = min(valid_resp_len, response_len) - 1
            reward_row[position] = total_reward
            applied = True
        return applied

    async def run_single(self, data: DataProto):
        item = data[0]
        reward_score, metrics_summary, trace_payload = self._score_item(item)
        response_text = self._decode_response(item)
        step, validate = self._extract_step_info(item)
        self._save_records(
            [
                self._build_record(
                    item,
                    response_text=response_text,
                    user_feedback_text=trace_payload.get("user_feedback_after_response"),
                    total_reward=reward_score,
                    metrics_summary=metrics_summary,
                    parsed_trace=trace_payload["tool_trace"],
                    dialogue_snapshot=trace_payload.get("dialogue"),
                    turn_token_spans=self._extract_turn_token_spans(item),
                    turn_rewards=trace_payload.get("turn_rewards"),
                    recommended_item_ids=trace_payload.get("recommended_item_ids"),
                    recommended_items=trace_payload.get("recommended_items"),
                    expanded_recall=trace_payload.get("expanded_recall"),
                    final_retrieval=trace_payload.get("final_retrieval"),
                    search_trajectory=trace_payload.get("search_trajectory"),
                    feedback_query_summary=trace_payload.get("feedback_query_summary"),
                    skill_usage=self._extract_skill_usage(item),
                )
            ],
            step=step,
            validate=validate,
        )
        # reward_extra_info is later stacked into numpy arrays inside verl.
        # Keep only scalar metrics here; variable-length traces are persisted
        # through step records above and would break np.array aggregation.
        reward_extra_info = self._reward_extra_info(metrics_summary)
        return {"reward_score": reward_score, "reward_extra_info": reward_extra_info}

    def __call__(self, data: DataProto, return_dict: bool = False):
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        records = []

        for i in range(len(data)):
            item = data[i]
            prompt_len = item.batch["prompts"].shape[-1]
            valid_resp_len = int(item.batch["attention_mask"][prompt_len:].sum().item())
            total_reward, metrics_summary, trace_payload = self._score_item(item)
            parsed_trace = trace_payload["tool_trace"]

            if valid_resp_len > 0:
                turn_token_spans = self._extract_turn_token_spans(item)
                self._apply_turn_level_rewards(
                    reward_tensor[i],
                    turn_token_spans=turn_token_spans,
                    turn_rewards=trace_payload.get("turn_rewards", {}),
                    failure_penalty=float(trace_payload.get("failure_penalty", 0.0) or 0.0),
                    total_reward=total_reward,
                    valid_resp_len=valid_resp_len,
                )

            for key, value in self._reward_extra_info(metrics_summary).items():
                reward_extra_info[key].append(value)

            records.append(
                self._build_record(
                    item,
                    response_text=self._decode_response(item),
                    user_feedback_text=trace_payload.get("user_feedback_after_response"),
                    total_reward=total_reward,
                    metrics_summary=metrics_summary,
                    parsed_trace=parsed_trace,
                    dialogue_snapshot=trace_payload.get("dialogue"),
                    turn_token_spans=self._extract_turn_token_spans(item),
                    turn_rewards=trace_payload.get("turn_rewards"),
                    recommended_item_ids=trace_payload.get("recommended_item_ids"),
                    recommended_items=trace_payload.get("recommended_items"),
                    expanded_recall=trace_payload.get("expanded_recall"),
                    final_retrieval=trace_payload.get("final_retrieval"),
                    search_trajectory=trace_payload.get("search_trajectory"),
                    feedback_query_summary=trace_payload.get("feedback_query_summary"),
                    skill_usage=self._extract_skill_usage(item),
                )
            )

        step, validate = self._extract_step_info(data[0])
        self._save_records(records, step=step, validate=validate)
        for key, values in list(reward_extra_info.items()):
            reward_extra_info[key] = [float(np.mean(values))] * len(data)
        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
