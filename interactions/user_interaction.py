from __future__ import annotations

import asyncio
import re
from typing import Any, Optional
from uuid import uuid4

from verl.interactions.base import BaseInteraction

from SimRec.reward_constants import (
    FORMAT_BONUS,
)
from SimRec.simulator import LLMUserSimulator, clean_text, mask_target_item_ids
from SimRec.tool_call_runtime import (
    append_trace,
    build_metrics,
    ensure_simrec_state,
    extract_recommended_item_ids,
    extract_structured_recommendations,
    format_dialogue,
    is_clarification_question,
)

BROKEN_TOOL_CALL_RE = re.compile(r'^\s*\{\s*"name"\s*:\s*"[^"}]*\s*$', re.IGNORECASE)
ROLE_TOKEN_LEAK_RE = re.compile(r"<\|im_(start|end)\|>|<tool_call>|</tool_call>", re.IGNORECASE)
TOOL_COMMAND_LEAK_RE = re.compile(
    r"\b(?:do)?(?:getitemdetails|get_user_preference|get_usr_preference|search_products|getitem|search)\s*\(",
    re.IGNORECASE,
)
TOOL_JSON_LEAK_RE = re.compile(
    r'\{\s*"(?:name|tool_name)"\s*:\s*"(?:search_products|get_item_details|get_user_preference|get_usr_preference)"',
    re.IGNORECASE,
)
INCOMPLETE_REPLY_RE = re.compile(r"(?:\.{3}|\u2026)\s*$")
HIDDEN_REASONING_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalize_target_item_ids(value: Any, fallback: str = "") -> list[str]:
    raw_targets = value if value is not None else [fallback]
    if not isinstance(raw_targets, (list, tuple, set)):
        raw_targets = [raw_targets]
    targets: list[str] = []
    for item in raw_targets:
        item_id = str(item or "").strip().upper()
        if item_id and item_id not in targets:
            targets.append(item_id)
    return targets


class SimRecUserInteraction(BaseInteraction):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._skip_recommendation_feedback = _as_bool(
            config.get("disable_post_recommendation_feedback")
            or config.get("skip_recommendation_feedback")
            or config.get("skip_user_feedback_after_recommendation")
        )
        self._instance_dict: dict[str, dict[str, Any]] = {}
        self._simulator = LLMUserSimulator(
            base_url=config.get("base_url"),
            model=config.get("model"),
            api_key=config.get("api_key"),
            timeout=float(config.get("timeout", 60.0)),
            temperature=float(config.get("temperature", 0.2)),
            max_tokens=int(config.get("max_tokens", 220)),
            enable_thinking=config.get("enable_thinking"),
            seed=int(config.get("seed", 42)),
        )

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        *,
        qid: str = "",
        user_id: str = "",
        target_item_id: str = "",
        reference_review: str = "",
        reference_query: str = "",
        initial_user_utterance: str = "",
        is_validate: bool = False,
        **kwargs,
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        validation_mode = _as_bool(is_validate or kwargs.get("validate") or kwargs.get("validation"))
        evaluation_target_item_id = str(target_item_id or "").upper()
        evaluation_target_item_ids = _normalize_target_item_ids(kwargs.get("target_item_ids"), evaluation_target_item_id)
        # Target IDs are evaluation/controller state, not user-simulator
        # context.  The interaction layer still keeps them below for success
        # and termination checks, while the simulator only sees the shopper
        # need and the dialogue.
        sample = {
            "qid": qid,
            "user_id": user_id,
            "target_item_id": "",
            "target_item_ids": [],
            "reference_review": reference_review,
            "reference_query": reference_query,
        }
        session = self._simulator.start_episode(sample, initial_user_utterance=initial_user_utterance or None)
        opener = mask_target_item_ids(
            await asyncio.to_thread(self._simulator.generate_initial_user_utterance, session),
            evaluation_target_item_ids,
        )
        self._instance_dict[instance_id] = {
            "qid": qid,
            "user_id": user_id,
            "session": session,
            "target_item_id": evaluation_target_item_id,
            "target_item_ids": evaluation_target_item_ids,
            "validation_mode": validation_mode,
            "initial_user_utterance": opener,
            "done": False,
            "success": False,
        }
        return instance_id

    def get_initial_user_utterance(self, instance_id: str) -> str:
        record = self._instance_dict[instance_id]
        return str(record.get("initial_user_utterance") or "")

    async def generate_response(
        self,
        instance_id: str,
        messages: list[dict[str, Any]],
        *,
        extra_fields: dict[str, Any] | None = None,
        **kwargs,
    ) -> tuple[bool, str, float, dict[str, Any]]:
        record = self._instance_dict[instance_id]
        session = record["session"]
        extra_fields = extra_fields or {}
        state = ensure_simrec_state(extra_fields)
        dialogue = format_dialogue(messages, max_turns=8)

        assistant_text = ""
        visible_dialogue = [dict(message) for message in dialogue]
        for index in range(len(visible_dialogue) - 1, -1, -1):
            message = visible_dialogue[index]
            if message.get("role") == "recommender":
                assistant_text = self._visible_assistant_text(message.get("content", ""))
                message["content"] = assistant_text
                break

        if self._is_invalid_assistant_reply(assistant_text):
            state["last_action_type"] = "invalid_reply"
            user_reply = "I still need a concrete suggestion or a short question to help narrow things down."
            reward = 0.0
            metrics = build_metrics(
                action_type="invalid_reply",
                success=False,
                valid_action=False,
            )
            append_trace(
                extra_fields,
                source="interaction",
                action=assistant_text,
                obs={"obs": user_reply, "dialogue": visible_dialogue + [{"role": "user", "content": user_reply}]},
                reward=reward,
                metrics=metrics,
                done=False,
                valid_action=False,
            )
            return False, user_reply, reward, {"tool_interact_info": extra_fields.get("tool_interact_info", [])}

        is_question = is_clarification_question(assistant_text)
        if is_question:
            user_reply = mask_target_item_ids(
                await asyncio.to_thread(self._simulator.generate_user_reply, session, assistant_text, visible_dialogue),
                record.get("target_item_ids"),
            )
            state["clarification_count"] = int(state.get("clarification_count", 0) or 0) + 1
            state["dialogue_action_count"] = int(state.get("dialogue_action_count", 0) or 0) + 1
            state["last_action_type"] = "clarify"
            reward = 0.0
            metrics = build_metrics(
                action_type="clarify",
                success=False,
                valid_action=True,
                dialogue_action=True,
            )
            append_trace(
                extra_fields,
                source="interaction",
                action=assistant_text,
                obs={"obs": user_reply, "dialogue": visible_dialogue + [{"role": "user", "content": user_reply}]},
                reward=reward,
                metrics=metrics,
                done=False,
                valid_action=True,
            )
            return False, user_reply, reward, {"tool_interact_info": extra_fields.get("tool_interact_info", [])}

        structured_recommendations = extract_structured_recommendations(assistant_text)
        if not structured_recommendations:
            user_reply = (
                "Please give the recommendation in the required format with "
                "recommended_item_id: and recommended_item_description:."
            )
            reward = 0.0
            metrics = build_metrics(
                action_type="invalid_reply",
                success=False,
                valid_action=False,
            )
            append_trace(
                extra_fields,
                source="interaction",
                action=assistant_text,
                obs={"obs": user_reply, "dialogue": visible_dialogue + [{"role": "user", "content": user_reply}]},
                reward=reward,
                metrics=metrics,
                done=False,
                valid_action=False,
            )
            return False, user_reply, reward, {"tool_interact_info": extra_fields.get("tool_interact_info", [])}

        candidate_ids = extract_recommended_item_ids(assistant_text)
        recent_ids = {str(item_id).upper() for item_id in state.get("last_search_result_ids", [])}
        if state.get("last_viewed_item_id"):
            recent_ids.add(str(state["last_viewed_item_id"]).upper())
        grounded = bool(candidate_ids and any(item_id in recent_ids for item_id in candidate_ids))

        target_item_ids = record.get("target_item_ids") or session.target_item_ids or [session.target_item_id]
        target_item_ids = _normalize_target_item_ids(target_item_ids)
        success = bool(set(target_item_ids).intersection(candidate_ids))
        recommendation_count = int(state.get("recommendation_count", 0) or 0) + 1
        state["recommendation_count"] = recommendation_count
        state["last_action_type"] = "recommend"
        should_terminate = success or recommendation_count >= 2
        if self._skip_recommendation_feedback:
            # Ablation mode: end the episode after the assistant recommendation
            # without calling the shopper simulator for post-response feedback.
            should_terminate = True
            user_reply = ""
        elif should_terminate:
            user_reply = ""
        else:
            user_reply = mask_target_item_ids(
                await asyncio.to_thread(
                    self._simulator.generate_user_reply,
                    session,
                    assistant_text,
                    visible_dialogue,
                    recommended_item_ids=candidate_ids,
                    grounded=grounded,
                ),
                record.get("target_item_ids"),
            )
        session.accepted = success
        session.done = should_terminate
        record["done"] = should_terminate
        record["success"] = success
        state["recommendation_made"] = True
        state["must_recommend_next"] = False

        # Format compliance is independent of target correctness.  Hit@1 is
        # scored separately below by the reward manager.
        format_reward = FORMAT_BONUS
        reward = format_reward

        metrics = build_metrics(
            action_type="recommend",
            success=success,
            valid_action=True,
            format_reward=format_reward,
            grounded=grounded,
            search_hit_at_k=bool(state.get("search_hit_at_k")),
            retrieval_ndcg_at_10=float(state.get("retrieval_ndcg_at_10", 0.0) or 0.0),
            retrieval_ndcg_at_100=float(state.get("retrieval_ndcg_at_100", 0.0) or 0.0),
            forced_recommendation=bool(extra_fields.get("forced_recommendation")),
        )
        metrics["natural_search_hit_at_k"] = float(state.get("natural_search_hit_at_k", 0.0) or 0.0)
        metrics["target_injected"] = float(state.get("target_injected", 0.0) or 0.0)
        metrics["target_available_in_results"] = float(state.get("target_available_in_results", 0.0) or 0.0)
        append_trace(
            extra_fields,
            source="interaction",
            action=assistant_text,
            obs={
                "obs": user_reply,
                "dialogue": visible_dialogue + ([{"role": "user", "content": user_reply}] if user_reply else []),
            },
            reward=reward,
            metrics=metrics,
            done=should_terminate,
            valid_action=True,
        )
        return should_terminate, user_reply, reward, {"tool_interact_info": extra_fields.get("tool_interact_info", [])}

    async def finalize_interaction(self) -> None:
        self._instance_dict.clear()

    @staticmethod
    def _visible_assistant_text(text: Any) -> str:
        without_thinking = HIDDEN_REASONING_RE.sub("", str(text or ""))
        return clean_text(without_thinking)

    @staticmethod
    def _is_invalid_assistant_reply(text: str) -> bool:
        cleaned = clean_text(text)
        if not cleaned:
            return True

        lowered = cleaned.lower()
        if ROLE_TOKEN_LEAK_RE.search(text):
            return True
        if TOOL_COMMAND_LEAK_RE.search(cleaned):
            return True
        if TOOL_JSON_LEAK_RE.search(cleaned):
            return True
        if BROKEN_TOOL_CALL_RE.match(cleaned):
            return True
        if '"name":' in cleaned and "{" in cleaned and "}" not in cleaned:
            return True
        if lowered.startswith('{"name":') or lowered.startswith("{ 'name':") or lowered.startswith("{name:"):
            return True
        if lowered.endswith('"name":') or lowered.endswith('"arguments":'):
            return True
        if INCOMPLETE_REPLY_RE.search(cleaned):
            return True
        return False
