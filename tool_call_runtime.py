from __future__ import annotations

import random
import re
from typing import Any


STRUCTURED_RECOMMEND_ID_RE = re.compile(
    r"(?:\u63a8\u8350\u5546\u54c1id|recommended_item_id)\s*[:\uff1a]\s*([A-Z0-9]{2,20})",
    re.IGNORECASE,
)
STRUCTURED_RECOMMEND_DESCRIPTION_RE = re.compile(
    r"(?:\u63a8\u8350\u5546\u54c1description|recommended_item_description)\s*[:\uff1a]\s*",
    re.IGNORECASE,
)
QUESTION_END_RE = re.compile(r"[?\uff1f]\s*[\"'\u2019\u201d)\]]*$")

TOOL_ACTION_TYPES = frozenset({"search", "get_item_details", "get_user_preference"})
DIALOGUE_ACTION_TYPES = frozenset({"clarify"})
ASSISTANT_ACTION_TYPES = frozenset({"clarify", "recommend"})


def inject_target_result_randomly(
    rendered_results: list[dict[str, Any]],
    target_item_id: str,
    target_record: dict[str, Any] | None,
    *,
    fallback_type: str = "dense",
) -> bool:
    """Insert a target with the same visible schema as a normal result."""

    if not target_record or not target_item_id:
        return False
    target_id = str(target_item_id).strip().upper()
    if any(str(item.get("id", "")).strip().upper() == target_id for item in rendered_results):
        return False

    content = str(
        target_record.get("contents")
        or target_record.get("content")
        or target_record.get("metadata")
        or target_record.get("title")
        or target_record.get("name")
        or ""
    )
    result_type = str((rendered_results[0].get("type") if rendered_results else None) or fallback_type)
    scores = [float(item.get("score", 0.0) or 0.0) for item in rendered_results]
    insert_at = random.SystemRandom().randrange(len(rendered_results) + 1)
    if not scores:
        score = 0.0
    elif insert_at == 0:
        score = scores[0]
    elif insert_at >= len(scores):
        score = scores[-1]
    else:
        score = (scores[insert_at - 1] + scores[insert_at]) / 2.0

    rendered_results.insert(
        insert_at,
        {"id": target_id, "content": content, "score": score, "type": result_type},
    )
    return True


def ensure_simrec_state(extra_fields: dict[str, Any]) -> dict[str, Any]:
    state = extra_fields.get("simrec_state")
    if not isinstance(state, dict):
        state = {
            "search_history": [],
            "last_search_results": [],
            "last_search_result_ids": [],
            "seen_search_result_ids": [],
            "last_search_observation": None,
            "last_viewed_item_id": None,
            "user_preference_observation": None,
            "search_invoked_count": 0,
            "duplicate_search_count": 0,
            "duplicate_result_search_count": 0,
            "search_miss_count": 0,
            "revised_search_invoked_count": 0,
            "details_invoked_count": 0,
            "user_preference_invoked_count": 0,
            "recommendation_count": 0,
            "recommendation_made": False,
            "must_recommend_next": False,
            "search_hit_at_k": False,
            "retrieval_ndcg_at_10": 0.0,
            "retrieval_ndcg_at_100": 0.0,
            "observed_category": "",
            "observed_category_confidence": 0.0,
        }
        extra_fields["simrec_state"] = state
    else:
        # Keep rollouts resumed from older checkpoints compatible with the
        # current action bookkeeping.
        state.setdefault("search_history", [])
        state.setdefault("last_search_results", [])
        state.setdefault("last_search_result_ids", [])
        state.setdefault("seen_search_result_ids", [])
        state.setdefault("last_search_observation", None)
        state.setdefault("last_viewed_item_id", None)
        state.setdefault("user_preference_observation", None)
        state.setdefault("search_invoked_count", 0)
        state.setdefault("duplicate_search_count", 0)
        state.setdefault("duplicate_result_search_count", 0)
        state.setdefault("search_miss_count", 0)
        state.setdefault("revised_search_invoked_count", 0)
        state.setdefault("details_invoked_count", 0)
        state.setdefault("user_preference_invoked_count", 0)
        state.setdefault("recommendation_count", 0)
        state.setdefault("recommendation_made", False)
        state.setdefault("must_recommend_next", False)
        state.setdefault("search_hit_at_k", False)
        state.setdefault("retrieval_ndcg_at_10", 0.0)
        state.setdefault("retrieval_ndcg_at_100", 0.0)
        state.setdefault("observed_category", "")
        state.setdefault("observed_category_confidence", 0.0)
    state.setdefault("clarification_count", 0)
    state.setdefault("dialogue_action_count", 0)
    state.setdefault("last_action_type", None)
    return state


def ensure_tool_trace(extra_fields: dict[str, Any]) -> list[dict[str, Any]]:
    trace = extra_fields.get("tool_interact_info")
    if not isinstance(trace, list):
        trace = []
        extra_fields["tool_interact_info"] = trace
    return trace


def extract_structured_recommendations(text: str) -> list[dict[str, str]]:
    content = str(text or "")
    id_matches = list(STRUCTURED_RECOMMEND_ID_RE.finditer(content))
    if not id_matches:
        return []

    recommendations: list[dict[str, str]] = []
    for index, id_match in enumerate(id_matches):
        item_id = id_match.group(1).strip().upper()
        segment_start = id_match.end()
        segment_end = id_matches[index + 1].start() if index + 1 < len(id_matches) else len(content)
        segment = content[segment_start:segment_end]

        description = ""
        description_match = STRUCTURED_RECOMMEND_DESCRIPTION_RE.search(segment)
        if description_match:
            description = segment[description_match.end() :].strip()
        if item_id and description:
            recommendations.append({"item_id": item_id, "description": description})
    return recommendations


def extract_recommended_item_ids(text: str) -> list[str]:
    return [entry["item_id"] for entry in extract_structured_recommendations(text)]


def is_clarification_question(text: str) -> bool:
    """Return whether text is a shopper-facing clarification question."""

    content = str(text or "").strip()
    if not content or extract_structured_recommendations(content):
        return False
    return bool(QUESTION_END_RE.search(content))


def classify_assistant_action(text: str) -> str:
    """Classify a visible assistant reply in the native SimRec protocol."""

    if extract_structured_recommendations(text):
        return "recommend"
    if is_clarification_question(text):
        return "clarify"
    return "invalid_reply"


def format_dialogue(messages: list[dict[str, Any]], *, max_turns: int = 8) -> list[dict[str, str]]:
    """Normalize dialogue to user/recommender only.

    Export only user-visible dialogue turns. Internal tool calls and tool
    observations stay in tool_trace and should not appear as recommender replies.
    """

    normalized: list[dict[str, str]] = []

    def _content_to_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type == "text":
                        text = str(item.get("text", "")).strip()
                        if text:
                            chunks.append(text)
                    else:
                        chunks.append(f"<{item_type}>")
                else:
                    text = str(item).strip()
                    if text:
                        chunks.append(text)
            return "\n".join(chunk for chunk in chunks if chunk)
        return str(content).strip()

    for message in messages:
        role = message.get("role")
        text = _content_to_text(message.get("content"))
        if role == "user":
            if text:
                normalized.append({"role": "user", "content": text})
            continue

        if role in {"assistant", "recommender", "rec"}:
            if not normalized or normalized[-1]["role"] != "recommender":
                normalized.append({"role": "recommender", "content": ""})
            if text:
                if normalized[-1]["content"]:
                    normalized[-1]["content"] += "\n" + text
                else:
                    normalized[-1]["content"] = text

    normalized = [turn for turn in normalized if turn.get("content")]
    return normalized[-max_turns:]


def build_metrics(
    *,
    action_type: str,
    success: bool,
    valid_action: bool,
    format_reward: float = 0.0,
    tool_reward: float = 0.0,
    result_reward: float = 0.0,
    turn_penalty: float = 0.0,
    grounded: bool = False,
    search_invoked: bool = False,
    item_details_invoked: bool = False,
    user_preference_invoked: bool = False,
    search_hit_at_k: bool = False,
    retrieval_ndcg_at_10: float = 0.0,
    retrieval_ndcg_at_100: float = 0.0,
    forced_recommendation: bool = False,
    dialogue_action: bool = False,
) -> dict[str, Any]:
    is_dialogue_action = bool(dialogue_action or action_type in DIALOGUE_ACTION_TYPES)
    return {
        "success": float(success),
        "valid_action": float(valid_action),
        "format_reward": float(format_reward),
        "tool_reward": float(tool_reward),
        "result_reward": float(result_reward),
        "turn_penalty": float(turn_penalty),
        "grounded_recommendation": float(grounded),
        "search_invoked": float(search_invoked),
        "item_details_invoked": float(item_details_invoked),
        "user_preference_invoked": float(user_preference_invoked),
        "search_hit_at_k": float(search_hit_at_k),
        "retrieval_ndcg_at_10": float(retrieval_ndcg_at_10),
        "retrieval_ndcg_at_100": float(retrieval_ndcg_at_100),
        "forced_recommendation": float(forced_recommendation),
        "dialogue_action": float(is_dialogue_action),
        "action_type": action_type,
    }


def append_trace(
    extra_fields: dict[str, Any],
    *,
    source: str,
    action: Any,
    obs: Any,
    reward: float,
    metrics: dict[str, Any],
    done: bool,
    valid_action: bool,
    raw_action: str | None = None,
    action_extraction: str | None = None,
) -> None:
    trace = ensure_tool_trace(extra_fields)
    turn_span_index = extra_fields.get("current_turn_span_index")
    trace.append(
        {
            "source": source,
            "turn_span_index": turn_span_index if isinstance(turn_span_index, int) else None,
            "action": action,
            "raw_action": raw_action,
            "action_extraction": action_extraction,
            "obs": obs,
            "reward": float(reward),
            "metrics": metrics,
            "done": bool(done),
            "valid_action": bool(valid_action),
        }
    )
