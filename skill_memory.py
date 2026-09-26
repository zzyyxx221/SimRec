from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path
from typing import Any

from SimRec.skill_evolution import infer_category_from_text, normalize_category


DEFAULT_SKILL_TOP_K = int(os.environ.get("SIMREC_SKILL_TOP_K", "3"))
DEFAULT_MIN_RELIABILITY = float(os.environ.get("SIMREC_SKILL_MIN_RELIABILITY", "0.0"))
DEFAULT_MAX_CANDIDATE_SKILLS = int(os.environ.get("SIMREC_SKILL_MAX_CANDIDATES", "3000"))
MAX_RENDERED_SKILL_CHARS = int(os.environ.get("SIMREC_SKILL_MAX_RENDERED_CHARS", "900"))
DEFAULT_CATEGORY_MIN_CONFIDENCE = float(os.environ.get("SIMREC_SKILL_CATEGORY_MIN_CONFIDENCE", "0.6"))
DEFAULT_MAX_SKILLS_PER_TYPE = int(os.environ.get("SIMREC_SKILL_MAX_PER_TYPE", "2"))
HASH_EMBEDDING_DIM = int(os.environ.get("SIMREC_SKILL_HASH_EMBEDDING_DIM", "128"))
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+-]*", re.IGNORECASE)
PREFERENCE_CUE_RE = re.compile(
    r"\b("
    r"my style|my preference|preferences?|favorite|favourite|"
    r"i would like|i'd like|something i would like|something stylish|"
    r"stylish|taste|recommend something|what would i like"
    r")\b",
    re.IGNORECASE,
)
EXACT_CONSTRAINT_RE = re.compile(
    r"\b("
    r"\d|inch|inches|cm|mm|oz|gb|tb|mfi|usb-c|lightning|ps3|ps4|ps5|"
    r"brand new|unused|redeemable|compatible|certified|specific|exact"
    r")\b",
    re.IGNORECASE,
)


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def compact_text(value: Any, limit: int) -> str:
    text = normalize_text(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def tokenize(value: Any) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(str(value or ""))]


def token_hash_embedding(tokens: set[str], *, dim: int = HASH_EMBEDDING_DIM) -> list[float]:
    if dim <= 0 or not tokens:
        return []
    vector = [0.0] * dim
    for token in tokens:
        digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[bucket] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0.0:
        return []
    return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return float(sum(a * b for a, b in zip(left, right, strict=False)))


def normalize_vector(value: Any) -> list[float]:
    if not isinstance(value, list) or not value:
        return []
    try:
        vector = [float(item) for item in value]
    except (TypeError, ValueError):
        return []
    norm = math.sqrt(sum(item * item for item in vector))
    if norm <= 0.0:
        return []
    return [item / norm for item in vector]


def observed_categories_from_state(state: dict[str, Any], *, min_confidence: float) -> list[str]:
    raw_categories = state.get("observed_categories")
    categories: list[str] = []
    if isinstance(raw_categories, list):
        for value in raw_categories:
            category = normalize_category(value)
            if category and category not in categories:
                categories.append(category)
            if len(categories) >= 2:
                break
    category = normalize_category(state.get("observed_category"))
    if category and category not in categories:
        categories.append(category)
    if not categories:
        return []
    confidence = state.get("observed_category_confidence")
    if confidence is None:
        confidence = 1.0
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = 0.0
    return categories[:2] if confidence_value >= min_confidence else []


def load_json_records(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def skill_reliability(skill: dict[str, Any]) -> float:
    if "reliability" in skill:
        payload = skill.get("reliability")
        if isinstance(payload, dict):
            alpha = float(payload.get("alpha", 1.0) or 1.0)
            beta = float(payload.get("beta", 1.0) or 1.0)
            return alpha / max(1e-9, alpha + beta)
        try:
            return float(payload)
        except (TypeError, ValueError):
            return 0.0
    usage_count = float(skill.get("usage_count", 0) or 0)
    success_count = float(skill.get("success_count", 0) or 0)
    return float((success_count + 1.0) / (usage_count + 2.0))


def skill_type_bonus(skill_type: str, phase: str) -> float:
    if phase == "initial_search" and skill_type in {"initial_search_query", "query_construction"}:
        return 0.35
    if phase == "feedback_revision" and skill_type in {"feedback_query_revision", "query_revision"}:
        return 0.45
    if phase == "recommendation" and skill_type in {"recommendation_format", "recommendation_composition"}:
        return 0.30
    if phase == "recommendation" and skill_type in {"evidence_check", "evidence_verification"}:
        return 0.18
    if phase == "evidence_check" and skill_type in {"evidence_check", "evidence_verification"}:
        return 0.30
    if phase == "initial_search" and skill_type in {"user_preference_lookup", "preference_lookup"}:
        return 0.08
    if phase == "preference_lookup" and skill_type in {"user_preference_lookup", "preference_lookup"}:
        return 0.30
    return 0.0


def infer_phase(messages: list[dict[str, Any]], state: dict[str, Any] | None = None) -> str:
    state = state or {}
    user_turns = sum(1 for message in messages if message.get("role") == "user")
    latest_user = ""
    for message in reversed(messages):
        if message.get("role") == "user":
            latest_user = normalize_text(message.get("content"))
            break
    if latest_user and PREFERENCE_CUE_RE.search(latest_user) and not EXACT_CONSTRAINT_RE.search(latest_user):
        return "preference_lookup"
    if state.get("must_recommend_next") or state.get("last_search_results"):
        if user_turns >= 2 and state.get("recommendation_count", 0):
            return "feedback_revision"
        return "recommendation"
    if user_turns >= 2:
        return "feedback_revision"
    return "initial_search"


def build_query_text(messages: list[dict[str, Any]], state: dict[str, Any] | None = None) -> str:
    state = state or {}
    public_parts = []
    for message in messages[-8:]:
        role = message.get("role")
        if role not in {"user", "assistant", "recommender"}:
            continue
        content = normalize_text(message.get("content"))
        if content:
            public_parts.append(f"{role}: {content}")
    if state.get("search_history"):
        public_parts.append("previous searches: " + " | ".join(str(item) for item in state.get("search_history", [])[-3:]))
    if state.get("last_search_result_ids"):
        public_parts.append("last result ids: " + " ".join(str(item) for item in state.get("last_search_result_ids", [])[:5]))
    return " ".join(public_parts)


@dataclass(frozen=True)
class RetrievedSkill:
    skill: dict[str, Any]
    score: float
    rank: int


class SkillMemory:
    def __init__(
        self,
        skills: list[dict[str, Any]],
        *,
        top_k: int = DEFAULT_SKILL_TOP_K,
        min_reliability: float = DEFAULT_MIN_RELIABILITY,
        max_candidate_skills: int = DEFAULT_MAX_CANDIDATE_SKILLS,
        category_min_confidence: float = DEFAULT_CATEGORY_MIN_CONFIDENCE,
        max_skills_per_type: int = DEFAULT_MAX_SKILLS_PER_TYPE,
    ):
        self.top_k = max(0, int(top_k))
        self.min_reliability = float(min_reliability)
        self.max_candidate_skills = max(1, int(max_candidate_skills))
        self.category_min_confidence = float(category_min_confidence)
        self.max_skills_per_type = max(1, int(max_skills_per_type))
        self.skills = [skill for skill in skills if self._is_active(skill)]
        self._skill_tokens = [set(tokenize(self._skill_text(skill))) for skill in self.skills]
        self._skill_token_lens = [len(tokens) for tokens in self._skill_tokens]
        self._skill_embeddings = [
            normalize_vector(skill.get("embedding")) or token_hash_embedding(tokens) for skill, tokens in zip(self.skills, self._skill_tokens, strict=False)
        ]
        self._inverted_index: dict[str, list[int]] = defaultdict(list)
        self._phase_index: dict[str, list[int]] = defaultdict(list)
        self._category_index: dict[str, list[int]] = defaultdict(list)
        self._idf: dict[str, float] = {}
        versions = Counter(str(skill.get("library_version") or "") for skill in self.skills if skill.get("library_version"))
        self.library_version = versions.most_common(1)[0][0] if versions else "unversioned"
        for index, (skill, tokens) in enumerate(zip(self.skills, self._skill_tokens, strict=False)):
            for token in tokens:
                self._inverted_index[token].append(index)
            phases = [str(skill.get("phase"))] if skill.get("phase") else self._matching_phases(str(skill.get("skill_type") or ""))
            for phase in phases:
                self._phase_index[phase].append(index)
            level, categories = self._scope(skill)
            if level == "global":
                self._category_index["__global__"].append(index)
            for category in categories:
                self._category_index[category].append(index)
        total_docs = max(1, len(self.skills))
        self._idf = {
            token: math.log(1.0 + (total_docs - len(indices) + 0.5) / (len(indices) + 0.5))
            for token, indices in self._inverted_index.items()
        }

    @classmethod
    def from_path(
        cls,
        path: str | os.PathLike[str] | None,
        *,
        top_k: int = DEFAULT_SKILL_TOP_K,
        min_reliability: float = DEFAULT_MIN_RELIABILITY,
        max_candidate_skills: int = DEFAULT_MAX_CANDIDATE_SKILLS,
        category_min_confidence: float = DEFAULT_CATEGORY_MIN_CONFIDENCE,
        max_skills_per_type: int = DEFAULT_MAX_SKILLS_PER_TYPE,
    ) -> "SkillMemory":
        if not path:
            return cls(
                [],
                top_k=top_k,
                min_reliability=min_reliability,
                max_candidate_skills=max_candidate_skills,
                category_min_confidence=category_min_confidence,
                max_skills_per_type=max_skills_per_type,
            )
        skill_path = Path(path).expanduser().resolve()
        if not skill_path.exists():
            return cls(
                [],
                top_k=top_k,
                min_reliability=min_reliability,
                max_candidate_skills=max_candidate_skills,
                category_min_confidence=category_min_confidence,
                max_skills_per_type=max_skills_per_type,
            )
        return cls(
            load_json_records(skill_path),
            top_k=top_k,
            min_reliability=min_reliability,
            max_candidate_skills=max_candidate_skills,
            category_min_confidence=category_min_confidence,
            max_skills_per_type=max_skills_per_type,
        )

    def _is_active(self, skill: dict[str, Any]) -> bool:
        status = str(skill.get("status") or "active").lower()
        include_candidates = env_flag("SIMREC_SKILL_INCLUDE_CANDIDATES", False)
        return (status == "active" or (status == "candidate" and include_candidates)) and skill_reliability(
            skill
        ) >= self.min_reliability

    @staticmethod
    def _skill_text(skill: dict[str, Any]) -> str:
        fields = [
            skill.get("skill_type"),
            skill.get("trigger"),
            skill.get("policy"),
            skill.get("rationale"),
            skill.get("retrieval_text"),
        ]
        action = skill.get("action")
        if isinstance(action, dict):
            fields.extend(str(value) for key, value in action.items() if key not in {"item_id"} and value is not None)
        return " ".join(str(field or "") for field in fields)

    @staticmethod
    def _matching_phases(skill_type: str) -> list[str]:
        if skill_type in {"initial_search_query", "query_construction"}:
            return ["initial_search"]
        if skill_type in {"feedback_query_revision", "query_revision"}:
            return ["feedback_revision"]
        if skill_type in {"recommendation_format", "recommendation_composition"}:
            return ["recommendation"]
        if skill_type in {"evidence_check", "evidence_verification"}:
            return ["evidence_check"]
        if skill_type in {"user_preference_lookup", "preference_lookup"}:
            return ["preference_lookup"]
        return []

    @staticmethod
    def _scope(skill: dict[str, Any]) -> tuple[str, list[str]]:
        scope = skill.get("scope")
        if not isinstance(scope, dict):
            return "global", []
        level = str(scope.get("level") or "global").lower()
        categories = [normalize_category(value) for value in scope.get("categories") or []]
        return level, [value for value in categories if value]

    def _candidate_indices(self, query_tokens: set[str], phase: str, categories: list[str]) -> list[int]:
        overlap_counts: Counter[int] = Counter()
        for token in query_tokens:
            overlap_counts.update(self._inverted_index.get(token, ()))

        ranked = [index for index, _ in overlap_counts.most_common(self.max_candidate_skills)]
        selected = set(ranked)

        # Keep phase/category fallbacks so broad requests can still retrieve
        # reusable policies when lexical overlap is weak.
        fallback_limit = min(len(self._phase_index.get(phase, ())), max(50, self.max_candidate_skills // 5))
        fallback = list(self._phase_index.get(phase, ())[:fallback_limit])
        fallback.extend(self._category_index.get("__global__", ())[:fallback_limit])
        for category in categories[:2]:
            fallback.extend(self._category_index.get(category, ())[:fallback_limit])
        for index in fallback:
            if index not in selected:
                ranked.append(index)
                selected.add(index)
            if len(ranked) >= self.max_candidate_skills:
                break
        return ranked

    def _bm25_score(self, query_tokens: set[str], index: int) -> float:
        skill_tokens = self._skill_tokens[index]
        if not skill_tokens:
            return 0.0
        score = sum(self._idf.get(token, 0.0) for token in query_tokens & skill_tokens)
        return score / max(1.0, math.sqrt(float(len(skill_tokens))))

    def retrieve(
        self,
        *,
        messages: list[dict[str, Any]],
        state: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedSkill]:
        if not self.skills:
            return []
        if state is None:
            state = {}
        limit = self.top_k if top_k is None else max(0, int(top_k))
        if limit <= 0:
            return []
        phase = infer_phase(messages, state)
        query_text = build_query_text(messages, state)
        query_tokens = set(tokenize(query_text))
        if not query_tokens:
            return []
        category = normalize_category(state.get("observed_category"))
        if not category:
            category, confidence = infer_category_from_text(query_text)
            if category:
                state["observed_category"] = category
                state["observed_category_confidence"] = confidence
        active_categories = observed_categories_from_state(state, min_confidence=self.category_min_confidence)
        query_embedding = normalize_vector(state.get("skill_query_embedding")) or token_hash_embedding(query_tokens)
        scored: list[tuple[float, dict[str, Any]]] = []
        for index in self._candidate_indices(query_tokens, phase, active_categories):
            skill = self.skills[index]
            level, skill_categories = self._scope(skill)
            if level == "category" and not any(category in skill_categories for category in active_categories):
                continue
            skill_tokens = self._skill_tokens[index]
            if not skill_tokens:
                continue
            overlap = len(query_tokens & skill_tokens)
            lexical = overlap / math.sqrt(max(1, len(query_tokens)) * max(1, self._skill_token_lens[index]))
            bm25 = self._bm25_score(query_tokens, index)
            dense = max(0.0, cosine(query_embedding, self._skill_embeddings[index]))
            score = 0.45 * lexical + 0.25 * bm25 + 0.20 * dense
            score += skill_type_bonus(str(skill.get("skill_type") or ""), phase)
            score += 0.15 * skill_reliability(skill)
            score += 0.18 if level == "category" and any(category in skill_categories for category in active_categories) else 0.03
            if str(skill.get("status") or "active").lower() == "candidate":
                score -= 0.08
            if score <= 0.0:
                continue
            scored.append((score, skill))
        scored.sort(key=lambda item: item[0], reverse=True)
        if active_categories and limit > 1:
            category_items = [item for item in scored if self._scope(item[1])[0] == "category"]
            global_items = [item for item in scored if self._scope(item[1])[0] == "global"]
            chosen = category_items[: max(1, limit - 1)] + global_items[:1]
            chosen_ids = {str(item[1].get("skill_id")) for item in chosen}
            chosen.extend(item for item in scored if str(item[1].get("skill_id")) not in chosen_ids)
            scored = chosen
        selected: list[tuple[float, dict[str, Any]]] = []
        type_counts: Counter[str] = Counter()
        for score, skill in scored:
            skill_type = str(skill.get("skill_type") or "")
            if type_counts[skill_type] >= self.max_skills_per_type:
                continue
            selected.append((score, skill))
            type_counts[skill_type] += 1
            if len(selected) >= limit:
                break
        return [RetrievedSkill(skill=skill, score=score, rank=index + 1) for index, (score, skill) in enumerate(selected)]

    @staticmethod
    def render(retrieved: list[RetrievedSkill]) -> str:
        if not retrieved:
            return ""
        lines = ["Learned reusable skills for this turn:"]
        for item in retrieved:
            skill = item.skill
            skill_type = str(skill.get("skill_type") or "skill").replace("_", " ")
            policy_value = skill.get("policy") or skill.get("rationale") or ""
            policy = "; ".join(normalize_text(value) for value in policy_value) if isinstance(policy_value, list) else normalize_text(policy_value)
            trigger = normalize_text(skill.get("trigger") or "")
            action_hint = ""
            if skill.get("skill_type") in {"recommendation_format", "recommendation_composition"}:
                action_hint = " Use exactly recommended_item_id and recommended_item_description with grounded evidence."
            elif skill.get("skill_type") in {"evidence_check", "evidence_verification"}:
                action_hint = " Inspect item details when snippets do not verify a hard constraint."
            elif skill.get("skill_type") in {"user_preference_lookup", "preference_lookup"}:
                action_hint = " Look up user preference only for broad or preference-heavy requests."
            rendered = f"{item.rank}. {skill_type}: {policy}"
            if trigger:
                rendered += f" Trigger: {compact_text(trigger, 180)}"
            rendered += action_hint
            lines.append(compact_text(rendered, 420))
        return compact_text("\n".join(lines), MAX_RENDERED_SKILL_CHARS)

    @staticmethod
    def usage_payload(
        retrieved: list[RetrievedSkill],
        *,
        phase: str,
        turn_index: int,
        observed_category: str = "",
        library_version: str = "unversioned",
        injected: bool = True,
        injection_propensity: float = 1.0,
    ) -> list[dict[str, Any]]:
        rows = []
        for item in retrieved:
            skill = item.skill
            rows.append(
                {
                    "turn_index": int(turn_index),
                    "phase": phase,
                    "skill_id": skill.get("skill_id"),
                    "skill_type": skill.get("skill_type"),
                    "rank": int(item.rank),
                    "score": float(item.score),
                    "reliability": skill_reliability(skill),
                    "observed_category": normalize_category(observed_category),
                    "library_version": library_version,
                    "injected": bool(injected),
                    "injection_propensity": float(injection_propensity),
                }
            )
        return rows


_SKILL_MEMORY_CACHE: dict[tuple[str, int, float, int, float, int], tuple[int, int, SkillMemory]] = {}


def get_skill_memory_from_env() -> SkillMemory | None:
    if not env_flag("SIMREC_ENABLE_SKILLS", False):
        return None
    path = os.environ.get("SIMREC_SKILL_LIBRARY_PATH")
    if not path:
        return None
    top_k = int(os.environ.get("SIMREC_SKILL_TOP_K", str(DEFAULT_SKILL_TOP_K)))
    min_reliability = float(os.environ.get("SIMREC_SKILL_MIN_RELIABILITY", str(DEFAULT_MIN_RELIABILITY)))
    max_candidate_skills = int(os.environ.get("SIMREC_SKILL_MAX_CANDIDATES", str(DEFAULT_MAX_CANDIDATE_SKILLS)))
    category_min_confidence = float(os.environ.get("SIMREC_SKILL_CATEGORY_MIN_CONFIDENCE", str(DEFAULT_CATEGORY_MIN_CONFIDENCE)))
    max_skills_per_type = int(os.environ.get("SIMREC_SKILL_MAX_PER_TYPE", str(DEFAULT_MAX_SKILLS_PER_TYPE)))
    key = (str(Path(path).expanduser()), top_k, min_reliability, max_candidate_skills, category_min_confidence, max_skills_per_type)
    try:
        stat = Path(path).expanduser().stat()
    except OSError:
        return None
    cached = _SKILL_MEMORY_CACHE.get(key)
    if cached is None or cached[:2] != (stat.st_mtime_ns, stat.st_size):
        memory = SkillMemory.from_path(
            path,
            top_k=top_k,
            min_reliability=min_reliability,
            max_candidate_skills=max_candidate_skills,
            category_min_confidence=category_min_confidence,
            max_skills_per_type=max_skills_per_type,
        )
        _SKILL_MEMORY_CACHE[key] = (stat.st_mtime_ns, stat.st_size, memory)
        return memory
    return cached[2]
