from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


CATEGORY_ALIASES = {
    "baby": "Baby",
    "baby products": "Baby",
    "beauty": "Beauty",
    "beauty and personal care": "Beauty",
    "care": "Beauty",
    "clothing": "Clothing",
    "clothing shoes and jewelry": "Clothing",
    "electronics": "Electronics",
    "health and household": "Health",
    "health": "Health",
    "household": "Health",
    "home": "Home",
    "home and kitchen": "Home",
    "kindle": "Books",
    "kindle store": "Books",
    "books": "Books",
    "office": "Office",
    "office products": "Office",
    "pet": "Pet",
    "pet supplies": "Pet",
    "sports": "Sports",
    "sports and outdoors": "Sports",
    "tools": "Tools",
    "tools and home improvement": "Tools",
    "games": "Video Games",
    "video games": "Video Games",
    "automotive": "Automotive",
    "car accessories": "Automotive",
    "grocery": "Grocery",
    "grocery and gourmet food": "Grocery",
    "toys": "Toys",
    "toys and games": "Toys",
    "movies": "Movies & Music",
    "movies and tv": "Movies & Music",
    "music": "Movies & Music",
    "cds and vinyl": "Movies & Music",
}

CATEGORY_KEYWORDS = {
    "Baby": ("baby", "infant", "toddler", "diaper", "stroller", "crib", "pacifier"),
    "Beauty": ("beauty", "skin", "skincare", "hair", "shampoo", "cosmetic", "makeup", "lotion", "serum"),
    "Clothing": ("shirt", "dress", "shoe", "sneaker", "boot", "jacket", "clothing", "size", "fit", "jewelry", "necklace", "glove", "watch", "wallet"),
    "Electronics": ("computer", "laptop", "mouse", "keyboard", "headphone", "usb", "cable", "charger", "phone", "tv", "camera", "webcam", "graphics card", "adapter", "smart bulb", "alexa"),
    "Health": ("health", "supplement", "vitamin", "medical", "massage", "pain", "household", "cleaner"),
    "Home": ("kitchen", "home", "jar", "bottle", "furniture", "pillow", "blanket", "mirror", "cook", "bedding", "storage", "decor", "rack", "mattress", "recliner"),
    "Books": ("book", "novel", "author", "paperback", "hardcover", "kindle", "edition", "volume"),
    "Office": ("office", "printer", "paper", "notebook", "desk", "stapler", "label"),
    "Pet": ("pet", "dog", "cat", "aquarium", "leash", "litter"),
    "Sports": ("sport", "outdoor", "running", "fitness", "golf", "camping", "hiking", "exercise"),
    "Tools": ("tool", "drill", "mount", "hardware", "screw", "repair", "workshop"),
    "Video Games": ("game", "gaming", "playstation", "xbox", "nintendo", "switch", "ps3", "ps4", "ps5"),
    "Automotive": ("car", "vehicle", "automotive", "jeep", "truck", "windshield", "hood", "trunk", "diesel"),
    "Grocery": ("food", "grocery", "olive oil", "oatmeal", "popcorn", "snack", "coffee", "tea", "flavor"),
    "Toys": ("toy", "lego", "building set", "doll", "marble run", "playset", "action figure"),
    "Movies & Music": ("movie", "film", "dvd", "blu-ray", "album", "music", "song", "television series", "tv series"),
}

ASIN_RE = re.compile(r"\bB[A-Z0-9]{9}\b", re.IGNORECASE)
ID_FIELD_RE = re.compile(r"\b(?:item|product|target)[ _-]?id\b", re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+-]*", re.IGNORECASE)
NUMERIC_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:inch|inches|cm|mm|oz|gb|tb|w|v|mah|pack|count)?\b", re.IGNORECASE)


def clean_text(value: Any, limit: int | None = None) -> str:
    text = " ".join(str(value or "").split())
    if limit is not None and len(text) > limit:
        return text[: max(0, limit - 3)].rstrip() + "..."
    return text


def normalize_category(value: Any) -> str:
    raw = clean_text(value).replace("_", " ").replace("/", " ").lower()
    raw = " ".join(raw.split())
    if not raw:
        return ""
    if raw in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[raw]
    for alias, category in CATEGORY_ALIASES.items():
        if alias in raw:
            return category
    return ""


def category_from_source(source: Any) -> str:
    text = clean_text(source)
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"mcauley-lab/amazon-c4", "amazon-c4", "amazon_c4"}:
        return ""
    for suffix in ("_train", "_test", "_val", "-train", "-test", "-val"):
        if lowered.endswith(suffix):
            text = text[: -len(suffix)]
            lowered = lowered[: -len(suffix)]
            break
    for prefix in ("amazon_c4_", "mcauley-lab/amazon-c4_", "mcauley_lab_amazon_c4_"):
        if lowered.startswith(prefix):
            text = text[len(prefix) :]
            break
    return normalize_category(text)


def infer_category_from_text(value: Any) -> tuple[str, float]:
    text = clean_text(value).lower()
    if not text:
        return "", 0.0
    scores: Counter[str] = Counter()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if re.search(rf"\b{re.escape(keyword)}s?\b", text):
                scores[category] += 1
    if not scores:
        return "", 0.0
    ranked = scores.most_common(2)
    category, score = ranked[0]
    margin = score - (ranked[1][1] if len(ranked) > 1 else 0)
    confidence = min(0.95, 0.45 + 0.12 * score + 0.08 * max(0, margin))
    return category, confidence


def infer_categories_from_results(results: Iterable[dict[str, Any]], *, max_categories: int = 2) -> list[tuple[str, float]]:
    counts: Counter[str] = Counter()
    total = 0
    for result in results:
        if not isinstance(result, dict):
            continue
        category = normalize_category(result.get("category") or result.get("main_category"))
        if not category:
            content = clean_text(result.get("content"))
            match = re.search(r"\bCategory:\s*([^|\n.;]+)", content, re.IGNORECASE)
            if match:
                category = normalize_category(match.group(1))
        if category:
            counts[category] += 1
            total += 1
    if not counts:
        return []
    ranked = []
    for category, count in counts.most_common(max(1, max_categories)):
        ranked.append((category, min(0.98, 0.5 + 0.5 * count / max(1, total))))
    return ranked


def infer_category_from_results(results: Iterable[dict[str, Any]]) -> tuple[str, float]:
    ranked = infer_categories_from_results(results, max_categories=1)
    return ranked[0] if ranked else ("", 0.0)


def classify_scenario(text: Any) -> str:
    value = clean_text(text).lower()
    if any(term in value for term in ("compatible", "compatibility", "usb", "lightning", "platform", "ps3", "ps4", "ps5", "xbox")):
        return "compatibility"
    if NUMERIC_RE.search(value):
        return "numeric_constraint"
    if any(term in value for term in ("include", "bundle", "come with", "without", "exclude", "pack of")):
        return "bundle_or_exclusion"
    if any(term in value for term in ("new", "unused", "authentic", "certified", "edition", "region")):
        return "condition_or_version"
    if any(term in value for term in ("style", "look", "taste", "cute", "color", "fit", "comfort")):
        return "style_or_fit"
    return "general_constraints"


def _metric(record: dict[str, Any], key: str) -> float:
    try:
        return float(record.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _dialogue(record: dict[str, Any], role: str) -> list[str]:
    return [
        clean_text(turn.get("content"), 700)
        for turn in record.get("dialogue") or []
        if isinstance(turn, dict) and turn.get("role") == role and clean_text(turn.get("content"))
    ]


def _tool_name_and_args(action: Any) -> tuple[str, dict[str, Any]]:
    try:
        payload = json.loads(str(action or ""))
    except json.JSONDecodeError:
        return "", {}
    if not isinstance(payload, dict):
        return "", {}
    args = payload.get("arguments")
    return clean_text(payload.get("tool_name") or payload.get("name")), args if isinstance(args, dict) else {}


def build_evidence_events(record: dict[str, Any]) -> list[dict[str, Any]]:
    category = normalize_category(record.get("target_category")) or category_from_source(record.get("source"))
    users = _dialogue(record, "user")
    if not category:
        category, _ = infer_category_from_text(record.get("reference_query") or (users[0] if users else ""))
    searches = [item for item in record.get("search_trajectory") or [] if isinstance(item, dict)]
    outcome = {
        "reward": _metric(record, "reward"),
        "success": _metric(record, "success"),
        "hit_at_1": _metric(record, "recommend_hit_at_1"),
        "final_ndcg_at_100": _metric(record, "final_retrieval_ndcg_at_100"),
    }
    common = {
        "qid": clean_text(record.get("qid")),
        "source": clean_text(record.get("source")),
        "category": category or "Unknown",
        "outcome": outcome,
    }
    events: list[dict[str, Any]] = []
    if searches:
        first = searches[0]
        need = users[0] if users else clean_text(record.get("reference_query"), 700)
        events.append(
            {
                **common,
                "phase": "initial_search",
                "skill_type": "query_construction",
                "scenario": classify_scenario(need),
                "context": need,
                "action": {"query": clean_text(first.get("query"), 500)},
                "local_outcome": {
                    "ndcg_at_100": _metric(first, "ndcg_at_100"),
                    "hit_at_10": _metric(first, "hit_at_10"),
                    "duplicate_query": bool(first.get("duplicate_query")),
                    "new_result_count": int(first.get("new_result_count", 0) or 0),
                },
            }
        )
        previous_ndcg = _metric(first, "ndcg_at_100")
        previous_query = clean_text(first.get("query"), 500)
        for index, search in enumerate(searches[1:], start=1):
            feedback_count = int(search.get("feedback_count", index) or index)
            feedback = users[min(feedback_count, len(users) - 1)] if users else ""
            ndcg = _metric(search, "ndcg_at_100")
            events.append(
                {
                    **common,
                    "phase": "feedback_revision",
                    "skill_type": "query_revision",
                    "scenario": classify_scenario(feedback),
                    "context": feedback,
                    "action": {
                        "previous_query": previous_query,
                        "revised_query": clean_text(search.get("query"), 500),
                    },
                    "local_outcome": {
                        "ndcg_before": previous_ndcg,
                        "ndcg_after": ndcg,
                        "ndcg_delta": ndcg - previous_ndcg,
                        "hit_at_10": _metric(search, "hit_at_10"),
                        "duplicate_query": bool(search.get("duplicate_query")),
                    },
                }
            )
            previous_ndcg = max(previous_ndcg, ndcg)
            previous_query = clean_text(search.get("query"), 500)

    for trace in record.get("tool_trace") or []:
        if not isinstance(trace, dict):
            continue
        tool_name, _ = _tool_name_and_args(trace.get("action"))
        if tool_name in {"get_item_details", "item_details"}:
            context = users[-1] if users else clean_text(record.get("reference_query"), 700)
            events.append(
                {
                    **common,
                    "phase": "evidence_check",
                    "skill_type": "evidence_verification",
                    "scenario": classify_scenario(context),
                    "context": context,
                    "action": {"tool": "get_item_details"},
                    "local_outcome": {
                        "valid_action": bool(trace.get("valid_action")),
                        "grounded_recommendation": _metric(record, "grounded_recommendation"),
                        "hit_at_1": outcome["hit_at_1"],
                    },
                }
            )
        elif tool_name in {"get_user_preference", "user_preference"}:
            context = users[0] if users else clean_text(record.get("reference_query"), 700)
            events.append(
                {
                    **common,
                    "phase": "preference_lookup",
                    "skill_type": "preference_lookup",
                    "scenario": "broad_preference",
                    "context": context,
                    "action": {"tool": "get_user_preference"},
                    "local_outcome": {"valid_action": bool(trace.get("valid_action")), **outcome},
                }
            )

    recommended = [item for item in record.get("recommended_items") or [] if isinstance(item, dict)]
    if recommended:
        context = users[-1] if users else clean_text(record.get("reference_query"), 700)
        events.append(
            {
                **common,
                "phase": "recommendation",
                "skill_type": "recommendation_composition",
                "scenario": classify_scenario(context),
                "context": context,
                "action": {"description": clean_text(recommended[0].get("description"), 500)},
                "local_outcome": {
                    "structured": True,
                    "grounded_recommendation": _metric(record, "grounded_recommendation"),
                    "hit_at_1": outcome["hit_at_1"],
                },
            }
        )
    return events


def build_evidence_bundles(
    events: Iterable[dict[str, Any]], *, min_distinct_qids: int = 4, max_examples: int = 12
) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        key = (
            clean_text(event.get("category")) or "Unknown",
            clean_text(event.get("phase")),
            clean_text(event.get("skill_type")),
            clean_text(event.get("scenario")),
        )
        groups[key].append(event)
    bundles = []
    for key, items in sorted(groups.items()):
        qids = {clean_text(item.get("qid")) for item in items if clean_text(item.get("qid"))}
        if len(qids) < min_distinct_qids:
            continue
        successes = sorted(
            items,
            key=lambda item: (
                float((item.get("outcome") or {}).get("hit_at_1", 0.0)),
                float((item.get("outcome") or {}).get("reward", 0.0)),
            ),
            reverse=True,
        )
        positives = [item for item in successes if float((item.get("outcome") or {}).get("success", 0.0)) > 0][: max_examples * 2 // 3]
        negatives = [item for item in reversed(successes) if float((item.get("outcome") or {}).get("success", 0.0)) <= 0][: max_examples // 3]
        selected = (positives + negatives)[:max_examples]
        digest = hashlib.sha1("|".join(key).encode("utf-8")).hexdigest()[:12]
        bundles.append(
            {
                "bundle_id": f"{key[0].lower().replace(' ', '_')}:{key[1]}:{key[3]}:{digest}",
                "category": key[0],
                "phase": key[1],
                "skill_type": key[2],
                "scenario": key[3],
                "support_count": len(qids),
                "evidence": selected,
            }
        )
    return bundles


def skill_text(skill: dict[str, Any]) -> str:
    policy = skill.get("policy")
    if isinstance(policy, list):
        policy = " ".join(str(item) for item in policy)
    avoid = skill.get("avoid")
    if isinstance(avoid, list):
        avoid = " ".join(str(item) for item in avoid)
    return clean_text(" ".join(str(value or "") for value in (skill.get("trigger"), policy, avoid, skill.get("retrieval_text"))))


def validate_abstract_skill(skill: dict[str, Any], *, min_support: int = 4) -> list[str]:
    errors: list[str] = []
    text = skill_text(skill)
    if not clean_text(skill.get("trigger")):
        errors.append("missing trigger")
    if not clean_text(skill.get("policy")):
        errors.append("missing policy")
    if ASIN_RE.search(text) or ID_FIELD_RE.search(text):
        errors.append("contains item identifier")
    action = skill.get("action") or skill.get("tool_policy") or {}
    if isinstance(action, dict) and any(key in action for key in ("item_id", "query", "description", "recommended_item_id")):
        errors.append("contains trajectory-specific action fields")
    support_count = int(skill.get("support_count", 0) or 0)
    if support_count < min_support:
        errors.append(f"support_count<{min_support}")
    scope = skill.get("scope")
    if not isinstance(scope, dict) or scope.get("level") not in {"global", "category"}:
        errors.append("invalid scope")
    if isinstance(scope, dict) and scope.get("level") == "category" and not scope.get("categories"):
        errors.append("category scope has no categories")
    if len(text) > 1800:
        errors.append("skill text is too long")
    return errors


def finalize_candidate(raw: dict[str, Any], bundle: dict[str, Any], *, model: str, prompt_hash: str) -> dict[str, Any]:
    candidate = dict(raw)
    category = normalize_category(bundle.get("category"))
    level = clean_text((candidate.get("scope") or {}).get("level")).lower()
    if level not in {"global", "category"}:
        level = "category" if category and category != "Unknown" else "global"
    candidate["scope"] = {
        "level": level,
        "categories": [category] if level == "category" and category else [],
    }
    candidate["phase"] = clean_text(candidate.get("phase") or bundle.get("phase"))
    candidate["skill_type"] = clean_text(candidate.get("skill_type") or bundle.get("skill_type"))
    candidate["trigger"] = clean_text(candidate.get("trigger"), 500)
    policy = candidate.get("policy")
    candidate["policy"] = [clean_text(item, 500) for item in policy if clean_text(item)] if isinstance(policy, list) else clean_text(policy, 900)
    avoid = candidate.get("avoid")
    candidate["avoid"] = [clean_text(item, 400) for item in avoid if clean_text(item)] if isinstance(avoid, list) else []
    candidate["retrieval_text"] = clean_text(candidate.get("retrieval_text") or skill_text(candidate), 1000)
    candidate["support_count"] = int(bundle.get("support_count", 0) or 0)
    candidate["status"] = "candidate"
    candidate["version"] = int(candidate.get("version", 1) or 1)
    candidate["parent_ids"] = list(candidate.get("parent_ids") or [])
    candidate["reliability"] = {"alpha": 1.0, "beta": 1.0, "usage_count": 0}
    candidate["provenance_ref"] = f"skill_evidence/{bundle.get('bundle_id')}.json"
    candidate["distillation"] = {"model": model, "prompt_hash": prompt_hash}
    identity = "|".join(
        [candidate["scope"]["level"], category, candidate["phase"], candidate["skill_type"], candidate["trigger"], clean_text(candidate["policy"])]
    )
    slug = re.sub(r"[^a-z0-9]+", "_", clean_text(candidate.get("name") or candidate["skill_type"]).lower()).strip("_")[:48]
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]
    candidate["skill_id"] = clean_text(candidate.get("skill_id")) or f"{slug}:{digest}"
    return candidate


def token_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_tokens = set(TOKEN_RE.findall(skill_text(left).lower()))
    right_tokens = set(TOKEN_RE.findall(skill_text(right).lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def merge_candidates(skills: Iterable[dict[str, Any]], *, threshold: float = 0.72) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    status_priority = {"active": 3, "candidate": 2, "deprecated": 1}
    for skill in sorted(
        skills,
        key=lambda item: (
            status_priority.get(str(item.get("status") or "active").lower(), 0),
            int(item.get("support_count", 0) or 0),
        ),
        reverse=True,
    ):
        duplicate = False
        for existing in kept:
            if skill.get("phase") != existing.get("phase") or skill.get("scope") != existing.get("scope"):
                continue
            if token_similarity(skill, existing) >= threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(skill)
    return sorted(kept, key=lambda item: str(item.get("skill_id")))


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
    return [payload] if isinstance(payload, dict) else []


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count
