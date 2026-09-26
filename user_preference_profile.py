from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


SENTENCE_SPLIT_RE = re.compile(r"[.!?;\n]+")
MULTI_SPACE_RE = re.compile(r"\s+")
COMMON_STOPWORDS = {
    "a",
    "an",
    "and",
    "all",
    "are",
    "as",
    "at",
    "about",
    "also",
    "able",
    "actually",
    "amazing",
    "amazingly",
    "awesome",
    "ask",
    "be",
    "been",
    "behind",
    "because",
    "but",
    "by",
    "can",
    "could",
    "did",
    "didn",
    "does",
    "don",
    "done",
    "excellent",
    "for",
    "from",
    "get",
    "gets",
    "getting",
    "good",
    "great",
    "got",
    "had",
    "has",
    "have",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "just",
    "kind",
    "like",
    "looking",
    "made",
    "make",
    "makes",
    "my",
    "need",
    "nice",
    "of",
    "on",
    "or",
    "perfect",
    "product",
    "products",
    "purchase",
    "really",
    "right",
    "service",
    "so",
    "something",
    "still",
    "that",
    "them",
    "then",
    "the",
    "these",
    "they",
    "their",
    "this",
    "to",
    "use",
    "very",
    "want",
    "was",
    "well",
    "were",
    "when",
    "with",
    "works",
    "you",
    "your",
    "see",
}


def clean_text(value: Any) -> str:
    text = str(value or "")
    text = MULTI_SPACE_RE.sub(" ", text.replace("\n", " "))
    return text.strip()


def compact_text(text: Any, limit: int = 200) -> str:
    compact = clean_text(text)
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)] + "..."


def split_sentences(text: Any) -> list[str]:
    return [part.strip(" -,:") for part in SENTENCE_SPLIT_RE.split(clean_text(text)) if len(part.strip()) >= 12]


def normalized_text_key(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", clean_text(text).lower())


def normalize_history_entry(row: dict[str, Any]) -> dict[str, Any] | None:
    user_id = clean_text(row.get("user_id"))
    if not user_id:
        return None

    if any(key in row for key in ("reference_query", "reference_review", "target_item_id")):
        return {
            "user_id": user_id,
            "qid": clean_text(row.get("qid")),
            "item_id": clean_text(row.get("target_item_id")).upper(),
            "query": clean_text(row.get("reference_query")),
            "review_text": clean_text(row.get("reference_review")),
            "title": clean_text(row.get("item_title") or row.get("title")),
            "category": clean_text(row.get("category") or row.get("main_category")),
            "rating": row.get("rating"),
            "timestamp": row.get("timestamp"),
            "verified_purchase": row.get("verified_purchase"),
            "source_type": "simrec",
        }

    if any(key in row for key in ("review_text", "item_title")):
        return {
            "user_id": user_id,
            "qid": clean_text(row.get("qid")),
            "item_id": clean_text(row.get("parent_asin") or row.get("asin")).upper(),
            "query": "",
            "review_text": clean_text(row.get("review_text")),
            "title": clean_text(row.get("item_title")),
            "category": clean_text(row.get("category") or row.get("main_category")),
            "rating": row.get("rating"),
            "timestamp": row.get("timestamp"),
            "verified_purchase": row.get("verified_purchase"),
            "source_type": "profile_history",
        }

    if any(key in row for key in ("asin", "parent_asin", "text", "title")):
        return {
            "user_id": user_id,
            "qid": "",
            "item_id": clean_text(row.get("parent_asin") or row.get("asin")).upper(),
            "query": "",
            "review_text": clean_text(row.get("text")),
            "title": clean_text(row.get("title")),
            "category": clean_text(row.get("category") or row.get("main_category")),
            "rating": row.get("rating"),
            "helpful_vote": row.get("helpful_vote"),
            "timestamp": row.get("timestamp"),
            "verified_purchase": row.get("verified_purchase"),
            "source_type": "amazon_review",
        }

    return None


def normalize_history_entries(row: dict[str, Any]) -> list[dict[str, Any]]:
    user_id = clean_text(row.get("user_id"))
    user_profile = row.get("user_profile")
    if user_id and isinstance(user_profile, dict):
        purchase_history = user_profile.get("purchase_history")
        if isinstance(purchase_history, list):
            entries = []
            for history_row in purchase_history:
                if not isinstance(history_row, dict):
                    continue
                normalized = normalize_history_entry({"user_id": user_id, **history_row})
                if normalized is not None:
                    entries.append(normalized)
            return entries

    normalized = normalize_history_entry(row)
    return [normalized] if normalized is not None else []


def history_sort_key(item: dict[str, Any]) -> tuple[int, str, str]:
    try:
        timestamp_key = int(item.get("timestamp"))
    except Exception:
        timestamp_key = -1
    return timestamp_key, clean_text(item.get("qid")), clean_text(item.get("item_id")).upper()


def history_text(item: dict[str, Any]) -> str:
    return " ".join(
        part
        for part in (
            item.get("query", ""),
            item.get("review_text", ""),
            item.get("title", ""),
            item.get("category", ""),
        )
        if part
    ).strip()


def merge_history_items(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    for key, value in incoming.items():
        if value in (None, ""):
            continue
        current = merged.get(key)
        if current in (None, ""):
            merged[key] = value
            continue
        if key == "review_text" and len(clean_text(value)) > len(clean_text(current)):
            merged[key] = value
        if (
            key == "title"
            and incoming.get("source_type") == "profile_history"
            and existing.get("source_type") != "profile_history"
        ):
            merged[key] = value
    return merged


def dedupe_and_sort_history(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[str, ...], dict[str, Any]] = {}
    for item in items:
        item_id = clean_text(item.get("item_id")).upper()
        timestamp = str(item.get("timestamp") or "")
        qid = clean_text(item.get("qid"))
        if item_id and timestamp:
            key = ("item_ts", item_id, timestamp)
        elif qid and item_id:
            key = ("qid_item", qid, item_id)
        else:
            key = ("text", qid, item_id, normalized_text_key(item.get("review_text", ""))[:120])
        if key in deduped:
            deduped[key] = merge_history_items(deduped[key], item)
        else:
            deduped[key] = item
    return sorted(deduped.values(), key=history_sort_key)


def load_user_histories(profile_sources: list[str], max_history: int = 0) -> dict[str, list[dict[str, Any]]]:
    profiles: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in profile_sources:
        path = Path(source).expanduser()
        if not path.exists():
            continue
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            for user_id, value in data.items():
                entries = value.get("purchase_history") if isinstance(value, dict) else value
                if not isinstance(entries, list):
                    continue
                for row in entries:
                    if isinstance(row, dict):
                        profiles[user_id].extend(normalize_history_entries({"user_id": user_id, **row}))
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                for item in normalize_history_entries(row):
                    profiles[item["user_id"]].append(item)

    result = {}
    for user_id, items in profiles.items():
        history = dedupe_and_sort_history(items)
        if max_history > 0:
            history = history[-max_history:]
        result[user_id] = history
    return result


def filter_history(
    history: list[dict[str, Any]],
    exclude_qid: str | None = None,
    exclude_item_id: str | None = None,
) -> list[dict[str, Any]]:
    exclude_qid = clean_text(exclude_qid)
    exclude_item_id = clean_text(exclude_item_id).upper()
    return [
        item
        for item in history
        if (not exclude_qid or clean_text(item.get("qid")) != exclude_qid)
        and (not exclude_item_id or clean_text(item.get("item_id")).upper() != exclude_item_id)
    ]


def derive_price_tendency(history: list[dict[str, Any]]) -> str:
    joined = " ".join(history_text(item) for item in history).lower()
    if any(token in joined for token in ("cheap", "cheapest", "budget", "affordable", "low price", "value")):
        return "Budget conscious"
    if any(token in joined for token in ("premium", "high quality", "best", "durable", "worth")):
        return "Quality focused"
    return "Flexible"


def derive_common_needs(history: list[dict[str, Any]], limit: int = 3) -> list[str]:
    candidates = []
    for item in history:
        for source in (item.get("query", ""), item.get("review_text", ""), item.get("title", "")):
            for sentence in split_sentences(source):
                if sentence not in candidates:
                    candidates.append(sentence)
                if len(candidates) >= limit:
                    return candidates
    return candidates[:limit] or ["No recurring needs extracted from history."]


def derive_preferred_terms(history: list[dict[str, Any]], limit: int = 5) -> str:
    counts: defaultdict[str, int] = defaultdict(int)
    for item in history:
        text = history_text(item).lower()
        for token in re.findall(r"[a-z0-9][a-z0-9\-]{2,}", text):
            if token in COMMON_STOPWORDS:
                continue
            counts[token] += 1
    ranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    if not ranked:
        return "No strong recurring keywords."
    return ", ".join(token for token, _ in ranked[:limit])


def derive_rating_summary(history: list[dict[str, Any]]) -> str:
    ratings = []
    low_rated = []
    for item in history:
        try:
            rating = float(item.get("rating"))
        except Exception:
            continue
        ratings.append(rating)
        if rating < 3.0:
            label = clean_text(item.get("title")) or clean_text(item.get("item_id")).upper() or "an item"
            low_rated.append(label)
    if not ratings:
        return "No rating history available."
    summary = f"Average rating {sum(ratings) / len(ratings):.1f}/5 across {len(ratings)} history entries."
    if low_rated:
        summary += " Low-rated signals: " + "; ".join(compact_text(item, limit=80) for item in low_rated[:2]) + "."
    return summary


LLM_USER_PROFILE_PROMPT = """Based on the user's purchase history below, generate a concise user preference profile.

## User Purchase History (chronological order):
{purchase_history}

## Statistics:
- Average rating: {avg_rating}

## Requirements:
1. Summarize the user's likely purchase preferences and favorite product types.
2. If low-rated items clearly indicate dislikes, mention that briefly.
3. Keep the profile to exactly one short natural sentence of 20-30 words.
4. Output ONLY in the <answer> tag, no other explanatory text.
5. Do not include unsupported details, user IDs, bullet points, or multiple sentences.

## Output Format:
<answer>
The user likely prefers practical, durable products in their favorite categories and avoids items with clear quality or usability issues.
</answer>"""


def _format_rating(value: Any) -> str:
    try:
        return f"{float(value):.1f}/5"
    except Exception:
        return "unknown"


def _history_line(item: dict[str, Any]) -> str:
    parts = []
    category = clean_text(item.get("category"))
    title = clean_text(item.get("title"))
    review = compact_text(item.get("review_text"), limit=180)
    query = compact_text(item.get("query"), limit=120)
    if category:
        parts.append(f"Category: {category}")
    if title:
        parts.append(f"Review title: {compact_text(title, limit=100)}")
    if item.get("rating") is not None:
        parts.append(f"Rating: {_format_rating(item.get('rating'))}")
    if query:
        parts.append(f"Need/query: {query}")
    if review:
        parts.append(f"Review: {review}")
    return "- " + " | ".join(parts) if parts else "- Unknown purchase"


def build_llm_profile_prompt(
    history: list[dict[str, Any]],
    max_history: int = 30,
) -> str:
    selected = sorted(history, key=history_sort_key)[-max(1, int(max_history)) :]
    purchase_history = "\n".join(_history_line(item) for item in selected) or "No purchase history available."
    ratings = []
    for item in selected:
        try:
            ratings.append(float(item.get("rating")))
        except Exception:
            continue
    avg_rating = f"{sum(ratings) / len(ratings):.1f}/5" if ratings else "unknown"
    return LLM_USER_PROFILE_PROMPT.format(purchase_history=purchase_history, avg_rating=avg_rating)


def parse_profile_answer(response_text: str) -> str | None:
    text = clean_text(response_text)
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        text = clean_text(match.group(1))
    text = re.sub(r"</?answer>", "", text, flags=re.IGNORECASE).strip()
    if not text:
        return None
    return text


def normalize_user_preference_text(text: str) -> str:
    text = clean_text(text)
    if not text:
        return ""
    if text.lower().startswith("user preference:"):
        prefix, body = text.split(":", 1)
        body = clean_text(body)
        return f"{prefix}: {body}"
    return f"User preference: {text}"


def _join_phrases(items: list[str]) -> str:
    cleaned = [compact_text(item, limit=80) for item in items if clean_text(item)]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"


def _category_term_variants(categories: list[str]) -> set[str]:
    variants: set[str] = set()
    for category in categories:
        for token in re.findall(r"[a-z0-9][a-z0-9\-]{2,}", category.lower()):
            variants.add(token)
            if token.endswith("s") and len(token) > 3:
                variants.add(token[:-1])
            else:
                variants.add(token + "s")
    return variants


def derive_preference_sentence(user_id: str, history: list[dict[str, Any]]) -> str:
    categories = []
    for item in history:
        category = clean_text(item.get("category")).lower()
        if category and category not in categories:
            categories.append(category)
    category_part = f"{_join_phrases(categories[:2])} products" if categories else "products"

    terms = derive_preferred_terms(history, limit=4)
    if terms == "No strong recurring keywords.":
        title_terms = []
        for item in reversed(history):
            title = clean_text(item.get("title"))
            if title and title not in title_terms:
                title_terms.append(title)
            if len(title_terms) >= 2:
                break
        preference_part = _join_phrases(title_terms) or "items similar to their recent purchases"
    else:
        category_terms = _category_term_variants(categories)
        term_items = [term.strip() for term in terms.split(",") if term.strip() and term.strip().lower() not in category_terms]
        if term_items:
            preference_part = f"{category_part} related to {', '.join(term_items[:4])}"
        else:
            preference_part = category_part

    price_tendency = derive_price_tendency(history).lower()
    return (
        f"User preference: User {clean_text(user_id)} likely prefers {preference_part} "
        f"with a {price_tendency} shopping style."
    )


def render_user_preference(
    user_id: str,
    history: list[dict[str, Any]],
    history_k: int = 5,
    exclude_qid: str | None = None,
    exclude_item_id: str | None = None,
) -> tuple[str, bool]:
    user_id = clean_text(user_id)
    if not user_id:
        return "User preference: Missing user ID for this episode.", False

    filtered = filter_history(history, exclude_qid=exclude_qid, exclude_item_id=exclude_item_id)
    if not filtered:
        return (
            f"User preference: User {user_id} has no prior purchase history available in local data.",
            False,
        )

    history_k = max(1, int(history_k))
    selected = sorted(filtered, key=history_sort_key)[-history_k:]
    return derive_preference_sentence(user_id, selected), True


def render_user_profile_field(
    user_id: str,
    user_profile: Any,
    history_k: int = 5,
    exclude_qid: str | None = None,
    exclude_item_id: str | None = None,
) -> tuple[str, bool]:
    if isinstance(user_profile, str) and clean_text(user_profile):
        text = clean_text(user_profile)
        if text.lower() in {"none", "null", "no prior history"}:
            return render_user_preference(user_id, [], history_k, exclude_qid, exclude_item_id)
        if text.startswith("User preference:"):
            return text, True
        return f"User preference: {text}", True
    if isinstance(user_profile, dict):
        history = normalize_history_entries({"user_id": user_id, "user_profile": user_profile})
        history = dedupe_and_sort_history(history)
        return render_user_preference(
            user_id,
            history,
            history_k=history_k,
            exclude_qid=exclude_qid,
            exclude_item_id=exclude_item_id,
        )
    return render_user_preference(user_id, [], history_k, exclude_qid, exclude_item_id)
