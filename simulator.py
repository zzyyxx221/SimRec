from __future__ import annotations

import email.utils
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
OPEN_THINK_RE = re.compile(r"^\s*<think>.*", re.DOTALL | re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
RETRYABLE_HTTP_STATUS_CODES = {408, 409, 422, 425, 429, 500, 502, 503, 504}
TARGET_ITEM_ID_MASK = "[TARGET_ITEM_ID]"


def _parse_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Unsupported boolean value: {value}")


def clean_text(value: Any) -> str:
    text = THINK_BLOCK_RE.sub(" ", str(value or ""))
    text = OPEN_THINK_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def mask_target_item_ids(value: Any, target_item_ids: list[str] | None) -> str:
    """Remove target item identifiers from shopper-facing simulator text."""
    text = str(value or "")
    for item_id in sorted({str(item or "").strip() for item in target_item_ids or [] if str(item or "").strip()}, key=len, reverse=True):
        text = re.sub(re.escape(item_id), TARGET_ITEM_ID_MASK, text, flags=re.IGNORECASE)
    return clean_text(text)


@dataclass
class UserSession:
    qid: str
    target_item_id: str
    reference_query: str
    target_item_ids: list[str] | None = None
    initial_user_utterance: str = ""
    turn_index: int = 1
    recommendation_attempts: int = 0
    accepted: bool = False
    done: bool = False


def _build_fallback_initial_user_utterance(reference_query: str) -> str:
    query = clean_text(reference_query)
    if query:
        return query if query.endswith((".", "!", "?")) else f"{query}."
    return "I'm looking for a product that fits my needs."


def _normalize_target_item_ids(sample: dict[str, Any]) -> list[str]:
    raw_targets = sample.get("target_item_ids")
    if raw_targets is None:
        raw_targets = sample.get("target_ids")
    if raw_targets is None:
        raw_targets = [sample.get("target_item_id")]
    if not isinstance(raw_targets, (list, tuple, set)):
        raw_targets = [raw_targets]

    targets: list[str] = []
    for item in raw_targets:
        item_id = str(item or "").strip().upper()
        if item_id and item_id not in targets:
            targets.append(item_id)
    return targets


class HeuristicUserSimulator:
    """Minimal compatibility shim for the legacy environment path."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def start_episode(self, sample: dict[str, Any], initial_user_utterance: str | None = None) -> UserSession:
        opener = clean_text(initial_user_utterance) or _build_fallback_initial_user_utterance(
            sample.get("reference_query", ""),
        )
        return UserSession(
            qid=str(sample.get("qid", "0")),
            target_item_id=str(sample.get("target_item_id", "")).upper(),
            reference_query=clean_text(sample.get("reference_query", "")),
            target_item_ids=_normalize_target_item_ids(sample),
            initial_user_utterance=opener,
        )

    def answer_question(self, session: UserSession, question: str) -> str:
        del question
        session.turn_index += 1
        return "I mainly care that it matches what I'm looking for."

    def respond_to_recommendation(
        self,
        session: UserSession,
        rec_text: str,
        recommended_item_ids: list[str] | None = None,
        grounded: bool = False,
    ) -> dict[str, Any]:
        session.recommendation_attempts += 1
        recommended_item_ids = [str(item).upper() for item in (recommended_item_ids or []) if str(item).strip()]
        rec_text = clean_text(rec_text)
        target_item_ids = session.target_item_ids or ([session.target_item_id] if session.target_item_id else [])
        target_hit = bool(set(target_item_ids).intersection(recommended_item_ids))
        if not target_hit and any(item_id and item_id in rec_text.upper() for item_id in target_item_ids):
            target_hit = True
        if target_hit:
            session.accepted = True
            session.done = True
            return {"user_reply": "That sounds right for me. I'd go with that one.", "accepted": True, "done": True}
        if not recommended_item_ids and not grounded:
            return {
                "user_reply": "I still need a concrete product suggestion, not just a general description.",
                "accepted": False,
                "done": False,
            }
        return {"user_reply": "That doesn't feel like the right fit yet. Can you try another option?", "accepted": False, "done": False}


class LLMUserSimulator:
    """User simulator backed by an OpenAI-compatible chat completion endpoint."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        temperature: float = 0.2,
        max_tokens: int = 220,
        enable_thinking: bool | None = None,
        seed: int = 42,
    ):
        self.base_url = (base_url or os.environ.get("SIMREC_USER_LLM_BASE_URL", "")).rstrip("/")
        self.model = model or os.environ.get("SIMREC_USER_LLM_MODEL", "")
        self.api_key = api_key or os.environ.get("SIMREC_USER_LLM_API_KEY", "EMPTY")
        self.timeout = float(os.environ.get("SIMREC_USER_LLM_TIMEOUT", timeout))
        self.temperature = float(os.environ.get("SIMREC_USER_LLM_TEMPERATURE", temperature))
        self.max_tokens = int(os.environ.get("SIMREC_USER_LLM_MAX_TOKENS", max_tokens))
        self.max_retries = int(os.environ.get("SIMREC_USER_LLM_MAX_RETRIES", 6))
        self.retry_backoff_seconds = float(os.environ.get("SIMREC_USER_LLM_RETRY_BACKOFF_SECONDS", 10))
        self.max_retry_sleep_seconds = float(os.environ.get("SIMREC_USER_LLM_MAX_RETRY_SLEEP_SECONDS", 300))
        self.min_request_interval_seconds = max(
            0.0,
            float(os.environ.get("SIMREC_USER_LLM_MIN_REQUEST_INTERVAL_SECONDS", 0)),
        )
        self._last_request_at = 0.0
        fallback_on_error = _parse_optional_bool(os.environ.get("SIMREC_USER_LLM_FALLBACK_ON_ERROR", "true"))
        self.fallback_on_error = True if fallback_on_error is None else fallback_on_error
        env_enable_thinking = _parse_optional_bool(os.environ.get("SIMREC_USER_LLM_ENABLE_THINKING"))
        self.enable_thinking = env_enable_thinking if env_enable_thinking is not None else _parse_optional_bool(enable_thinking)
        self.seed = seed

    def start_episode(self, sample: dict[str, Any], initial_user_utterance: str | None = None) -> UserSession:
        del initial_user_utterance
        return UserSession(
            qid=str(sample.get("qid", "0")),
            target_item_id=str(sample.get("target_item_id", "")).upper(),
            reference_query=clean_text(sample.get("reference_query", "")),
            target_item_ids=_normalize_target_item_ids(sample),
        )

    def _require_client(self) -> None:
        if not self.base_url:
            raise RuntimeError("SIMREC_USER_LLM_BASE_URL is not configured")
        if not self.model:
            raise RuntimeError("SIMREC_USER_LLM_MODEL is not configured")

    def _rate_limit_sleep_seconds(self, exc: urllib.error.HTTPError, detail: str, attempt: int) -> float:
        retry_after = exc.headers.get("Retry-After")
        if retry_after:
            try:
                return min(self.max_retry_sleep_seconds, max(1.0, float(retry_after)))
            except ValueError:
                retry_at = email.utils.parsedate_to_datetime(retry_after)
                if retry_at is not None:
                    return min(self.max_retry_sleep_seconds, max(1.0, retry_at.timestamp() - time.time() + 1.0))

        reset_match = re.search(r"Limit resets at:\s*([^\n]+?)(?:\s*UTC)?(?:\n|$)", detail)
        if reset_match:
            reset_text = reset_match.group(1).strip()
            try:
                reset_at = email.utils.parsedate_to_datetime(f"{reset_text} UTC")
                if reset_at is not None:
                    return min(self.max_retry_sleep_seconds, max(1.0, reset_at.timestamp() - time.time() + 1.0))
            except (TypeError, ValueError):
                pass

        return min(self.max_retry_sleep_seconds, self.retry_backoff_seconds * (2**attempt))

    def _retry_sleep_seconds(self, attempt: int) -> float:
        return min(
            self.max_retry_sleep_seconds,
            self.retry_backoff_seconds * (2**attempt),
        )

    def _http_retry_sleep_seconds(self, exc: urllib.error.HTTPError, detail: str, attempt: int) -> float:
        if exc.code == 429:
            return self._rate_limit_sleep_seconds(exc, detail, attempt)
        return self._retry_sleep_seconds(attempt)

    def _throttle_before_request(self) -> None:
        if self.min_request_interval_seconds <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        sleep_seconds = self.min_request_interval_seconds - elapsed
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    def _chat_completion(self, messages: list[dict[str, str]]) -> str:
        self._require_client()
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        base_url = self.base_url.rstrip("/")
        if base_url.endswith("/v1"):
            endpoint = f"{base_url}/chat/completions"
        else:
            endpoint = f"{base_url}/v1/chat/completions"

        body = ""
        for attempt in range(self.max_retries + 1):
            self._throttle_before_request()
            request = urllib.request.Request(
                url=endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = response.read().decode("utf-8")
                self._last_request_at = time.monotonic()
                break
            except urllib.error.HTTPError as exc:
                self._last_request_at = time.monotonic()
                detail = exc.read().decode("utf-8", errors="ignore")
                if exc.code not in RETRYABLE_HTTP_STATUS_CODES or attempt >= self.max_retries:
                    raise RuntimeError(f"user simulator HTTP {exc.code}: {detail}") from exc
                sleep_seconds = self._http_retry_sleep_seconds(exc, detail, attempt)
                print(
                    f"user simulator HTTP {exc.code}; retrying in {sleep_seconds:.1f}s "
                    f"(attempt {attempt + 1}/{self.max_retries})",
                    flush=True,
                )
                time.sleep(sleep_seconds)
            except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
                self._last_request_at = time.monotonic()
                if attempt >= self.max_retries:
                    raise RuntimeError(f"user simulator connection error: {exc}") from exc
                sleep_seconds = self._retry_sleep_seconds(attempt)
                print(
                    f"user simulator connection error; retrying in {sleep_seconds:.1f}s "
                    f"(attempt {attempt + 1}/{self.max_retries}): {exc}",
                    flush=True,
                )
                time.sleep(sleep_seconds)

        payload = json.loads(body)
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError(f"user simulator returned no choices: {body}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(str(part.get("text", "")))
            if texts:
                return "\n".join(texts)
        raise RuntimeError(f"user simulator returned unexpected message payload: {message}")

    def generate_initial_user_utterance(self, session: UserSession) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are simulating the shopper in a recommendation dialogue. "
                    "State your shopping need clearly in natural English. "
                    "Keep it concise, realistic, and specific enough for a recommender to search."
                    " Never output, quote, or reveal any product or item ID, even if it appears in the context."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Reference query: {session.reference_query or 'N/A'}"
                ),
            },
        ]
        try:
            user_reply = mask_target_item_ids(self._chat_completion(messages), session.target_item_ids)
        except Exception as exc:
            if not self.fallback_on_error:
                raise
            user_reply = _build_fallback_initial_user_utterance(session.reference_query)
            print(
                f"user simulator failed to generate initial utterance; using fallback: {exc}",
                flush=True,
            )
        if not user_reply:
            raise ValueError("user simulator model returned empty initial user_reply")
        return user_reply

    def generate_user_reply(
        self,
        session: UserSession,
        assistant_text: str,
        dialogue: list[dict[str, Any]] | None = None,
        recommended_item_ids: list[str] | None = None,
        grounded: bool = False,
    ) -> str:
        recommended_ids = [
            str(item).upper()
            for item in (recommended_item_ids or [])
            if str(item).strip()
        ]
        # The simulator must judge the recommendation from the shopper need
        # and the visible product evidence.  Target IDs remain available to
        # the interaction/reward controller, but are never sent to this LLM.
        recommendation_context = (
            f"Assistant recommended item ids: {recommended_ids or 'N/A'}\n"
            f"Recommendation was grounded in recent tool results: {bool(grounded)}\n\n"
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are simulating the shopper in a recommendation dialogue. "
                    "Stay in character as a shopper with a clear product need. "
                    "Continue the conversation naturally in plain English. "
                    "If the assistant asks a question, answer it directly. "
                    "If the assistant gives a recommendation that does not fit the shopping need, do not accept it; "
                    "give corrective feedback with one or two concrete missing requirements so the recommender can search again. "
                    "If the assistant gives a recommendation that fits the shopping need, accept it briefly. "
                    "Do not use rigid labels or templates; sound like a normal shopper. "
                    "Do not output JSON, role labels, tags, hidden metadata, or any product/item ID. "
                    "Never reveal, quote, or repeat product or item IDs."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Your shopping need is based on:\n"
                    f"- Reference query: {session.reference_query or 'N/A'}\n\n"
                    f"{recommendation_context}"
                    f"Recent dialogue:\n{dialogue or []}\n\n"
                    f"Latest assistant message:\n{clean_text(assistant_text)}\n\n"
                    "Write the next shopper reply only."
                ),
            },
        ]
        try:
            session.turn_index += 1
            target_item_ids = session.target_item_ids or ([session.target_item_id] if session.target_item_id else [])
            user_reply = mask_target_item_ids(self._chat_completion(messages), target_item_ids)
            if not user_reply:
                raise ValueError("user simulator model returned empty reply")
            return user_reply
        except Exception as exc:
            if self.fallback_on_error:
                print(
                    f"user simulator failed to generate reply; using fallback: {exc}",
                    flush=True,
                )
                if clean_text(assistant_text).endswith("?"):
                    return "I mainly care that it matches what I'm looking for."
                if recommended_ids:
                    return "That does not quite match what I need. Please try another option that better fits the details I described."
                return "That doesn't feel like the right fit yet. Can you try another option?"
            raise RuntimeError(f"user simulator failed to generate reply: {exc}") from exc
