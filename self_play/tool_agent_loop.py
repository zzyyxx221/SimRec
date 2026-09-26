from __future__ import annotations

import json
import os
import re
from typing import Any

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

from SimRec.self_play.reward_constants import FORMAT_BONUS
from SimRec.self_play.tool_call_runtime import ensure_simrec_self_play_state, extract_structured_recommendations
from SimRec.self_play.tool_call_runtime import (
    append_trace,
    build_metrics,
    extract_recommended_item_ids,
    format_dialogue,
)

_ROLE_LEAK_MARKERS = ("<|im_start|>", "<|im_end|>", "\nuser\n", "\nassistant\n")
_PARTIAL_TOOL_NAME_RE = re.compile(r'"(?:name|tool_name)"\s*:\s*"([^"\n<}]*)', re.IGNORECASE)
_ARGUMENTS_RE = re.compile(r'"arguments"\s*:\s*(\{.*\})', re.DOTALL)
_TOOL_JSON_RE = re.compile(r'\{\s*"(?:name|tool_name)"\s*:\s*"(?:search_products|get_item_details|get_user_preference)"', re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
BROKEN_TOOL_CALL_RE = re.compile(r'^\s*\{\s*"name"\s*:\s*"[^"}]*\s*$', re.IGNORECASE)
ROLE_TOKEN_LEAK_RE = re.compile(r"<\|im_(start|end)\|>|<tool_call>|</tool_call>", re.IGNORECASE)
_HIDDEN_REASONING_RE = re.compile(r"<think>.*?(?:</think>|$)", re.IGNORECASE | re.DOTALL)
TOOL_COMMAND_LEAK_RE = re.compile(
    r"\b(?:do)?(?:getitemdetails|get_user_preference|search_products|getitem|search)\s*\(",
    re.IGNORECASE,
)
INCOMPLETE_REPLY_RE = re.compile(r"(?:\.{3}|\u2026)\s*$")
TURN_PENALTY = float(os.environ.get("SIMREC_TURN_PENALTY", "0.02"))

SELF_PLAY_USER_SYSTEM_PROMPT = (
    "/no_think\n"
    "You are role-playing the shopper in a recommendation dialogue.\n"
    "Use the private shopping need to produce only the next shopper utterance.\n"
    "Stay natural, concise, and consistent with the private need. Preserve concrete constraints.\n"
    "If the recommender asks a question, answer it directly.\n"
    "If the recommender suggests a product, compare it against the private need. "
    "Accept only if it satisfies the core constraints; otherwise say what is missing or wrong.\n"
    "Do not simply repeat your previous message after a bad recommendation.\n"
    "Do not output JSON, role labels, hidden metadata, or tool calls."
)


@register("simrec_self_play_tool_agent")
class SimRecToolAgentLoop(ToolAgentLoop):
    @staticmethod
    def _user_simulator_mode() -> str:
        mode = os.environ.get("SIMREC_USER_SIMULATOR_MODE", "self_play").strip().lower().replace("-", "_")
        return "api" if mode in {"api", "llm", "external_api"} else "self_play"

    @staticmethod
    def _train_self_play_user() -> bool:
        value = os.environ.get("SIMREC_TRAIN_SELF_PLAY_USER", "true")
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _require_second_recommendation() -> bool:
        """Keep a successful first recommendation open for one feedback turn."""
        value = os.environ.get("SIMREC_REQUIRE_SECOND_RECOMMENDATION", "false")
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _tool_obs_to_text(obs: Any) -> str:
        if isinstance(obs, dict):
            raw_obs = obs.get("obs")
            if isinstance(raw_obs, str):
                return raw_obs.strip()
            if raw_obs is not None:
                return str(raw_obs).strip()
        if isinstance(obs, str):
            return obs.strip()
        if obs is None:
            return ""
        return str(obs).strip()

    def _append_internal_tool_messages(self, agent_data) -> None:
        trace = agent_data.extra_fields.get("tool_interact_info")
        if not isinstance(trace, list) or not trace:
            return

        cursor = agent_data.extra_fields.get("tool_trace_message_cursor", 0)
        try:
            start_index = max(0, int(cursor))
        except (TypeError, ValueError):
            start_index = 0

        for entry in trace[start_index:]:
            if not isinstance(entry, dict) or entry.get("source") != "tool":
                continue

            action = entry.get("action")
            if isinstance(action, str) and action.strip():
                agent_data.messages.append({"role": "assistant_tool_call", "content": action.strip()})

            obs_text = self._tool_obs_to_text(entry.get("obs"))
            if obs_text:
                agent_data.messages.append({"role": "tool", "content": obs_text})

        agent_data.extra_fields["tool_trace_message_cursor"] = len(trace)

    @staticmethod
    def _publish_dialogue_snapshot(agent_data, *, user_feedback: str | None = None) -> None:
        """Expose the final visible dialogue and latest user feedback to the reward manager.

        ``tool_interact_info`` is also used as an audit trace, but its last
        interaction entry can be copied before the user simulator finishes.
        Keep the final conversation state in extra_fields so record creation
        does not depend on that mutation timing.
        """

        agent_data.extra_fields["final_dialogue"] = format_dialogue(agent_data.messages, max_turns=8)
        if user_feedback is not None:
            cleaned_feedback = str(user_feedback).strip()
            if cleaned_feedback:
                agent_data.extra_fields["last_user_feedback"] = cleaned_feedback

    async def _handle_pending_state(self, agent_data, sampling_params: dict[str, Any]) -> AgentState:
        if self._user_simulator_mode() == "api" and self.interaction_config_file and agent_data.interaction:
            has_user_message = any(message.get("role") == "user" for message in agent_data.messages)
            if not has_user_message and hasattr(agent_data.interaction, "get_initial_user_utterance"):
                opener = agent_data.interaction.get_initial_user_utterance(agent_data.request_id)
                opener = self._trim_role_leakage(str(opener))
                if opener:
                    agent_data.messages.append({"role": "user", "content": opener})
                    self._publish_dialogue_snapshot(agent_data)
        state = await super()._handle_pending_state(agent_data, sampling_params)
        if self._user_simulator_mode() == "api":
            return state
        has_user_message = any(message.get("role") == "user" for message in agent_data.messages)
        if not has_user_message:
            await self._generate_self_play_user_turn(agent_data, sampling_params, opening=True)
        return state

    def _build_generation_sampling_params(self, sampling_params: dict[str, Any]) -> dict[str, Any]:
        local_sampling = dict(sampling_params)
        # Keep enough of the shared response budget for a post-feedback
        # recommender turn.  Some checkpoints keep emitting punctuation after
        # a valid structured recommendation; without this per-turn cap the
        # first turn consumes the full episode budget and feedback cannot
        # affect retrieval or a revised recommendation.
        per_turn_cap = int(os.environ.get("SIMREC_MAX_ASSISTANT_TOKENS_PER_TURN", "0") or 0)
        if per_turn_cap > 0:
            if "max_tokens" in local_sampling:
                local_sampling["max_tokens"] = min(int(local_sampling["max_tokens"]), per_turn_cap)
            elif "max_new_tokens" in local_sampling:
                local_sampling["max_new_tokens"] = min(int(local_sampling["max_new_tokens"]), per_turn_cap)
            else:
                local_sampling["max_tokens"] = per_turn_cap
        stop_tokens = list(local_sampling.get("stop") or [])
        if self.tool_parser_name in {"hermes", "qwen3_coder"} and "</tool_call>" not in stop_tokens:
            stop_tokens.append("</tool_call>")
        if stop_tokens:
            local_sampling["stop"] = stop_tokens
        return local_sampling

    def _trim_role_leakage(self, text: str) -> str:
        candidate = text or ""
        for marker in _ROLE_LEAK_MARKERS:
            stop_at = candidate.find(marker)
            if stop_at >= 0:
                candidate = candidate[:stop_at]
        return candidate.strip()

    async def _decode_response_text(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        return await self.loop.run_in_executor(
            None,
            lambda token_ids=token_ids, skip_special_tokens=skip_special_tokens: self.tokenizer.decode(
                token_ids, skip_special_tokens=skip_special_tokens
            ),
        )

    @staticmethod
    def _clean_text(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    def _build_user_generation_messages(
        self,
        agent_data,
        *,
        opening: bool,
        assistant_text: str = "",
    ) -> list[dict[str, str]]:
        kwargs = agent_data.interaction_kwargs or {}
        dialogue = format_dialogue(agent_data.messages, max_turns=8)
        if opening:
            task = (
                "Write the shopper's first message only. Use the reference query nearly verbatim when available, "
                "without dropping concrete requirements.\n"
                f"Reference query: {kwargs.get('reference_query') or 'N/A'}"
            )
        else:
            task = (
                "Write the next shopper reply only.\n"
                f"Private reference query: {kwargs.get('reference_query') or 'N/A'}\n\n"
                f"Recent dialogue:\n{dialogue}\n\n"
                f"Latest recommender message:\n{self._clean_text(assistant_text)}"
            )
        return [
            {"role": "system", "content": SELF_PLAY_USER_SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]

    def _fallback_user_utterance(self, agent_data) -> str:
        kwargs = agent_data.interaction_kwargs or {}
        query = self._clean_text(kwargs.get("reference_query", ""))
        if query:
            return query if query.endswith((".", "!", "?")) else f"{query}."
        return "I'm looking for a product that fits my needs."

    async def _append_user_turn(self, agent_data, user_text: str) -> None:
        user_text = self._trim_role_leakage(self._clean_text(user_text))
        if not user_text:
            user_text = self._fallback_user_utterance(agent_data)
        add_messages = [{"role": "user", "content": user_text}]
        response_ids = await self.apply_chat_template(add_messages, remove_system_prompt=True)
        turn_start = len(agent_data.response_mask)
        agent_data.prompt_ids += response_ids
        train_user = self._train_self_play_user()
        agent_data.response_mask += ([1] if train_user else [0]) * len(response_ids)
        if not train_user and agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        turn_spans = agent_data.extra_fields.setdefault("turn_token_spans", [])
        turn_span_index = len(turn_spans)
        turn_spans.append(
            {
                "turn_index": agent_data.user_turns,
                "role": "user",
                "start": turn_start,
                "end": len(agent_data.response_mask),
                # Keep the span metadata consistent with response_mask.  The
                # reward manager uses this field to decide whether a self-play
                # user turn receives the shared terminal reward; API user
                # turns remain non-trainable because train_user is False.
                "response_mask": int(train_user),
            }
        )
        agent_data.extra_fields["current_turn_span_index"] = turn_span_index
        agent_data.messages.extend(add_messages)

    async def _generate_self_play_user_turn(
        self,
        agent_data,
        sampling_params: dict[str, Any],
        *,
        opening: bool = False,
        assistant_text: str = "",
    ) -> str:
        fixed_feedback = self._clean_text(os.environ.get("SIMREC_FIXED_USER_FEEDBACK", ""))
        if fixed_feedback and not opening:
            await self._append_user_turn(agent_data, fixed_feedback)
            self._publish_dialogue_snapshot(agent_data, user_feedback=fixed_feedback)
            return fixed_feedback

        user_sampling = dict(sampling_params)
        user_sampling["max_tokens"] = int(
            (agent_data.interaction_kwargs or {}).get("self_play_user_max_tokens")
            or user_sampling.get("max_tokens")
            or 160
        )
        user_sampling["temperature"] = float(
            (agent_data.interaction_kwargs or {}).get("self_play_user_temperature")
            or user_sampling.get("temperature")
            or 0.7
        )
        prompt_ids = await self.apply_chat_template(
            self._build_user_generation_messages(agent_data, opening=opening, assistant_text=assistant_text)
        )
        with simple_timer("generate_self_play_user", agent_data.metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=f"{agent_data.request_id}-user-{agent_data.user_turns + 1}",
                prompt_ids=prompt_ids,
                sampling_params=user_sampling,
                image_data=None,
                video_data=None,
            )
        user_text = self._trim_role_leakage(await self._decode_response_text(output.token_ids, skip_special_tokens=True))
        await self._append_user_turn(agent_data, user_text)
        if not opening:
            self._publish_dialogue_snapshot(agent_data, user_feedback=user_text)
        else:
            self._publish_dialogue_snapshot(agent_data)
        return user_text

    def _tool_allows_empty_arguments(self, tool_name: str) -> bool:
        tool = self.tools.get(tool_name)
        if tool is None or tool.tool_schema.function is None:
            return False
        required = tool.tool_schema.function.parameters.required
        return not required

    def _parse_tool_call_candidate(self, candidate: str) -> FunctionCall | None:
        trimmed = self._trim_role_leakage(candidate)
        if not trimmed:
            return None

        try:
            payload = json.loads(trimmed)
        except Exception:
            payload = None

        if isinstance(payload, dict):
            tool_name = str(payload.get("name") or payload.get("tool_name") or "").strip()
            arguments = payload.get("arguments", {})
            if tool_name in self.tools and isinstance(arguments, dict):
                return FunctionCall(name=tool_name, arguments=json.dumps(arguments, ensure_ascii=False))

        name_match = _PARTIAL_TOOL_NAME_RE.search(trimmed)
        if not name_match:
            return None
        tool_name = name_match.group(1).strip()
        if tool_name not in self.tools:
            return None

        arguments_match = _ARGUMENTS_RE.search(trimmed)
        if arguments_match:
            try:
                arguments = json.loads(arguments_match.group(1))
            except Exception:
                arguments = None
            if isinstance(arguments, dict):
                return FunctionCall(name=tool_name, arguments=json.dumps(arguments, ensure_ascii=False))

        if self._tool_allows_empty_arguments(tool_name):
            return FunctionCall(name=tool_name, arguments="{}")
        return None

    @staticmethod
    def _json_object_candidates(text: str) -> list[str]:
        candidates: list[str] = []
        start: int | None = None
        depth = 0
        in_string = False
        escaped = False
        for idx, char in enumerate(text or ""):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == "{":
                if depth == 0:
                    start = idx
                depth += 1
            elif char == "}" and depth:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start : idx + 1])
                    start = None
        return candidates

    def _recover_tool_calls(self, raw_text: str) -> list[FunctionCall]:
        candidates: list[str] = []
        trimmed = self._trim_role_leakage(raw_text)
        if trimmed:
            candidates.append(trimmed)

        if "<tool_call>" in raw_text:
            fragment = raw_text.split("<tool_call>", 1)[1]
            fragment = fragment.split("</tool_call>", 1)[0]
            fragment = self._trim_role_leakage(fragment)
            if fragment:
                candidates.append(fragment)
        candidates.extend(self._json_object_candidates(raw_text))

        seen: set[tuple[str, str]] = set()
        recovered: list[FunctionCall] = []
        for candidate in candidates:
            tool_call = self._parse_tool_call_candidate(candidate)
            if tool_call is None:
                continue
            key = (tool_call.name, tool_call.arguments)
            if key in seen:
                continue
            seen.add(key)
            recovered.append(tool_call)
        return recovered

    @staticmethod
    def _tool_arguments(tool_call: FunctionCall) -> dict[str, Any]:
        try:
            payload = json.loads(tool_call.arguments or "{}")
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _select_tool_calls(self, agent_data, tool_calls: list[FunctionCall]) -> tuple[list[FunctionCall], bool]:
        if not tool_calls:
            return [], False

        state = ensure_simrec_self_play_state(agent_data.extra_fields)
        has_results = bool(state.get("last_search_results"))
        must_recommend = bool(state.get("must_recommend_next"))
        details_count = int(state.get("details_invoked_count", 0) or 0)
        recommendation_count = int(state.get("recommendation_count", 0) or 0)

        for tool_call in tool_calls:
            if tool_call.name == "get_user_preference":
                continue
            if must_recommend and has_results and recommendation_count <= 0:
                if tool_call.name == "get_item_details" and details_count <= 0:
                    return [tool_call], False
                return [], True
            if tool_call.name == "search_products":
                if recommendation_count > 0:
                    revised_search_count = int(state.get("revised_search_invoked_count", 0) or 0)
                    if revised_search_count >= 1:
                        return [], True
                return [tool_call], False
            if tool_call.name == "get_item_details":
                if has_results:
                    return [tool_call], False
                continue
        return [], bool(must_recommend and has_results and recommendation_count <= 0)

    @staticmethod
    def _compact_sentence(text: str, limit: int = 260) -> str:
        cleaned = " ".join(str(text or "").split())
        if not cleaned:
            return ""
        sentence = _SENTENCE_SPLIT_RE.split(cleaned)[0].strip()
        if len(sentence) > limit:
            sentence = sentence[: limit - 3].rstrip() + "..."
        return sentence

    def _build_forced_recommendation(self, agent_data) -> str | None:
        state = ensure_simrec_self_play_state(agent_data.extra_fields)
        results = state.get("last_search_results")
        if not isinstance(results, list) or not results:
            return None
        viewed_id = str(state.get("last_viewed_item_id") or "").upper()
        item = None
        if viewed_id:
            item = next((row for row in results if str(row.get("id", "")).upper() == viewed_id), None)
        if item is None:
            item = results[0]
        item_id = str(item.get("id", "")).upper().strip()
        if not item_id:
            return None
        content = self._compact_sentence(item.get("content", ""))
        if not content:
            content = "This candidate is the strongest match from the latest retrieval results."
        description = (
            f"{content} It is recommended because it is the strongest available match from the latest search evidence. "
            "The choice is based only on retrieved catalog information."
        )
        return f"recommended_item_id: {item_id}\nrecommended_item_description: {description}"

    def _append_assistant_turn(self, agent_data, assistant_message: str, *, forced: bool = False) -> None:
        agent_data.assistant_turns += 1
        turn_end = len(agent_data.response_mask)
        turn_start = turn_end - len(agent_data.response_ids)
        turn_spans = agent_data.extra_fields.setdefault("turn_token_spans", [])
        turn_span_index = len(turn_spans)
        turn_spans.append(
            {
                "turn_index": agent_data.assistant_turns,
                "role": "recommender",
                "start": turn_start,
                "end": turn_end,
            }
        )
        agent_data.extra_fields["current_turn_span_index"] = turn_span_index
        agent_data.extra_fields["current_assistant_forced"] = bool(forced)
        if forced:
            agent_data.extra_fields["forced_recommendation"] = True
            agent_data.extra_fields["action_extraction"] = "forced_recommendation"
        if self.interaction_config_file and assistant_message:
            agent_data.messages.append({"role": "assistant", "content": assistant_message})

    async def _handle_generating_state(
        self, agent_data, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        max_total_tokens = self.prompt_length + self.response_length
        generation_sampling_params = self._build_generation_sampling_params(sampling_params)
        agent_data.extra_fields["_last_sampling_params"] = dict(sampling_params)

        # Stop before calling the rollout server if the full sequence budget is already exhausted.
        if not ignore_termination and len(agent_data.prompt_ids) >= max_total_tokens:
            return AgentState.TERMINATED

        with simple_timer("generate_sequences", agent_data.metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=generation_sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        if not agent_data.extra_fields:
            agent_data.extra_fields.update(output.extra_fields)
        else:
            max_global_steps = output.extra_fields.get("max_global_steps", None)
            if max_global_steps:
                agent_data.extra_fields["max_global_steps"] = max_global_steps

        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        # A generation may reach the response budget after it has already
        # produced a valid recommendation.  Do not return yet: the
        # recommendation must still be parsed and passed to the interaction
        # layer so that the user feedback is recorded.  The flag below stops
        # the episode immediately after that observation is appended.
        response_limit_reached = bool(
            not ignore_termination and len(agent_data.response_mask) >= self.response_length
        )

        tools = [tool.tool_schema for tool in self.tools.values()]
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, tools)
        raw_response = await self._decode_response_text(agent_data.response_ids, skip_special_tokens=False)
        if _HIDDEN_REASONING_RE.search(raw_response or ""):
            agent_data.extra_fields["hidden_reasoning_detected"] = True
        if not agent_data.tool_calls:
            agent_data.tool_calls = self._recover_tool_calls(raw_response)

        if agent_data.tool_calls:
            selected_tool_calls, should_force_recommend = self._select_tool_calls(agent_data, agent_data.tool_calls)
            if selected_tool_calls:
                agent_data.tool_calls = selected_tool_calls[:1]
                agent_data.extra_fields["current_turn_span_index"] = None
                agent_data.extra_fields["recovered_tool_raw_action"] = raw_response
                agent_data.extra_fields["recovered_tool_action_extraction"] = "native_or_recovered_tool_call"
                return AgentState.PROCESSING_TOOLS
            if should_force_recommend:
                forced_message = self._build_forced_recommendation(agent_data)
                if forced_message:
                    self._append_assistant_turn(agent_data, forced_message, forced=True)
                    agent_data.extra_fields["pending_terminate_after_observation"] = bool(
                        self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns
                    )
                    return AgentState.INTERACTING

        if self.interaction_config_file:
            assistant_message = self._trim_role_leakage(
                await self._decode_response_text(agent_data.response_ids, skip_special_tokens=True)
            )
            pending_last_turn = bool(self.max_assistant_turns and agent_data.assistant_turns + 1 >= self.max_assistant_turns)
            state = ensure_simrec_self_play_state(agent_data.extra_fields)
            should_force = (
                pending_last_turn
                and not extract_structured_recommendations(assistant_message)
                and bool(state.get("last_search_results"))
            )
            if should_force or (_TOOL_JSON_RE.search(assistant_message or "") and bool(state.get("last_search_results"))):
                forced_message = self._build_forced_recommendation(agent_data)
                if forced_message:
                    assistant_message = forced_message
                    self._append_assistant_turn(agent_data, assistant_message, forced=True)
                else:
                    self._append_assistant_turn(agent_data, assistant_message)
            else:
                self._append_assistant_turn(agent_data, assistant_message)
        else:
            self._append_assistant_turn(agent_data, "")

        pending_terminate = bool(
            response_limit_reached
            or (self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns)
        )
        agent_data.extra_fields["pending_terminate_after_observation"] = pending_terminate

        if self.interaction_config_file:
            return AgentState.INTERACTING
        return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data) -> AgentState:
        next_state = await super()._handle_processing_tools_state(agent_data)
        self._append_internal_tool_messages(agent_data)
        agent_data.extra_fields.pop("recovered_tool_raw_action", None)
        agent_data.extra_fields.pop("recovered_tool_action_extraction", None)
        if agent_data.extra_fields.get("pending_terminate_after_observation"):
            return AgentState.TERMINATED
        return next_state

    @staticmethod
    def _is_invalid_assistant_reply(text: str) -> bool:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        if not cleaned:
            return True
        lowered = cleaned.lower()
        if ROLE_TOKEN_LEAK_RE.search(text):
            return True
        if TOOL_COMMAND_LEAK_RE.search(cleaned):
            return True
        if _TOOL_JSON_RE.search(cleaned):
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

    def _score_recommender_turn(
        self,
        agent_data,
        assistant_text: str,
        *,
        force_user_feedback: bool = False,
    ) -> tuple[bool, float]:
        kwargs = agent_data.interaction_kwargs or {}
        extra_fields = agent_data.extra_fields
        state = ensure_simrec_self_play_state(extra_fields)
        dialogue = format_dialogue(agent_data.messages, max_turns=8)
        assistant_text = self._clean_text(assistant_text)
        turn_penalty = TURN_PENALTY * max(0, int(getattr(agent_data, "assistant_turns", 0) or 0) - 1)

        if self._is_invalid_assistant_reply(assistant_text):
            metrics = build_metrics(
                action_type="invalid_reply",
                success=False,
                valid_action=False,
                turn_penalty=turn_penalty,
            )
            append_trace(
                extra_fields,
                source="interaction",
                action=assistant_text,
                obs={"obs": "", "dialogue": dialogue},
                reward=0.0,
                metrics=metrics,
                done=False,
                valid_action=False,
            )
            return False, 0.0

        if assistant_text.endswith("?"):
            metrics = build_metrics(
                action_type="clarify",
                success=False,
                valid_action=True,
                turn_penalty=turn_penalty,
            )
            append_trace(
                extra_fields,
                source="interaction",
                action=assistant_text,
                obs={"obs": "", "dialogue": dialogue},
                reward=0.0,
                metrics=metrics,
                done=False,
                valid_action=True,
            )
            return False, 0.0

        structured_recommendations = extract_structured_recommendations(assistant_text)
        if not structured_recommendations:
            metrics = build_metrics(
                action_type="invalid_reply",
                success=False,
                valid_action=False,
                turn_penalty=turn_penalty,
            )
            append_trace(
                extra_fields,
                source="interaction",
                action=assistant_text,
                obs={"obs": "", "dialogue": dialogue},
                reward=0.0,
                metrics=metrics,
                done=False,
                valid_action=False,
            )
            return False, 0.0

        candidate_ids = extract_recommended_item_ids(assistant_text)
        recent_ids = {str(item_id).upper() for item_id in state.get("last_search_result_ids", [])}
        if state.get("last_viewed_item_id"):
            recent_ids.add(str(state["last_viewed_item_id"]).upper())
        grounded = bool(candidate_ids and any(item_id in recent_ids for item_id in candidate_ids))

        raw_targets = kwargs.get("target_item_ids") or kwargs.get("target_ids") or [kwargs.get("target_item_id")]
        if not isinstance(raw_targets, (list, tuple, set)):
            raw_targets = [raw_targets]
        target_item_ids = {str(item or "").strip().upper() for item in raw_targets if str(item or "").strip()}
        success = bool(target_item_ids and any(item_id in target_item_ids for item_id in candidate_ids))
        recommendation_count = int(state.get("recommendation_count", 0) or 0) + 1
        state["recommendation_count"] = recommendation_count
        state["recommendation_made"] = True
        state["must_recommend_next"] = False
        # Keep the normal policy unchanged unless explicitly requested.  For
        # one-shot-vs-best evaluation, require one post-recommendation user
        # turn even when the first recommendation already hits a target.
        # This makes a second search possible and gives the two metrics a
        # meaningful distinction.
        if self._require_second_recommendation():
            would_terminate = recommendation_count >= 2
        else:
            would_terminate = recommendation_count >= 2 or success
        should_terminate = would_terminate and not force_user_feedback
        extra_fields["recommendation_termination_pending"] = would_terminate
        forced_recommendation = bool(extra_fields.pop("current_assistant_forced", False))
        effective_success = success and not forced_recommendation
        format_reward = 0.0 if forced_recommendation else FORMAT_BONUS

        reward = format_reward

        metrics = build_metrics(
            action_type="recommend",
            success=effective_success,
            valid_action=True,
            format_reward=format_reward,
            turn_penalty=turn_penalty,
            grounded=grounded,
            search_hit_at_k=bool(state.get("search_hit_at_k")),
            retrieval_ndcg_at_10=float(state.get("retrieval_ndcg_at_10", 0.0) or 0.0),
            forced_recommendation=forced_recommendation,
        )
        append_trace(
            extra_fields,
            source="interaction",
            action=assistant_text,
            obs={"obs": "", "dialogue": dialogue},
            reward=reward,
            metrics=metrics,
            done=should_terminate,
            valid_action=True,
        )
        return should_terminate, reward

    async def _handle_interacting_state(self, agent_data) -> AgentState:
        if self._user_simulator_mode() == "api":
            return await self._handle_api_interacting_state(agent_data)

        assistant_text = ""
        for message in reversed(agent_data.messages):
            if message.get("role") in {"assistant", "recommender", "rec"}:
                assistant_text = self._clean_text(message.get("content", ""))
                break
        # Publish after the recommender turn is appended, including episodes
        # that terminate immediately without another user turn.
        self._publish_dialogue_snapshot(agent_data)
        fixed_feedback = self._clean_text(os.environ.get("SIMREC_FIXED_USER_FEEDBACK", ""))
        can_emit_fixed_feedback = bool(fixed_feedback) and not bool(
            agent_data.extra_fields.get("fixed_feedback_emitted")
        ) and (
            not self.max_user_turns or agent_data.user_turns < self.max_user_turns
        )
        should_terminate_sequence, reward = self._score_recommender_turn(
            agent_data,
            assistant_text,
            force_user_feedback=can_emit_fixed_feedback,
        )
        if can_emit_fixed_feedback:
            # Emit exactly one negative observation and leave room for the
            # subsequent search/recommendation turn. The second recommendation
            # follows the ordinary recommendation-count termination rule.
            agent_data.extra_fields["fixed_feedback_emitted"] = True
        if reward is not None:
            agent_data.turn_scores.append(reward)

        agent_data.extra_fields["current_turn_span_index"] = None

        # Fixed feedback is an explicit final observation.  It must be
        # appended before honoring a successful recommendation/max-turn
        # termination flag.
        if should_terminate_sequence and not can_emit_fixed_feedback:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED
        user_reply = await self._generate_self_play_user_turn(
            agent_data,
            sampling_params=agent_data.extra_fields.get("_last_sampling_params", {}),
            assistant_text=assistant_text,
        )
        trace = agent_data.extra_fields.get("tool_interact_info")
        if isinstance(trace, list) and trace:
            last_entry = trace[-1]
            if isinstance(last_entry, dict) and last_entry.get("source") == "interaction":
                obs = last_entry.get("obs")
                if not isinstance(obs, dict):
                    obs = {}
                    last_entry["obs"] = obs
                obs["obs"] = user_reply
                obs["dialogue"] = format_dialogue(agent_data.messages, max_turns=8)
        self._publish_dialogue_snapshot(agent_data, user_feedback=user_reply)
        if should_terminate_sequence or (
            agent_data.extra_fields.get("pending_terminate_after_observation") and not can_emit_fixed_feedback
        ):
            return AgentState.TERMINATED
        return AgentState.GENERATING

    async def _handle_api_interacting_state(self, agent_data) -> AgentState:
        should_terminate_sequence, interaction_responses, reward, additional_data = await agent_data.interaction.generate_response(
            agent_data.request_id,
            agent_data.messages,
            **agent_data.interaction_kwargs,
            extra_fields=agent_data.extra_fields,
        )
        add_messages: list[dict[str, Any]] = []
        if interaction_responses:
            agent_data.user_turns += 1
            add_messages = [{"role": "user", "content": interaction_responses}]
            agent_data.messages.extend(add_messages)

        if reward is not None:
            agent_data.turn_scores.append(reward)

        if isinstance(additional_data, dict):
            for key, value in additional_data.items():
                agent_data.extra_fields[key] = value

        if add_messages:
            response_ids = await self.apply_chat_template(add_messages, remove_system_prompt=True)
            agent_data.prompt_ids += response_ids
            agent_data.response_mask += [0] * len(response_ids)
            if agent_data.response_logprobs:
                agent_data.response_logprobs += [0.0] * len(response_ids)
            self._publish_dialogue_snapshot(agent_data, user_feedback=interaction_responses)
        else:
            self._publish_dialogue_snapshot(agent_data)
        agent_data.extra_fields["current_turn_span_index"] = None

        if should_terminate_sequence or agent_data.extra_fields.get("pending_terminate_after_observation"):
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED
        return AgentState.GENERATING
