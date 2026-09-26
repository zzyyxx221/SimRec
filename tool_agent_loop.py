from __future__ import annotations

import json
import os
import random
import re
from typing import Any

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

from SimRec.skill_memory import SkillMemory, get_skill_memory_from_env, infer_phase
from SimRec.tool_call_runtime import (
    classify_assistant_action,
    ensure_simrec_state,
    extract_structured_recommendations,
)

_ROLE_LEAK_MARKERS = ("<|im_start|>", "<|im_end|>", "\nuser\n", "\nassistant\n")
_PARTIAL_TOOL_NAME_RE = re.compile(r'"(?:name|tool_name)"\s*:\s*"([^"\n<}]*)', re.IGNORECASE)
_ARGUMENTS_RE = re.compile(r'"arguments"\s*:\s*(\{.*\})', re.DOTALL)
_TOOL_JSON_RE = re.compile(
    r'\{\s*"(?:name|tool_name)"\s*:\s*"(?:search_products|get_item_details|get_user_preference|get_usr_preference)"',
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_HIDDEN_REASONING_RE = re.compile(r"<think>.*?(?:</think>|$)", re.IGNORECASE | re.DOTALL)
_TOOL_NAME_ALIASES = {
    "get_usr_preference": "get_user_preference",
}
FINAL_RECOMMENDATION_RESERVE_TOKENS = max(
    0, int(os.environ.get("SIMREC_FINAL_RECOMMENDATION_RESERVE_TOKENS", "160"))
)


@register("simrec_tool_agent")
class SimRecToolAgentLoop(ToolAgentLoop):
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> Any:
        output = await super().run(sampling_params, **kwargs)
        extra_info = kwargs.get("extra_info") if isinstance(kwargs.get("extra_info"), dict) else {}
        if "is_validate" in extra_info:
            output.extra_fields["validate"] = bool(extra_info.get("is_validate"))
        return output

    async def _encode_plain_text(self, text: str) -> list[int]:
        return await self.loop.run_in_executor(
            None,
            lambda text=text: self.tokenizer.encode(text, add_special_tokens=False),
        )

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

    async def _handle_pending_state(self, agent_data, sampling_params: dict[str, Any]) -> AgentState:
        if self.interaction_config_file and agent_data.interaction:
            has_user_message = any(message.get("role") == "user" for message in agent_data.messages)
            if not has_user_message and hasattr(agent_data.interaction, "get_initial_user_utterance"):
                opener = agent_data.interaction.get_initial_user_utterance(agent_data.request_id)
                opener = self._trim_role_leakage(str(opener))
                if opener:
                    agent_data.messages.append({"role": "user", "content": opener})
        return await super()._handle_pending_state(agent_data, sampling_params)

    def _build_generation_sampling_params(self, sampling_params: dict[str, Any]) -> dict[str, Any]:
        local_sampling = dict(sampling_params)
        stop_tokens = list(local_sampling.get("stop") or [])
        if self.tool_parser_name in {"hermes", "qwen3_coder"} and "</tool_call>" not in stop_tokens:
            stop_tokens.append("</tool_call>")
        if stop_tokens:
            local_sampling["stop"] = stop_tokens
        return local_sampling

    async def _append_skill_context_if_enabled(self, agent_data, *, max_total_tokens: int) -> None:
        memory = get_skill_memory_from_env()
        if memory is None:
            return
        state = ensure_simrec_state(agent_data.extra_fields)
        retrieved = memory.retrieve(messages=list(agent_data.messages), state=state)
        if not retrieved:
            return
        dropout_rate = min(1.0, max(0.0, float(os.environ.get("SIMREC_SKILL_DROPOUT_RATE", "0.0"))))
        injected = random.random() >= dropout_rate
        phase = infer_phase(list(agent_data.messages), state)
        usage = agent_data.extra_fields.setdefault("skill_usage", [])
        if not isinstance(usage, list):
            usage = []
            agent_data.extra_fields["skill_usage"] = usage
        usage.extend(
            SkillMemory.usage_payload(
                retrieved,
                phase=phase,
                turn_index=int(getattr(agent_data, "assistant_turns", 0) or 0) + 1,
                observed_category=str(state.get("observed_category") or ""),
                library_version=memory.library_version,
                injected=injected,
                injection_propensity=1.0 - dropout_rate,
            )
        )
        if not injected:
            return
        hint = SkillMemory.render(retrieved)
        if not hint:
            return
        text = (
            "\n\nLearned skills context for the next assistant turn. Use it as private guidance; "
            "do not quote or mention it to the shopper.\n"
            f"{hint}\n"
            "End learned skills context.\n"
        )
        token_ids = await self._encode_plain_text(text)
        reserved_tokens = max(FINAL_RECOMMENDATION_RESERVE_TOKENS, 16)
        remaining = max_total_tokens - len(agent_data.prompt_ids) - reserved_tokens
        if remaining <= 0:
            return
        if len(token_ids) > remaining:
            token_ids = token_ids[:remaining]
        if not token_ids:
            return
        agent_data.prompt_ids += token_ids
        agent_data.response_mask += [0] * len(token_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(token_ids)

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
            tool_name = _TOOL_NAME_ALIASES.get(tool_name.lower(), tool_name)
            arguments = payload.get("arguments", {})
            if tool_name in self.tools and isinstance(arguments, dict):
                return FunctionCall(name=tool_name, arguments=json.dumps(arguments, ensure_ascii=False))

        name_match = _PARTIAL_TOOL_NAME_RE.search(trimmed)
        if not name_match:
            return None
        tool_name = name_match.group(1).strip()
        tool_name = _TOOL_NAME_ALIASES.get(tool_name.lower(), tool_name)
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

        state = ensure_simrec_state(agent_data.extra_fields)
        has_results = bool(state.get("last_search_results"))
        must_recommend = bool(state.get("must_recommend_next"))
        details_count = int(state.get("details_invoked_count", 0) or 0)
        preference_count = int(state.get("user_preference_invoked_count", 0) or 0)
        recommendation_count = int(state.get("recommendation_count", 0) or 0)
        retrieval_hit = float(state.get("retrieval_ndcg_at_10", 0.0) or 0.0) > 0.0
        available_tool_names = set(self.tools.keys())

        for tool_call in tool_calls:
            if tool_call.name not in available_tool_names:
                continue
            if tool_call.name == "get_user_preference":
                if preference_count <= 0:
                    return [tool_call], False
                continue
            if must_recommend and has_results and recommendation_count <= 0:
                if tool_call.name == "search_products" and not retrieval_hit:
                    search_count = int(state.get("search_invoked_count", 0) or 0)
                    if search_count <= 1:
                        return [tool_call], False
                if tool_call.name == "get_item_details" and details_count <= 0:
                    return [tool_call], False
                return [], True
            if tool_call.name == "search_products":
                if recommendation_count > 0:
                    revised_search_count = int(state.get("revised_search_invoked_count", 0) or 0)
                    arguments = self._tool_arguments(tool_call)
                    query = " ".join(str(arguments.get("query") or "").lower().split())
                    previous_queries = {
                        " ".join(str(item or "").lower().split())
                        for item in state.get("search_history", [])
                        if str(item or "").strip()
                    }
                    if revised_search_count >= 1 or (query and query in previous_queries):
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
        state = ensure_simrec_state(agent_data.extra_fields)
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

    def _force_recommend_if_possible(self, agent_data, *, terminate_after_observation: bool = False) -> bool:
        if not self.interaction_config_file:
            return False
        state = ensure_simrec_state(agent_data.extra_fields)
        if not state.get("last_search_results"):
            return False
        forced_message = self._build_forced_recommendation(agent_data)
        if not forced_message:
            return False
        self._append_assistant_turn(agent_data, forced_message, forced=True)
        agent_data.extra_fields["pending_terminate_after_observation"] = bool(
            terminate_after_observation
            or (self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns)
        )
        return True

    def _append_assistant_turn(self, agent_data, assistant_message: str, *, forced: bool = False) -> None:
        agent_data.assistant_turns += 1
        turn_end = len(agent_data.response_mask)
        turn_start = turn_end - len(agent_data.response_ids)
        turn_spans = agent_data.extra_fields.setdefault("turn_token_spans", [])
        turn_span_index = len(turn_spans)
        turn_spans.append(
            {
                "turn_index": agent_data.assistant_turns,
                "start": turn_start,
                "end": turn_end,
            }
        )
        agent_data.extra_fields["current_turn_span_index"] = turn_span_index
        action_type = classify_assistant_action(assistant_message)
        if action_type in {"clarify", "recommend"}:
            state = ensure_simrec_state(agent_data.extra_fields)
            state["last_action_type"] = action_type
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
        state = ensure_simrec_state(agent_data.extra_fields)
        has_results = bool(state.get("last_search_results"))
        must_recommend = bool(state.get("must_recommend_next"))
        remaining_tokens = max_total_tokens - len(agent_data.prompt_ids)

        # Preserve a terminal grounded recommendation instead of silently ending
        # after tool observations consume the sequence budget.
        should_force_for_budget = (
            not ignore_termination
            and has_results
            and must_recommend
            and remaining_tokens <= FINAL_RECOMMENDATION_RESERVE_TOKENS
        )
        if should_force_for_budget and self._force_recommend_if_possible(
            agent_data, terminate_after_observation=True
        ):
            return AgentState.INTERACTING

        if not ignore_termination and remaining_tokens <= 0:
            return AgentState.TERMINATED

        await self._append_skill_context_if_enabled(agent_data, max_total_tokens=max_total_tokens)

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
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        raw_response = await self._decode_response_text(agent_data.response_ids, skip_special_tokens=False)
        if _HIDDEN_REASONING_RE.search(raw_response or ""):
            agent_data.extra_fields["hidden_reasoning_detected"] = True

        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            if self._force_recommend_if_possible(agent_data, terminate_after_observation=True):
                return AgentState.INTERACTING
            return AgentState.TERMINATED

        tools = [tool.tool_schema for tool in self.tools.values()]
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, tools)
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
                if self._force_recommend_if_possible(agent_data):
                    return AgentState.INTERACTING

        if self.interaction_config_file:
            assistant_message = self._trim_role_leakage(
                await self._decode_response_text(agent_data.response_ids, skip_special_tokens=True)
            )
            pending_last_turn = bool(self.max_assistant_turns and agent_data.assistant_turns + 1 >= self.max_assistant_turns)
            state = ensure_simrec_state(agent_data.extra_fields)
            has_structured_recommendation = bool(extract_structured_recommendations(assistant_message))
            missing_recommendation_after_search = (
                bool(state.get("must_recommend_next"))
                and bool(state.get("last_search_results"))
                and not has_structured_recommendation
            )
            should_force = (
                missing_recommendation_after_search
                or (
                    pending_last_turn
                    and not has_structured_recommendation
                    and bool(state.get("last_search_results"))
                )
            )
            if should_force or (_TOOL_JSON_RE.search(assistant_message or "") and bool(state.get("last_search_results"))):
                if not self._force_recommend_if_possible(
                    agent_data, terminate_after_observation=missing_recommendation_after_search
                ):
                    self._append_assistant_turn(agent_data, assistant_message)
            else:
                self._append_assistant_turn(agent_data, assistant_message)
        else:
            self._append_assistant_turn(agent_data, "")

        pending_terminate = bool(self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns)
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

    async def _handle_interacting_state(self, agent_data) -> AgentState:
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
                if key == "tool_interact_info":
                    agent_data.extra_fields[key] = value
                elif key not in agent_data.extra_fields:
                    agent_data.extra_fields[key] = value
                else:
                    agent_data.extra_fields[key] = value

        if add_messages:
            response_ids = await self.apply_chat_template(add_messages, remove_system_prompt=True)
            agent_data.prompt_ids += response_ids
            agent_data.response_mask += [0] * len(response_ids)
            if agent_data.response_logprobs:
                agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.extra_fields["current_turn_span_index"] = None

        if should_terminate_sequence or agent_data.extra_fields.get("pending_terminate_after_observation"):
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED
        return AgentState.GENERATING
