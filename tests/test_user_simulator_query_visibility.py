from __future__ import annotations

import unittest
from pathlib import Path

from SimRec.simulator import LLMUserSimulator


class UserSimulatorQueryVisibilityTest(unittest.TestCase):
    def test_external_user_simulator_never_receives_review(self) -> None:
        simulator = LLMUserSimulator(base_url="http://unused", model="unused", api_key="unused")
        captured_messages: list[dict[str, str]] = []
        secret_review = "REVIEW_ONLY_SECRET: blue leather with a hidden pocket"

        def fake_chat_completion(messages):
            captured_messages.extend(messages)
            return "I need a travel bag."

        simulator._chat_completion = fake_chat_completion
        session = simulator.start_episode(
            {
                "qid": "q1",
                "target_item_id": "B00TARGET",
                "reference_query": "I need a travel bag.",
                "reference_review": secret_review,
            }
        )

        self.assertFalse(hasattr(session, "reference_review"))
        self.assertEqual(simulator.generate_initial_user_utterance(session), "I need a travel bag.")
        simulator.generate_user_reply(session, "Please consider this option.")

        visible_context = "\n".join(str(message["content"]) for message in captured_messages)
        self.assertIn("I need a travel bag.", visible_context)
        self.assertNotIn(secret_review, visible_context)
        self.assertNotIn("Reference review", visible_context)

    def test_fallback_uses_query_not_review(self) -> None:
        simulator = LLMUserSimulator(base_url="http://unused", model="unused", api_key="unused")
        simulator._chat_completion = lambda _messages: (_ for _ in ()).throw(RuntimeError("offline"))
        session = simulator.start_episode(
            {
                "reference_query": "I need a compact charger",
                "reference_review": "REVIEW_ONLY_SECRET",
            }
        )
        self.assertEqual(simulator.generate_initial_user_utterance(session), "I need a compact charger.")

    def test_self_play_context_and_interaction_kwargs_exclude_review(self) -> None:
        project = Path(__file__).resolve().parents[1]
        loop_source = (project / "self_play" / "tool_agent_loop.py").read_text(encoding="utf-8")
        self.assertNotIn("reference_review", loop_source)

        datasets = (
            (project / "dataset.py", '"name": "simrec_user"'),
            (project / "self_play" / "dataset.py", '"name": "simrec_self_play_user"'),
        )
        for dataset_path, interaction_name in datasets:
            source = dataset_path.read_text(encoding="utf-8")
            start = source.index(interaction_name)
            end = source.index('"is_validate": validate,', start)
            self.assertNotIn("reference_review", source[start:end])


if __name__ == "__main__":
    unittest.main()
