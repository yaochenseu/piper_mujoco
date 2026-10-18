"""Semantic instruction planning and OpenAI-compatible VLM transport."""
import json
import unittest
from unittest.mock import patch

import numpy as np

from utils.piperx_vlm import (
    FallbackSemanticPlanner,
    OpenAICompatiblePlanner,
    RuleSemanticPlanner,
    VLMError,
    extract_json_object,
    infer_spatial_reference,
    normalize_destination,
)


class _Response:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


class VLMPlannerTests(unittest.TestCase):
    def test_rules_resolve_target_and_last_destination(self):
        plan = RuleSemanticPlanner().plan(None, "拿左边的香蕉放到右边")
        self.assertEqual(plan.target_key, "banana")
        self.assertEqual(plan.target_phrase, "banana")
        self.assertEqual(plan.spatial_reference, "leftmost")
        self.assertEqual(plan.destination, "right")
        self.assertEqual(plan.dino_prompt, "banana .")
        self.assertEqual(normalize_destination(None, "把右边的梨放到左边"), "left")
        self.assertEqual(infer_spatial_reference("把右边的梨放到左边"), "rightmost")

    def test_openai_client_sends_image_and_parses_json(self):
        planner = OpenAICompatiblePlanner("http://127.0.0.1:8000/v1", "deepseek-flash",
                                          api_key="test-key", timeout=5)
        captured = {}
        content = json.dumps({
            "target_phrase": "banana", "spatial_reference": "none",
            "action": "pick_place", "destination": "right",
        })

        def fake_urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return _Response({"choices": [{"message": {"content": content}}]})

        with patch("utils.piperx_vlm.urllib.request.urlopen", side_effect=fake_urlopen):
            plan = planner.plan(np.zeros((32, 32, 3), dtype=np.uint8),
                                "把香蕉放到右边")
        self.assertEqual(plan.target_key, "banana")
        self.assertEqual(plan.destination, "right")
        self.assertEqual(plan.backend, "openai-compatible")
        request = captured["request"]
        body = json.loads(request.data)
        self.assertEqual(request.full_url, "http://127.0.0.1:8000/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(body["model"], "deepseek-flash")
        self.assertEqual(body["max_tokens"], 2048)
        image = body["messages"][1]["content"][0]
        self.assertEqual(image["type"], "image_url")
        self.assertTrue(image["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_deepseek_base_uses_chat_completions_route(self):
        planner = OpenAICompatiblePlanner("https://api.deepseek.com", "deepseek-flash",
                                          api_key="test-key", timeout=5)
        self.assertEqual(planner.endpoint, "https://api.deepseek.com/chat/completions")
        content = json.dumps({"target_phrase": "banana", "destination": "right"})
        response = _Response({"choices": [{"message": {"content": content}}]})
        with patch("utils.piperx_vlm.urllib.request.urlopen", return_value=response):
            plan = planner.plan(np.zeros((8, 8, 3), dtype=np.uint8), "put banana on the right")
        self.assertEqual(plan.target_key, "banana")

    def test_json_fences_and_invalid_payload(self):
        self.assertEqual(extract_json_object('```json\n{"target_phrase": "pear"}\n```'),
                         {"target_phrase": "pear"})
        with self.assertRaises(VLMError):
            extract_json_object("not json")

    def test_empty_reasoning_limited_content_is_a_clear_error(self):
        planner = OpenAICompatiblePlanner("https://api.deepseek.com", "deepseek-flash",
                                          api_key="test-key", timeout=5)
        response = _Response({"choices": [{
            "message": {"content": "", "reasoning_content": "still thinking"},
            "finish_reason": "length",
        }]})
        with patch("utils.piperx_vlm.urllib.request.urlopen", return_value=response):
            with self.assertRaisesRegex(VLMError, "finish_reason='length'"):
                planner.plan(np.zeros((8, 8, 3), dtype=np.uint8), "put banana on the right")

    def test_auto_fallback_is_used_when_primary_fails(self):
        class BrokenPlanner:
            name = "broken"

            def plan(self, *_args, **_kwargs):
                raise VLMError("offline")

        planner = FallbackSemanticPlanner(BrokenPlanner(), RuleSemanticPlanner(),
                                          log=lambda *_: None)
        plan = planner.plan(None, "把草莓放到左边")
        self.assertEqual(plan.target_key, "strawberry")
        self.assertEqual(plan.destination, "left")
        self.assertEqual(plan.backend, "rules")


if __name__ == "__main__":
    unittest.main()
