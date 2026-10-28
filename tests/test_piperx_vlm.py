"""Open-vocabulary semantic planning and OpenAI-compatible VLM transport."""
import json
import os
import unittest
from unittest.mock import patch

import numpy as np

from utils.piperx_vlm import (
    FallbackSemanticPlanner,
    OpenAICompatiblePlanner,
    RuleSemanticPlanner,
    VLMError,
    build_semantic_planner,
    extract_json_object,
    infer_spatial_reference,
    normalize_destination,
    semantic_match_score,
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
    def test_rules_parse_open_vocabulary_and_last_destination(self):
        plan = RuleSemanticPlanner().plan(None, "拿左边的香蕉放到右边")
        self.assertEqual(plan.target_phrase, "香蕉")
        self.assertEqual(plan.target_label, "香蕉")
        self.assertEqual(plan.spatial_reference, "leftmost")
        self.assertEqual(plan.destination, "right")
        self.assertEqual(plan.dino_prompt, "香蕉 .")
        self.assertEqual(normalize_destination(None, "把右边的梨放到左边"), "left")
        self.assertEqual(infer_spatial_reference("把右边的梨放到左边"), "rightmost")

    def test_rules_accept_unregistered_objects(self):
        plan = RuleSemanticPlanner().plan(None, "抓取一个紫色独角兽")
        self.assertEqual(plan.target_phrase, "紫色独角兽")
        self.assertEqual(plan.destination, "center")

    def test_rules_parse_english_placement(self):
        plan = RuleSemanticPlanner().plan(None, "put banana on the right")
        self.assertEqual(plan.target_phrase, "banana")
        self.assertEqual(plan.destination, "right")
        self.assertEqual(plan.spatial_reference, "none")

    def test_rules_reject_instruction_without_target(self):
        with self.assertRaisesRegex(VLMError, "--text"):
            RuleSemanticPlanner().plan(None, "放到右边")

    def test_vlm_visibility_and_rotation_are_parsed(self):
        planner = OpenAICompatiblePlanner("http://127.0.0.1:8000/v1", "deepseek-flash",
                                          api_key="test-key", timeout=5)
        content = json.dumps({
            "target_phrase": "bottle",
            "target_visible": False,
            "selected_matches_target": False,
            "search_yaw_deg": 35,
            "search_reason": "drink bottle not in frame",
            "spatial_reference": "none",
            "action": "pick",
            "destination": "center",
        })
        captured = {}

        def fake_urlopen(request, timeout):
            captured["body"] = json.loads(request.data)
            return _Response({"choices": [{"message": {"content": content}}]})

        with patch("utils.piperx_vlm.urllib.request.urlopen", side_effect=fake_urlopen):
            plan = planner.plan(
                np.zeros((32, 32, 3), dtype=np.uint8),
                "抓取饮料瓶子",
                detection={"box": [10, 20, 30, 40], "label": "screwdriver",
                           "confidence": 0.4})
        self.assertFalse(plan.target_visible)
        self.assertFalse(plan.selected_matches_target)
        self.assertAlmostEqual(plan.search_yaw_deg, 35.0)
        self.assertEqual(plan.search_reason, "drink bottle not in frame")
        text = captured["body"]["messages"][1]["content"][1]["text"]
        self.assertIn("Detector box", text)
        self.assertIn("screwdriver", text)

    def test_rule_plan_defaults_to_visible_without_rotation(self):
        plan = RuleSemanticPlanner().plan(None, "抓取瓶子")
        self.assertTrue(plan.target_visible)
        self.assertIsNone(plan.selected_matches_target)
        self.assertEqual(plan.search_yaw_deg, 0.0)

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
        self.assertEqual(plan.target_phrase, "banana")
        self.assertEqual(plan.target_label, "banana")
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
        self.assertEqual(plan.target_phrase, "banana")

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
        self.assertEqual(plan.target_phrase, "草莓")
        self.assertEqual(plan.destination, "left")
        self.assertEqual(plan.backend, "rules")

    def test_openai_backend_requires_configuration(self):
        with patch.dict(os.environ, {"PIPERX_VLM_BASE_URL": "", "PIPERX_VLM_MODEL": ""}):
            with self.assertRaisesRegex(ValueError, "PIPERX_VLM_BASE_URL"):
                build_semantic_planner(log=lambda *_: None)
            with self.assertRaisesRegex(ValueError, "PIPERX_VLM_BASE_URL"):
                build_semantic_planner("auto", log=lambda *_: None)
            self.assertEqual(build_semantic_planner("rules").name, "rules")

    def test_semantic_match_score_is_open_vocabulary(self):
        self.assertGreater(semantic_match_score("banana", "yellow banana ."), 0)
        self.assertEqual(semantic_match_score("", "banana"), 0)
        self.assertEqual(semantic_match_score("pear", "banana"), 0)


if __name__ == "__main__":
    unittest.main()
