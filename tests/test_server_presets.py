import json
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from src.core.configuration import load_config
from src.models.qwen_adapter import QwenOpenAICompatibleAdapter


class ServerPresetTests(unittest.TestCase):
    def make_response(self):
        text = json.dumps({"final_answer": "A", "brief_explanation": "Visible", "confidence": 0.9})
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=text), finish_reason="stop",
            )],
            usage=None,
            model_dump=lambda: {"text": text},
        )

    def test_deepseek_server_preserves_images_and_generation_options(self):
        cfg = load_config(model="deepseek-vl2-tiny-server")["model"]
        client = MagicMock()
        client.chat.completions.create.return_value = self.make_response()
        with patch("src.models.qwen_adapter.OpenAI", return_value=client), \
             patch("src.models.qwen_adapter._to_data_url", side_effect=lambda path: path):
            adapter = QwenOpenAICompatibleAdapter(cfg)
            response = adapter.infer("system", "question", "rgb", "depth", "mask", "single_choice")
            adapter.repair_format("bad answer", "single_choice", ["A", "B"])
        self.assertEqual(response.parsed_json["final_answer"], "A")
        infer, repair = [call.kwargs for call in client.chat.completions.create.call_args_list]
        content = infer["messages"][1]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "question"})
        self.assertEqual([item["image_url"]["url"] for item in content[1:]], ["rgb", "mask", "depth"])
        for request in (infer, repair):
            self.assertEqual(request["model"], "deepseek-ai/deepseek-vl2-tiny")
            self.assertEqual(request["max_tokens"], 512)
            self.assertEqual(request["temperature"], 0.0)
            self.assertEqual(request["top_p"], 0.9)
            self.assertEqual(request["extra_body"], {"repetition_penalty": 1.1})
            self.assertNotIn("response_format", request)
        self.assertTrue(all(isinstance(message["content"], str) for message in repair["messages"]))

    def test_existing_qwen_requests_do_not_gain_new_parameters(self):
        client = MagicMock()
        client.chat.completions.create.return_value = self.make_response()
        with patch("src.models.qwen_adapter.OpenAI", return_value=client), \
             patch("src.models.qwen_adapter._to_data_url", side_effect=lambda path: path):
            adapter = QwenOpenAICompatibleAdapter(load_config(model="qwen3")["model"])
            adapter.infer("system", "question", "rgb", question_type="single_choice")
            adapter.repair_format("bad answer", "single_choice", ["A", "B"])
        requests = [call.kwargs for call in client.chat.completions.create.call_args_list]
        self.assertEqual([request["max_tokens"] for request in requests], [2048, 256])
        for request in requests:
            self.assertNotIn("top_p", request)
            self.assertNotIn("extra_body", request)
            self.assertEqual(request["response_format"]["type"], "json_schema")

    def test_new_presets_leave_existing_experiments_separate(self):
        direct = load_config(model="deepseek-vl2-tiny")
        server = load_config(model="deepseek-vl2-tiny-server")
        self.assertEqual(direct["model"]["provider"], "deepseek_vl2")
        self.assertEqual(server["model"]["provider"], "qwen_openai_compatible")
        self.assertNotEqual(direct["paths"]["output_dir"], server["paths"]["output_dir"])
        hf = load_config(model="llava-ov-7b-hf")
        self.assertEqual(hf["model"]["model_name"], "llava-hf/llava-onevision-qwen2-7b-ov-hf")
        self.assertNotEqual(hf["paths"]["output_dir"], load_config(model="llava-ov-7b")["paths"]["output_dir"])
        gemini = load_config(model="gemini-3.8-flash")
        self.assertEqual(gemini["model"]["provider"], "google")
        self.assertEqual(gemini["model"]["model_name"], "gemini-3.8-flash")
        self.assertEqual(gemini["model"]["api_key_env"], "GOOGLE_API_KEY")


if __name__ == "__main__":
    unittest.main()
