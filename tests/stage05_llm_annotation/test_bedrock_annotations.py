import json
from unittest.mock import MagicMock

from immigration_rhetoric.stage05_llm_annotation.bedrock_annotations import (
    MODELS,
    BedrockAnnotations,
)


def _mock_bedrock_runtime(body: dict):
    runtime = MagicMock()
    runtime.invoke_model.return_value = {
        "body": MagicMock(read=MagicMock(return_value=json.dumps(body).encode()))
    }
    return runtime


class TestMakeRequest:
    def test_invokes_model_and_parses_response_body(self):
        runtime = _mock_bedrock_runtime({"content": [{"text": "hello"}]})
        annotations = BedrockAnnotations(bedrock_runtime=runtime)

        result = annotations.make_request(
            {"some": "request"}, "anthropic.claude-3-haiku-20240307-v1:0"
        )

        assert result == {"content": [{"text": "hello"}]}
        runtime.invoke_model.assert_called_once()


class TestGenerateMessage:
    def test_claude_model(self):
        runtime = _mock_bedrock_runtime({"content": [{"text": "a claude response"}]})
        annotations = BedrockAnnotations(bedrock_runtime=runtime)

        result = annotations.generate_message(
            "prompt", 0.0, 100, MODELS["claude_haiku"]["modelId"]
        )
        assert result == "a claude response"

    def test_llama3_model(self):
        runtime = _mock_bedrock_runtime({"generation": "a llama3 response"})
        annotations = BedrockAnnotations(bedrock_runtime=runtime)

        result = annotations.generate_message(
            "prompt", 0.0, 100, MODELS["llama3_8b_instruct"]["modelId"]
        )
        assert result == "a llama3 response"

    def test_llama2_model(self):
        runtime = _mock_bedrock_runtime({"generation": "a llama2 response"})
        annotations = BedrockAnnotations(bedrock_runtime=runtime)

        result = annotations.generate_message(
            "prompt", 0.0, 100, MODELS["llama2_13b_chat"]["modelId"]
        )
        assert result == "a llama2 response"

    def test_mistral_model(self):
        runtime = _mock_bedrock_runtime({"outputs": [{"text": "a mistral response"}]})
        annotations = BedrockAnnotations(bedrock_runtime=runtime)

        result = annotations.generate_message(
            "prompt", 0.0, 100, MODELS["mistral_7b"]["modelId"]
        )
        assert result == "a mistral response"

    def test_unrecognised_model(self):
        annotations = BedrockAnnotations(bedrock_runtime=MagicMock())
        result = annotations.generate_message("prompt", 0.0, 100, "some.unknown-model")
        assert result == "Invalid model selected."
