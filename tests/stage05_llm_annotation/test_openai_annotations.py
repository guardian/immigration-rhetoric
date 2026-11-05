from unittest.mock import MagicMock, patch

from immigration_rhetoric.stage05_llm_annotation.openai_annotations import (
    MODELS,
    OpenAIAnnotations,
)


class TestOpenAIAnnotationsInit:
    def test_reads_api_key_from_environment(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with patch(
            "immigration_rhetoric.stage05_llm_annotation.openai_annotations.openai"
        ) as mock_openai:
            OpenAIAnnotations()
            assert mock_openai.api_key == "test-key"


class TestGenerateMessage:
    def test_gpt_model_returns_response_content(self):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "a gpt response"

        with patch(
            "immigration_rhetoric.stage05_llm_annotation.openai_annotations.openai"
        ) as mock_openai:
            mock_openai.chat.completions.create.return_value = mock_response
            annotations = OpenAIAnnotations()

            result = annotations.generate_message(
                "prompt", 0.0, 100, MODELS["gpt4_1"]["modelId"]
            )

        assert result == "a gpt response"
        mock_openai.chat.completions.create.assert_called_once_with(
            model=MODELS["gpt4_1"]["modelId"],
            messages=[{"role": "user", "content": "prompt"}],
            temperature=0.0,
            max_tokens=100,
        )

    def test_unrecognised_model(self):
        with patch(
            "immigration_rhetoric.stage05_llm_annotation.openai_annotations.openai"
        ):
            annotations = OpenAIAnnotations()
            result = annotations.generate_message(
                "prompt", 0.0, 100, "some-unknown-model"
            )
        assert result == "Invalid model selected."
