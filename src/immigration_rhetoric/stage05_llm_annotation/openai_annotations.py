import os

from immigration_rhetoric.stage05_llm_annotation.batch_annotations import (
    BatchAnnotations,
)
from tenacity import retry, stop_after_attempt, wait_random_exponential

import openai

MODELS = {
    "gpt4_1": {
        "modelId": "gpt-4.1-2025-04-14",
    },
    "gpt4": {
        "modelId": "gpt-4-1106-preview",
    },
    "gpt4o": {
        "modelId": "gpt-4o-2024-05-13",
    },
}


class OpenAIAnnotations(BatchAnnotations):
    """Annotates snippets using an OpenAI GPT model."""

    def __init__(self):
        self.models = MODELS
        openai.api_key = os.getenv("OPENAI_API_KEY")

    @staticmethod
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def make_request(request: dict):
        return openai.chat.completions.create(**request)

    def generate_message(
        self, prompt: str, temperature: float, max_tokens: int, model_id: str
    ) -> str:
        if "gpt" not in model_id:
            return "Invalid model selected."

        request = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = self.make_request(request)
        return response.choices[0].message.content
