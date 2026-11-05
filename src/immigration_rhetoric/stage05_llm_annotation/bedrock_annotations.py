import json
import logging

from immigration_rhetoric.stage05_llm_annotation.batch_annotations import (
    BatchAnnotations,
)
from tenacity import retry, stop_after_attempt, wait_random_exponential

MODELS = {
    "claude_sonnet_4": {
        "region_name": "us-east-1",
        "modelId": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    },
    "claude_sonnet_3_7": {
        "region_name": "us-east-1",
        "modelId": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    },
    "claude_sonnet": {
        "region_name": "us-east-1",
        "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
    },
    "claude_opus": {
        "region_name": "us-west-2",
        "modelId": "anthropic.claude-3-opus-20240229-v1:0",
    },
    "claude_haiku": {
        "region_name": "us-east-1",
        "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
    },
    "llama3_70b_instruct": {
        "region_name": "us-east-1",
        "modelId": "meta.llama3-70b-instruct-v1:0",
    },
    "llama3_8b_instruct": {
        "region_name": "us-east-1",
        "modelId": "meta.llama3-8b-instruct-v1:0",
    },
    "llama2_70b_chat": {
        "region_name": "us-east-1",
        "modelId": "meta.llama2-70b-chat-v1",
    },
    "llama2_13b_chat": {
        "region_name": "us-east-1",
        "modelId": "meta.llama2-13b-chat-v1",
    },
    "mistral_large": {
        "region_name": "us-east-1",
        "modelId": "mistral.mistral-large-2402-v1:0",
    },
    "mistral_small": {
        "region_name": "us-east-1",
        "modelId": "mistral.mistral-small-2402-v1:0",
    },
    "mistral_8x7b": {
        "region_name": "us-east-1",
        "modelId": "mistral.mixtral-8x7b-instruct-v0:1",
    },
    "mistral_7b": {
        "region_name": "us-east-1",
        "modelId": "mistral.mistral-7b-instruct-v0:2",
    },
}


class BedrockAnnotations(BatchAnnotations):
    """Annotates snippets using an AWS Bedrock model (Claude, Llama, or Mistral)."""

    def __init__(self, bedrock_runtime):
        self.bedrock_runtime = bedrock_runtime
        self.models = MODELS

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def make_request(self, request: dict, model_id: str) -> dict:
        body = json.dumps(request)
        response = self.bedrock_runtime.invoke_model(body=body, modelId=model_id)
        return json.loads(response.get("body").read())

    def generate_message(
        self, prompt: str, temperature: float, max_tokens: int, model_id: str
    ) -> str:
        if "claude" in model_id:
            request = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            response = self.make_request(request, model_id)
            return response["content"][0]["text"]

        if "llama3" in model_id:
            request = {
                "prompt": f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
                "max_gen_len": max_tokens,
                "temperature": temperature,
            }
            response = self.make_request(request, model_id)
            return response["generation"]

        if "llama2" in model_id:
            request = {
                "prompt": f"<s>[INST]\n\n{prompt} [/INST]",
                "max_gen_len": max_tokens,
                "temperature": temperature,
            }
            response = self.make_request(request, model_id)
            return response["generation"]

        if "mistral" in model_id:
            request = {
                "prompt": f"<s>[INST] {prompt} [/INST]",
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            response = self.make_request(request, model_id)
            return response["outputs"][0]["text"]

        logging.warning(f"Unrecognised model id: {model_id}")
        return "Invalid model selected."
