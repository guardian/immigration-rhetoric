import asyncio
import json
import logging

import numpy as np
import pandas as pd
from immigration_rhetoric.common.logging_utils import TqdmLoggingHandler
from immigration_rhetoric.stage05_llm_annotation.prompts import (
    clean_step2_response,
    convert_alphabetic_labels,
    highest_score_labels,
    prompt_step_1,
    prompt_step_2,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[TqdmLoggingHandler()],
)


class BatchAnnotations:
    """Annotates snippets with an LLM's sentiment towards immigration, via a two-step prompt.

    Subclasses (`BedrockAnnotations`, `OpenAIAnnotations`) provide `self.models` and implement
    `make_request`/`generate_message` for their specific API.
    """

    models: dict = {}

    @staticmethod
    def split_into_batches(input_df: pd.DataFrame, chunk_size: int = -1) -> list:
        """Split `input_df` into batches of `chunk_size` rows, or a single batch if `chunk_size` is -1."""
        if chunk_size <= 0:
            return [input_df]
        return [
            group
            for _, group in input_df.groupby(np.arange(len(input_df)) // chunk_size)
        ]

    def generate_message(
        self, prompt: str, temperature: float, max_tokens: int, model_id: str
    ) -> str:
        """Send `prompt` to the model and return its response text. Implemented by subclasses."""
        raise NotImplementedError

    def annotate_snippets_batch(
        self,
        snippet_batch_df: pd.DataFrame,
        model_name: str,
        temperature: float,
        max_tokens: int,
    ) -> pd.DataFrame:
        """Run the two-step annotation prompt over every row of a batch, writing each result to stdout as it goes."""
        model_id = self.models[model_name]["modelId"]
        logging.info(f"Starting batch {snippet_batch_df.index[0]} (model: {model_id})")

        step1_responses = []
        step2_responses = []
        cleaned_labels = []

        for i, row in snippet_batch_df.iterrows():
            row_dict = row.to_dict()
            try:
                response_step1 = self.generate_message(
                    prompt_step_1(row_dict["text"]), temperature, max_tokens, model_id
                )
                response_step2 = self.generate_message(
                    prompt_step_2(response_step1), temperature, max_tokens, model_id
                )

                step1_responses.append(response_step1)
                step2_responses.append(response_step2)
                try:
                    clean = clean_step2_response(response_step2)
                    highest = highest_score_labels(clean)
                    labels = convert_alphabetic_labels(highest)
                except Exception:
                    labels = response_step2
                cleaned_labels.append(labels)

                row_dict[f"{model_name}_step1_responses"] = response_step1
                row_dict[f"{model_name}_step2_responses"] = response_step2
                row_dict[f"{model_name}_labels"] = labels

                # Streamed to stdout (rather than logged/buffered) so a run can be redirected to a file and
                # tailed as it progresses, without losing hours of work if the process is killed partway through.
                print(json.dumps(row_dict, default=str))

            except Exception:
                logging.error(f"Error processing snippet {i}", exc_info=True)
                step1_responses.append("Error")
                step2_responses.append("Error")
                cleaned_labels.append("Error")

        snippet_batch_df[f"{model_name}_step1_responses"] = step1_responses
        snippet_batch_df[f"{model_name}_step2_responses"] = step2_responses
        snippet_batch_df[f"{model_name}_labels"] = cleaned_labels

        desired_columns = [
            "text",
            "meta",
            "snippet_metadata",
            f"{model_name}_step1_responses",
            f"{model_name}_step2_responses",
            f"{model_name}_labels",
        ]
        available_columns = [
            c for c in desired_columns if c in snippet_batch_df.columns
        ]
        return snippet_batch_df[available_columns]

    async def annotate_snippets(
        self,
        executor,
        model_name: str,
        temperature: float,
        max_tokens: int,
        input_file: str,
        chunk_size: int,
    ) -> list:
        """Load `input_file` (CSV or JSONL) and annotate it, running one batch per worker in `executor`."""
        loop = asyncio.get_event_loop()
        if input_file.endswith(".jsonl"):
            snippets_df = pd.read_json(input_file, lines=True)
        elif input_file.endswith(".csv"):
            snippets_df = pd.read_csv(input_file)
        else:
            raise ValueError("Input file must be a CSV or JSONL file.")

        snippet_batches = self.split_into_batches(snippets_df, chunk_size=chunk_size)

        batch_tasks = [
            loop.run_in_executor(
                executor,
                self.annotate_snippets_batch,
                snippet_batch_df,
                model_name,
                temperature,
                max_tokens,
            )
            for snippet_batch_df in snippet_batches
        ]

        completed, _ = await asyncio.wait(
            batch_tasks, return_when=asyncio.ALL_COMPLETED
        )
        return [t.result() for t in completed]
