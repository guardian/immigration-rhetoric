import asyncio
import concurrent.futures
import logging
from argparse import ArgumentParser, Namespace
from typing import Optional

import boto3
import pandas as pd
from immigration_rhetoric.stage05_llm_annotation.bedrock_annotations import (
    MODELS,
    BedrockAnnotations,
)


def run_bedrock_annotations(
    model_name: str,
    temperature: float,
    max_tokens: int,
    input_file: str,
    chunk_size: int,
    output_file: Optional[str],
    max_processes: int,
) -> None:
    """Annotate every snippet in `input_file` with an AWS Bedrock model's sentiment towards immigration."""
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_processes)

    region_name = MODELS[model_name]["region_name"]
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime", region_name=region_name
    )
    bedrock_annotations = BedrockAnnotations(bedrock_runtime=bedrock_runtime)

    results = asyncio.run(
        bedrock_annotations.annotate_snippets(
            executor=executor,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            input_file=input_file,
            chunk_size=chunk_size,
        )
    )

    if output_file:
        # Results are already streamed to stdout batch-by-batch as they complete; writing this combined
        # file is an optional convenience, not how the pipeline normally consumes the output.
        concat_results = pd.concat(results, axis=0)
        concat_results.to_json(output_file, orient="records", lines=True)
        logging.info(f"Results saved to {output_file}")


def parse_args() -> Namespace:
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        "--model-name",
        dest="model_name",
        type=str,
        required=True,
        help=f"Model name for request. One of: {', '.join(MODELS)}",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        dest="temperature",
        type=float,
        default=0.0,
        help="Temperature for request.",
    )
    parser.add_argument(
        "-n",
        "--max-tokens",
        dest="max_tokens",
        type=int,
        default=300,
        help="Max tokens for request.",
    )
    parser.add_argument(
        "-i",
        "--input-file",
        dest="input_file",
        type=str,
        required=True,
        help="Input CSV or JSONL file with snippets to process.",
    )
    parser.add_argument(
        "-c",
        "--chunk-size",
        dest="chunk_size",
        type=int,
        required=True,
        help="Number of snippets to include in each batch. Set to -1 if only one batch is desired.",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        dest="output_file",
        type=str,
        default=None,
        help="Optional combined output JSONL file (results are also streamed to stdout as each batch completes).",
    )
    parser.add_argument(
        "-p",
        "--max-processes",
        dest="max_processes",
        type=int,
        default=10,
        help="Number of concurrent batches to run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_bedrock_annotations(**vars(args))
