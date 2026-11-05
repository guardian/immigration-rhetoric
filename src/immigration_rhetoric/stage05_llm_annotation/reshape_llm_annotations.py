import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional

import pandas as pd
from immigration_rhetoric.config import ANNOTATIONS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

KEPT_COLUMNS = ["text", "meta", "snippet_metadata"]


def reshape_llm_annotations(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Convert a model's `{model_name}_labels` into the Prodigy-style rows stage06_agreement expects.

    Rows whose labels are not a list (the model's reply couldn't be parsed, or the request errored) are dropped.
    """
    labels = df[f"{model_name}_labels"]
    is_valid = labels.apply(lambda x: isinstance(x, list) and len(x) > 0)
    if not is_valid.all():
        logging.warning(
            f"Dropping {(~is_valid).sum()} of {len(df)} row(s) without parseable labels"
        )

    df = df[is_valid]
    reshaped = df[[c for c in KEPT_COLUMNS if c in df.columns]].copy()
    reshaped["accept"] = df[f"{model_name}_labels"]
    reshaped["answer"] = "accept"
    reshaped["_annotator_id"] = model_name
    return reshaped.reset_index(drop=True)


def run_reshape_llm_annotations(
    input_file: str, model_name: str, output_file: Optional[str]
) -> None:
    """Run the annotation reshaping on a given LLM model's output file."""
    output_path = Path(
        output_file or ANNOTATIONS_DIR / "raw_annotations" / f"{model_name}.jsonl"
    )
    reshaped = reshape_llm_annotations(pd.read_json(input_file, lines=True), model_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reshaped.to_json(output_path, orient="records", lines=True)
    logging.info(f"Saved {len(reshaped)} annotation(s) to {output_path}")


def parse_args() -> Namespace:
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-file",
        dest="input_file",
        type=str,
        required=True,
        help="JSONL output of run_bedrock_annotations/run_openai_annotations",
    )
    parser.add_argument(
        "-m",
        "--model-name",
        dest="model_name",
        type=str,
        required=True,
        help="Model name the annotations were produced with; also used as the `_annotator_id`",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        dest="output_file",
        type=str,
        default=None,
        help="Defaults to data/annotations/raw_annotations/{model_name}.jsonl",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_reshape_llm_annotations(**vars(args))
