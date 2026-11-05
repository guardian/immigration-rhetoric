import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

from immigration_rhetoric.common.logging_utils import TqdmLoggingHandler
from immigration_rhetoric.config import DATA_DIR
from immigration_rhetoric.stage08_inference.combine_inference_outputs import (
    InferenceOutputCombiner,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[TqdmLoggingHandler()],
)


def run_combine_inference_outputs(
    input_dir: str,
    output_dir: str,
    model_version: str,
    min_speech_date: str,
    max_speech_date: str,
) -> None:
    """Write the combined labelled snippets and per-speech summary as two jsonl files."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    speeches, snippets = InferenceOutputCombiner().combine(input_dir)

    suffix = f"{model_version}_{min_speech_date}_{max_speech_date}.jsonl"
    for name, df in [("speeches", speeches), ("labelled_snippets", snippets)]:
        output_path = output_dir / f"{name}_{suffix}"
        df.to_json(output_path, orient="records", lines=True)
        logging.info(f"Wrote {len(df)} rows to {output_path}")


def parse_args() -> Namespace:
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-dir",
        dest="input_dir",
        help="Directory of labelled_snippets_{year}.jsonl files",
        type=str,
        default=str(DATA_DIR / "inference"),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        help="Directory to write the combined files to",
        type=str,
        default=str(DATA_DIR / "analysis_datasets"),
    )
    parser.add_argument(
        "-v",
        "--model-version",
        dest="model_version",
        help="Version of the models used, for the output file names (e.g. v1.0.0)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--min-speech-date",
        dest="min_speech_date",
        help="Earliest speech date covered, for the output file names (format: YYYYMMDD)",
        type=str,
        default="19190101",
    )
    parser.add_argument(
        "--max-speech-date",
        dest="max_speech_date",
        help="Latest speech date covered, for the output file names (format: YYYYMMDD)",
        type=str,
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_combine_inference_outputs(**vars(args))
