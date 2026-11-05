import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
import spacy
from immigration_rhetoric.common.logging_utils import TqdmLoggingHandler
from immigration_rhetoric.common.utils import extract_numbers_as_int
from immigration_rhetoric.config import DATA_DIR, THIS_YEAR
from immigration_rhetoric.stage08_inference.classify_snippets import (
    RELEVANT_LABEL,
    SnippetClassifier,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[TqdmLoggingHandler()],
)


def run_inference(
    input_dir: str,
    output_dir: str,
    relevance_model: str,
    sentiment_model: str,
    start_year: int,
    end_year: int,
    processes: int,
    snippet_col: str,
    gpu_id: int,
) -> None:
    """Label each year's snippets with the relevance and sentiment models, writing one jsonl per year."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if gpu_id >= 0:
        spacy.require_gpu(gpu_id)

    logging.info("Loading models...")
    classifier = SnippetClassifier(
        relevance_nlp=spacy.load(relevance_model),
        sentiment_nlp=spacy.load(sentiment_model),
        n_process=processes,
    )

    for file_path in sorted(input_dir.glob("snippets_hansard_data_clean_*.csv")):
        year = extract_numbers_as_int(file_path.name)
        if not start_year <= year <= end_year:
            continue

        df = pd.read_csv(file_path).dropna(subset=["text"]).reset_index(drop=True)
        df = classifier.classify(df, snippet_col=snippet_col)

        output_path = output_dir / f"labelled_snippets_{year}.jsonl"
        df.to_json(output_path, orient="records", date_format="iso", lines=True)
        n_relevant = (df["relevance_label"] == RELEVANT_LABEL).sum()
        logging.info(
            f"{year}: labelled {len(df)} snippets ({n_relevant} about immigration) -> {output_path}"
        )


def parse_args() -> Namespace:
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-dir",
        dest="input_dir",
        help="Directory of snippets_hansard_data_clean_{year}.csv files",
        type=str,
        default=str(DATA_DIR / "clean_snippets"),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        help="Directory to write labelled_snippets_{year}.jsonl files to",
        type=str,
        default=str(DATA_DIR / "inference"),
    )
    parser.add_argument(
        "-r",
        "--relevance-model",
        dest="relevance_model",
        help="Path to (or package name of) the relevance model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-s",
        "--sentiment-model",
        dest="sentiment_model",
        help="Path to (or package name of) the sentiment model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--start-year",
        dest="start_year",
        help="Least recent snippets required (format: YYYY)",
        type=int,
        default=1919,
    )
    parser.add_argument(
        "--end-year",
        dest="end_year",
        help="Most recent snippets required (format: YYYY)",
        type=int,
        default=THIS_YEAR,
    )
    parser.add_argument(
        "-p",
        "--processes",
        dest="processes",
        help="Number of processes spaCy uses to run each model",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-c",
        "--snippet-col",
        dest="snippet_col",
        help="Name of the column holding the snippet text",
        type=str,
        default="snippet",
    )
    parser.add_argument(
        "-g",
        "--gpu-id",
        dest="gpu_id",
        help="GPU to use, -1 for CPU",
        type=int,
        default=-1,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(**vars(args))
