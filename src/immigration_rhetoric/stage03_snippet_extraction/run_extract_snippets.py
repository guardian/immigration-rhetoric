import logging
import os
import sys
import time
import warnings
from argparse import ArgumentParser, Namespace
from multiprocessing import Pool, set_start_method
from pathlib import Path
from typing import List, Optional

import pandas as pd
import spacy
from immigration_rhetoric.common.logging_utils import TqdmLoggingHandler
from immigration_rhetoric.common.utils import extract_numbers_as_int
from immigration_rhetoric.config import DATA_DIR, ROOT_DIR, THIS_YEAR
from immigration_rhetoric.stage03_snippet_extraction.clean_and_extract_snippets import (
    SnippetExtractor,
)
from tqdm import tqdm

# en_core_web_lg doesn't use transformers/torch, but spacy[transformers] being installed means torch is
# still imported as a side effect, which raises this on every process (main + each worker under spawn).
warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\.jit\._script")

# When output is redirected to a file/log (not an interactive terminal), tqdm can't overwrite its own
# progress line, so every refresh becomes a separate line. Disable the bar in that case.
TQDM_DISABLE = not sys.stderr.isatty()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[TqdmLoggingHandler()],
)


_nlp = None


def _init_worker(nlp) -> None:
    """Store the already-loaded model once per worker, so tasks don't each carry their own copy."""
    global _nlp
    _nlp = nlp
    logging.info(f"Worker process {os.getpid()} ready")


def _extract_snippets_for_year(args: tuple) -> tuple:
    """Runs snippet extraction for a single year's speeches. Used as a multiprocessing worker function."""
    df, snippet_extractor, export_filename = args
    return snippet_extractor.snippet_extract(df, export_filename, _nlp)


def load_speeches_by_year(
    speech_dir: Path, start_year: int, end_year: int
) -> List[pd.DataFrame]:
    """Load each year's joined speeches csv within the given year range into a dataframe."""
    dataframes = []
    for file_path in sorted(speech_dir.glob("speeches*.csv")):
        year = extract_numbers_as_int(file_path.name)
        if start_year <= year <= end_year:
            dataframes.append((year, pd.read_csv(file_path, low_memory=False)))
    return dataframes


def run_extract_snippets(
    output_dir: str,
    trigger_file: str,
    start_year: int,
    end_year: int,
    min_word_count: int,
    speeches_per_year: Optional[int],
    num_processes: int,
) -> None:
    """Clean speech text and extract immigration-related snippets around trigger terms, in parallel by year."""
    trigger_terms = pd.read_csv(trigger_file)["term"].to_list()

    snippet_extractor = SnippetExtractor(
        output_dir=output_dir,
        trigger_terms=trigger_terms,
        min_word_count=min_word_count,
        speeches_per_year=speeches_per_year,
    )

    logging.info("Loading spacy model...")
    load_start_time = time.perf_counter()
    nlp = spacy.load(
        "en_core_web_lg",
        disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"],
    )
    nlp.enable_pipe("senter")
    logging.info(f"Loaded spacy model in {time.perf_counter() - load_start_time:.1f}s")

    logging.info(f"Loading speeches from {snippet_extractor.speech_dir}...")
    load_start_time = time.perf_counter()
    dataframes = load_speeches_by_year(
        snippet_extractor.speech_dir, start_year, end_year
    )
    logging.info(
        f"Loaded {len(dataframes)} year(s) of speeches ({start_year}-{end_year}) "
        f"in {time.perf_counter() - load_start_time:.1f}s"
    )

    tasks = [(df, snippet_extractor, str(year)) for year, df in dataframes]
    num_workers = max(1, num_processes - 1)
    logging.info(
        f"Starting extraction across {len(tasks)} year(s) using {num_workers} worker process(es)..."
    )
    with Pool(processes=num_workers, initializer=_init_worker, initargs=(nlp,)) as pool:
        results = list(
            tqdm(
                pool.imap(_extract_snippets_for_year, tasks),
                total=len(tasks),
                desc="Extracting snippets",
                disable=TQDM_DISABLE,
            )
        )

    total_speeches = sum(speeches for speeches, _ in results)
    total_snippets = sum(snippets for _, snippets in results)
    logging.info(
        f"Extracted {total_snippets} snippets from {total_speeches} speeches with trigger terms, "
        f"across {len(tasks)} year(s) ({start_year}-{end_year})"
    )


def parse_args() -> Namespace:
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        help="Directory containing the speeches/ subdirectory to read, and to write clean_snippets/ into",
        type=str,
        default=str(DATA_DIR),
    )
    parser.add_argument(
        "-tf",
        "--trigger-file",
        dest="trigger_file",
        help="CSV file with a 'term' column listing the immigration trigger terms",
        type=str,
        default=str(ROOT_DIR / "data" / "trigger_terms.csv"),
    )
    parser.add_argument(
        "-s",
        "--start-year",
        dest="start_year",
        help="Least recent speeches required (format: YYYY)",
        type=int,
        default=1919,
    )
    parser.add_argument(
        "-e",
        "--end-year",
        dest="end_year",
        help="Most recent speeches required (format: YYYY)",
        type=int,
        default=THIS_YEAR,
    )
    parser.add_argument(
        "--min-word-count",
        dest="min_word_count",
        help="Minimum number of words a speech must have to be processed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--speeches-per-year",
        dest="speeches_per_year",
        help="Number of speeches to randomly sample from each year",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-n",
        "--num-processes",
        dest="num_processes",
        help="Number of processes to run concurrently",
        type=int,
        default=os.cpu_count(),
    )
    return parser.parse_args()


if __name__ == "__main__":
    set_start_method("spawn")
    args = parse_args()
    run_extract_snippets(**vars(args))
