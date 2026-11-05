from argparse import ArgumentParser, Namespace
from typing import Optional

import pandas as pd
from immigration_rhetoric.config import DATA_DIR, THIS_YEAR
from immigration_rhetoric.stage04_prepare_annotation_data.sample_snippets import (
    SnippetSampler,
)


def run_snippet_sampling(
    input_dir: str,
    output_dir: str,
    start_year: int,
    end_year: int,
    sample_size: int,
    sample_by_year_month: bool,
    exclude_sample_file: Optional[str],
) -> None:
    """Sample an equal number of extracted snippets per year (or year-month), ready for manual annotation."""
    sampler = SnippetSampler(input_dir=input_dir, output_dir=output_dir)

    snippets = sampler.load_snippets(start_year, end_year)

    sample_col = "year"
    if sample_by_year_month:
        snippets["year-month"] = (
            snippets["year"].astype(str) + "-" + snippets["month"].astype(str)
        )
        sample_col = "year-month"

    if exclude_sample_file:
        previous_sample = pd.read_csv(exclude_sample_file)
        snippets = sampler.exclude_already_sampled(snippets, previous_sample)

    sample = sampler.sample_by_column_value(snippets, sample_col, sample_size)

    sampler.output_sample(sample, sample_col)


def parse_args() -> Namespace:
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-dir",
        dest="input_dir",
        help="Directory containing extracted snippet csvs (stage03_snippet_extraction's clean_snippets/)",
        type=str,
        default=str(DATA_DIR / "clean_snippets"),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        help="Directory to write the sampled snippets to",
        type=str,
        default=str(DATA_DIR / "snippet_samples"),
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
        "-n",
        "--number-to-sample",
        dest="sample_size",
        help="Approximate number of snippets to sample",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--sample-by-year-month",
        dest="sample_by_year_month",
        help="Sample an even number of snippets per year-month, instead of per year",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--exclude-sample-file",
        dest="exclude_sample_file",
        help="CSV of a previous sample whose snippets should be excluded from this one",
        type=str,
        default=None,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_snippet_sampling(**vars(args))
