from argparse import ArgumentParser, Namespace
from typing import List

from immigration_rhetoric.stage07_model_training.build_training_data import (
    DEFAULT_SEEDS,
    TrainingDataBuilder,
)


def run_build_training_data(
    input_file: str,
    output_dir: str,
    seeds: List[int] = DEFAULT_SEEDS,
    holdout_frac: float = 0.1,
    eval_frac: float = 0.2,
) -> None:
    """Build holdout + per-seed train/eval spaCy files for the relevance and sentiment models."""
    builder = TrainingDataBuilder(
        output_dir=output_dir, holdout_frac=holdout_frac, eval_frac=eval_frac
    )
    builder.build(input_file=input_file, seeds=seeds)


def parse_args() -> Namespace:
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-file",
        dest="input_file",
        help="Consolidated annotations jsonl (stage06_agreement's '_consolidated' output) with 'text', "
        "'accept' and 'snippet_metadata' fields",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        help="Directory to write the .spacy and .jsonl files to",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-s",
        "--seeds",
        dest="seeds",
        help="Random seeds, one train/eval split is created per seed",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
    )
    parser.add_argument(
        "--holdout-frac",
        dest="holdout_frac",
        help="Fraction of snippets held out from all training/eval splits",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--eval-frac",
        dest="eval_frac",
        help="Fraction of the remaining snippets used for evaluation in each split",
        type=float,
        default=0.2,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_build_training_data(**vars(args))
