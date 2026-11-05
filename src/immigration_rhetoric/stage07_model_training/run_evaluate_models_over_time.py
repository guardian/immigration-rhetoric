from argparse import ArgumentParser, Namespace
from pathlib import Path

from immigration_rhetoric.stage07_model_training.evaluate_models_over_time import (
    PerformanceOverTimeEvaluator,
)


def run_evaluate_models_over_time(
    predictions: str, plot: str, min_year: int | None = 1923
) -> None:
    """Plot the combined models' eval performance by decade from their actual-vs-predicted labels."""
    evaluator = PerformanceOverTimeEvaluator()
    df = evaluator.load_predictions(Path(predictions), min_year=min_year)
    evaluator.plot_overall_by_decade(df, Path(plot))


def parse_args() -> Namespace:
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--predictions",
        dest="predictions",
        help="Path to the combined actual-vs-predicted labels jsonl written by run_evaluate_models",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-p",
        "--plot",
        dest="plot",
        help="Path to write the performance over time plot (png) to",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--min-year",
        dest="min_year",
        help="Leave out snippets before this year, and start the x-axis at the start of its decade "
        "(default 1923, where the paper's figures start; 0 keeps every snippet)",
        type=int,
        default=1923,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluate_models_over_time(**vars(args))
