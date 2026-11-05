import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

from immigration_rhetoric.stage07_model_training.bayesian_analysis import (
    PerformanceUncertaintyEstimator,
)


def run_bayesian_analysis(
    training_dir: str, seeds: list[int], output: str, hdi_prob: float = 0.95
) -> None:
    """Estimate each metric's mean F1 with a credible interval from every seed's eval metrics, and save a summary."""
    estimator = PerformanceUncertaintyEstimator()
    f1_scores = estimator.load_f1_scores(Path(training_dir), seeds)
    trace = estimator.fit(f1_scores)
    summary = estimator.summarise(trace, hdi_prob=hdi_prob)
    summary.to_csv(output)
    logging.info(f"Posterior summary ({hdi_prob:.0%} HDI):\n{summary.to_string()}")
    logging.info(f"Summary saved to: {output}")


def parse_args() -> Namespace:
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "-t",
        "--training-dir",
        dest="training_dir",
        help="Folder containing an rs{seed}/ folder of relevance, sentiment and combined metrics per seed",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-s",
        "--seeds",
        dest="seeds",
        help="Seeds of the splits to include",
        type=int,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        help="Path to write the summary csv to",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--hdi-prob",
        dest="hdi_prob",
        help="Probability mass of the highest density interval",
        type=float,
        default=0.95,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_bayesian_analysis(**vars(args))
