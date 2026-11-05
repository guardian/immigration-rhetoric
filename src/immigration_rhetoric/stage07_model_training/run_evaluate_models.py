from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional

from immigration_rhetoric.stage07_model_training.evaluate_models import ModelEvaluator


def run_evaluate_models(
    relevance_model: str,
    sentiment_model: str,
    relevance_eval: str,
    sentiment_eval: str,
    output_metrics: str,
    output_predictions: str,
    plot: Optional[str] = None,
    gpu_id: int = -1,
) -> None:
    """Evaluate the combined relevance+sentiment models and write metrics (and optionally a confusion matrix plot)."""
    evaluator = ModelEvaluator()
    evaluator.evaluate_models(
        relevance_model=Path(relevance_model),
        sentiment_model=Path(sentiment_model),
        relevance_eval=Path(relevance_eval),
        sentiment_eval=Path(sentiment_eval),
        output_path_metrics=Path(output_metrics),
        output_path_predictions=Path(output_predictions),
        plot=Path(plot) if plot else None,
        gpu_id=gpu_id,
    )


def parse_args() -> Namespace:
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "-r",
        "--relevance-model",
        dest="relevance_model",
        help="Path to the trained relevance model-best directory",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-s",
        "--sentiment-model",
        dest="sentiment_model",
        help="Path to the trained sentiment model-best directory",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--relevance-eval",
        dest="relevance_eval",
        help="Path to the relevance evaluation .spacy DocBin",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--sentiment-eval",
        dest="sentiment_eval",
        help="Path to the sentiment evaluation .spacy DocBin",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-m",
        "--output-metrics",
        dest="output_metrics",
        help="Path to write the combined evaluation metrics json to",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-predictions",
        dest="output_predictions",
        help="Path to write the combined actual-vs-predicted labels jsonl to",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-p",
        "--plot",
        dest="plot",
        help="Path to write a confusion matrix plot (png) to",
        type=str,
        default=None,
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
    run_evaluate_models(**vars(args))
