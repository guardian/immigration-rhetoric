from pathlib import Path
from unittest.mock import patch

from immigration_rhetoric.stage07_model_training.run_evaluate_models import (
    parse_args,
    run_evaluate_models,
)

REQUIRED_ARGS = [
    "script.py",
    "-r",
    "training/relevance/model-best",
    "-s",
    "training/sentiment/model-best",
    "--relevance-eval",
    "corpus/eval_relevance.spacy",
    "--sentiment-eval",
    "corpus/eval_sentiment.spacy",
    "-m",
    "metrics.json",
    "-o",
    "predictions.jsonl",
]


class TestParseArgs:
    def test_plot_defaults_to_none(self):
        with patch("sys.argv", REQUIRED_ARGS):
            args = parse_args()

        assert args.relevance_model == "training/relevance/model-best"
        assert args.plot is None
        assert args.gpu_id == -1


class TestRunEvaluateModels:
    @patch(
        "immigration_rhetoric.stage07_model_training.run_evaluate_models.ModelEvaluator"
    )
    def test_delegates_with_paths(self, mock_evaluator_cls):
        run_evaluate_models(
            relevance_model="rel",
            sentiment_model="sent",
            relevance_eval="rel.spacy",
            sentiment_eval="sent.spacy",
            output_metrics="metrics.json",
            output_predictions="predictions.jsonl",
            plot="cm.png",
            gpu_id=0,
        )

        mock_evaluator_cls.return_value.evaluate_models.assert_called_once_with(
            relevance_model=Path("rel"),
            sentiment_model=Path("sent"),
            relevance_eval=Path("rel.spacy"),
            sentiment_eval=Path("sent.spacy"),
            output_path_metrics=Path("metrics.json"),
            output_path_predictions=Path("predictions.jsonl"),
            plot=Path("cm.png"),
            gpu_id=0,
        )
