from pathlib import Path
from unittest.mock import patch

from immigration_rhetoric.stage07_model_training.run_evaluate_models_over_time import (
    parse_args,
    run_evaluate_models_over_time,
)

MODULE = "immigration_rhetoric.stage07_model_training.run_evaluate_models_over_time"


class TestParseArgs:
    def test_parses_paths_and_defaults_min_year(self):
        with patch(
            "sys.argv", ["script.py", "-i", "predictions.jsonl", "-p", "plot.png"]
        ):
            args = parse_args()

        assert args.predictions == "predictions.jsonl"
        assert args.plot == "plot.png"
        assert args.min_year == 1923


class TestRunEvaluateModelsOverTime:
    @patch(f"{MODULE}.PerformanceOverTimeEvaluator")
    def test_loads_then_plots(self, mock_evaluator_cls):
        evaluator = mock_evaluator_cls.return_value

        run_evaluate_models_over_time(
            predictions="predictions.jsonl", plot="plot.png", min_year=1950
        )

        evaluator.load_predictions.assert_called_once_with(
            Path("predictions.jsonl"), min_year=1950
        )
        evaluator.plot_overall_by_decade.assert_called_once_with(
            evaluator.load_predictions.return_value, Path("plot.png")
        )
