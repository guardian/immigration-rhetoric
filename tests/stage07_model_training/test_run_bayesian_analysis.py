from pathlib import Path
from unittest.mock import patch

from immigration_rhetoric.stage07_model_training.run_bayesian_analysis import (
    parse_args,
    run_bayesian_analysis,
)


class TestParseArgs:
    def test_parses_seeds_as_ints_and_defaults_hdi_prob(self):
        with patch(
            "sys.argv",
            ["script.py", "-t", "training", "-s", "17", "23", "-o", "summary.csv"],
        ):
            args = parse_args()

        assert args.training_dir == "training"
        assert args.seeds == [17, 23]
        assert args.hdi_prob == 0.95


class TestRunBayesianAnalysis:
    @patch(
        "immigration_rhetoric.stage07_model_training.run_bayesian_analysis.PerformanceUncertaintyEstimator"
    )
    def test_loads_fits_and_writes_summary(self, mock_estimator_cls, tmp_path):
        estimator = mock_estimator_cls.return_value
        output = tmp_path / "summary.csv"

        run_bayesian_analysis(
            training_dir="training", seeds=[17, 23], output=str(output), hdi_prob=0.9
        )

        estimator.load_f1_scores.assert_called_once_with(Path("training"), [17, 23])
        estimator.fit.assert_called_once_with(estimator.load_f1_scores.return_value)
        estimator.summarise.assert_called_once_with(
            estimator.fit.return_value, hdi_prob=0.9
        )
        estimator.summarise.return_value.to_csv.assert_called_once_with(str(output))
