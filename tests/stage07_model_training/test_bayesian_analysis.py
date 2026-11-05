import json

import pytest
from immigration_rhetoric.stage07_model_training.bayesian_analysis import (
    SUMMARY_ORDER,
    PerformanceUncertaintyEstimator,
    beta_from_mean_sd,
)

SEEDS = [17, 23]


def _write_metrics(training_dir, seed, offset):
    per_type = {
        label: {"p": 0.0, "r": 0.0, "f": 0.6 + offset + i / 100}
        for i, label in enumerate(
            [
                "POSITIVE_TOWARDS_IMMIGRATION",
                "NEUTRAL_TOWARDS_IMMIGRATION",
                "NEGATIVE_TOWARDS_IMMIGRATION",
                "NOT_ABOUT_IMMIGRATION",
            ]
        )
    }
    metrics = {
        "relevance": {"cats_macro_f": 0.9 + offset},
        "sentiment": {"cats_macro_f": 0.66 + offset},
        "combined": {"cats_macro_f": 0.67 + offset, "cats_f_per_type": per_type},
    }
    for model, values in metrics.items():
        model_dir = training_dir / f"rs{seed}" / model
        model_dir.mkdir(parents=True)
        (model_dir / "metrics.json").write_text(json.dumps(values))


class TestBetaFromMeanSd:
    def test_parameters_give_requested_mean_and_sd(self):
        alpha, beta = beta_from_mean_sd(0.65, 0.05)

        mean = alpha / (alpha + beta)
        var = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
        assert mean == pytest.approx(0.65)
        assert var**0.5 == pytest.approx(0.05)


class TestLoadF1Scores:
    def test_reads_each_metric_per_seed_in_seed_order(self, tmp_path):
        _write_metrics(tmp_path, 17, offset=0.0)
        _write_metrics(tmp_path, 23, offset=0.01)

        f1_scores = PerformanceUncertaintyEstimator.load_f1_scores(tmp_path, SEEDS)

        assert f1_scores["overall"] == pytest.approx([0.67, 0.68])
        assert f1_scores["relevance"] == pytest.approx([0.9, 0.91])
        assert f1_scores["sentiment"] == pytest.approx([0.66, 0.67])
        assert f1_scores["positive"] == pytest.approx([0.6, 0.61])
        assert f1_scores["neutral"] == pytest.approx([0.61, 0.62])
        assert f1_scores["negative"] == pytest.approx([0.62, 0.63])
        assert f1_scores["not_about"] == pytest.approx([0.63, 0.64])

    def test_missing_seed_raises(self, tmp_path):
        _write_metrics(tmp_path, 17, offset=0.0)

        with pytest.raises(FileNotFoundError):
            PerformanceUncertaintyEstimator.load_f1_scores(tmp_path, SEEDS)


class TestFitAndSummarise:
    def test_summary_has_one_row_per_metric_with_requested_interval(self, tmp_path):
        _write_metrics(tmp_path, 17, offset=0.0)
        _write_metrics(tmp_path, 23, offset=0.01)
        estimator = PerformanceUncertaintyEstimator(draws=50, tune=50, chains=1)

        trace = estimator.fit(estimator.load_f1_scores(tmp_path, SEEDS))
        summary = estimator.summarise(trace, hdi_prob=0.95)

        assert list(summary.index) == [f"mu_{name}" for name in SUMMARY_ORDER]
        assert {"mean", "sd", "hdi_2.5%", "hdi_97.5%", "r_hat"} <= set(summary.columns)
        assert summary.loc["mu_relevance", "mean"] == pytest.approx(0.905, abs=0.05)
