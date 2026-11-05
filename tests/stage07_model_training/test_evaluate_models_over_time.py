from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from immigration_rhetoric.stage07_model_training.evaluate_models_over_time import (
    PerformanceOverTimeEvaluator,
)

matplotlib.use("Agg")

POS = "POSITIVE_TOWARDS_IMMIGRATION"
NEU = "NEUTRAL_TOWARDS_IMMIGRATION"
NEG = "NEGATIVE_TOWARDS_IMMIGRATION"
OFF = "NOT_ABOUT_IMMIGRATION"


def _write_predictions(path, rows):
    pd.DataFrame(
        [
            {
                "date": date,
                "text": f"Snippet {i}",
                "predicted_combined_label": predicted,
                "predicted_combined_probs": {predicted: 1.0},
                "true_combined_label": true,
            }
            for i, (date, true, predicted) in enumerate(rows)
        ]
    ).to_json(path, orient="records", lines=True)


@pytest.fixture
def evaluator():
    return PerformanceOverTimeEvaluator()


@pytest.fixture
def predictions(tmp_path, evaluator):
    path = tmp_path / "predictions.jsonl"
    _write_predictions(
        path,
        [
            ("1975-03-01", POS, POS),
            ("1975-06-01", NEG, POS),
            ("1978-01-01", OFF, OFF),
            ("2001-01-01", NEU, NEU),
            ("2001-05-01", NEG, NEG),
            ("2003-01-01", POS, NEG),
            ("2022-01-01", OFF, OFF),
            (None, POS, POS),
        ],
    )
    return evaluator.load_predictions(path)


class TestLoadPredictions:
    def test_drops_undated_and_adds_columns(self, predictions):
        assert len(predictions) == 7
        assert predictions["year"].dtype == int
        assert predictions["decade"].tolist() == [
            1970,
            1970,
            1970,
            2000,
            2000,
            2000,
            2020,
        ]
        assert predictions["is_error"].tolist() == [
            False,
            True,
            False,
            False,
            False,
            True,
            False,
        ]

    def test_leaves_out_snippets_before_min_year(self, tmp_path, evaluator):
        path = tmp_path / "predictions.jsonl"
        _write_predictions(
            path,
            [
                ("1919-05-01", POS, POS),
                ("1923-01-01", NEG, NEG),
                ("2001-01-01", NEU, NEU),
            ],
        )

        df = evaluator.load_predictions(path, min_year=1923)

        assert df["year"].tolist() == [1923, 2001]


class TestCalculateScores:
    def test_macro_scores_as_percentages(self, predictions):
        scores = PerformanceOverTimeEvaluator.calculate_scores(
            predictions[predictions["decade"] == 1970]
        )

        # 1970s: POSITIVE has precision 0.5, recall 1; NOT_ABOUT is perfect; NEUTRAL and NEGATIVE score 0
        assert scores["Precision"] == pytest.approx((0.5 + 0 + 0 + 1.0) / 4 * 100)
        assert scores["Recall"] == pytest.approx((1.0 + 0 + 0 + 1.0) / 4 * 100)
        assert scores["FScore"] == pytest.approx((2 / 3 + 0 + 0 + 1.0) / 4 * 100)


class TestCalculateDecadeMetrics:
    def test_counts_errors_and_scores_per_decade(self, evaluator, predictions):
        counts, errors, scores = evaluator.calculate_decade_metrics(predictions)

        assert counts.to_dict() == {1970: 3, 2000: 3, 2020: 1}
        assert errors.to_dict() == {1970: 1, 2000: 1, 2020: 0}
        assert list(scores.index) == [1970, 2000, 2020]
        assert list(scores.columns) == ["FScore", "Precision", "Recall"]


class TestPlotOverallByDecade:
    def test_saves_png(self, tmp_path, evaluator, predictions):
        save_path = tmp_path / "overall_by_decade.png"

        evaluator.plot_overall_by_decade(predictions, save_path)

        assert save_path.exists()

    def test_x_axis_starts_at_the_first_decade(self, tmp_path, evaluator, predictions):
        # Keep the figure open so its axes can be inspected
        with patch.object(plt, "close"):
            evaluator.plot_overall_by_decade(predictions, tmp_path / "plot.png")
        ax = plt.gcf().axes[0]

        assert ax.get_xlim()[0] == 1970
        assert ax.get_xticks()[0] == 1970


class TestAlignYAxisLimits:
    def test_ticks_line_up(self):
        fig, ax = plt.subplots()
        ax_scores = ax.twinx()
        ax.plot([0, 1], [0, 37])
        ax_scores.plot([0, 1], [40, 70])

        PerformanceOverTimeEvaluator.align_y_axis_limits(ax, ax_scores)

        count_ticks, score_ticks = ax.get_yticks(), ax_scores.get_yticks()
        assert len(count_ticks) == len(score_ticks)
        assert count_ticks[0] == score_ticks[0] == 0
        assert score_ticks[-1] == 80
        count_ratio = count_ticks[-1] / ax.get_ylim()[1]
        assert score_ticks[-1] / ax_scores.get_ylim()[1] == pytest.approx(count_ratio)
        plt.close(fig)
