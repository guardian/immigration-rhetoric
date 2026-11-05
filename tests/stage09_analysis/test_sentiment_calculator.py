import numpy as np
import pandas as pd
import pytest
from immigration_rhetoric.stage09_analysis.plot_config import DataConfig
from immigration_rhetoric.stage09_analysis.sentiment_calculator import (
    SentimentCalculator,
)
from tests.stage09_analysis.conftest import NEG, NEU, NOT, POS, identity_corrections

LABELS = pd.Series([POS, POS, NEU, NEG, NOT])


def corrections_with(predicted, shares):
    corrections = identity_corrections()
    corrections[predicted] = shares
    return corrections


class TestUncorrected:
    def test_net_sentiment_ignores_not_about(self):
        assert SentimentCalculator().percent_pos_minus_neg(LABELS) == pytest.approx(25)

    def test_no_snippets_about_immigration_is_nan(self):
        assert np.isnan(SentimentCalculator().percent_pos_minus_neg(pd.Series([NOT])))


class TestCorrected:
    def test_moves_predicted_counts_to_true_labels(self):
        # Half of predicted not-about snippets are truly positive
        corrections = corrections_with(NOT, {POS: 0.5, NEU: 0.0, NEG: 0.0, NOT: 0.5})
        calculator = SentimentCalculator(corrections=corrections)

        # pos 2.5, neu 1, neg 1
        assert calculator.percent_pos_minus_neg(
            LABELS, corrected=True
        ) == pytest.approx(1.5 / 4.5 * 100)

    def test_identity_corrections_change_nothing(self, corrections):
        calculator = SentimentCalculator(corrections=corrections)

        assert calculator.percent_pos_minus_neg(
            LABELS, corrected=True
        ) == pytest.approx(calculator.percent_pos_minus_neg(LABELS))

    def test_uncorrected_by_default(self):
        corrections = corrections_with(NEG, {POS: 0.0, NEU: 1.0, NEG: 0.0, NOT: 0.0})

        assert SentimentCalculator(corrections=corrections).percent_pos_minus_neg(
            LABELS
        ) == pytest.approx(25)

    def test_defaults_to_the_published_corrections(self):
        calculator = SentimentCalculator()

        assert calculator.corrections == DataConfig.COUNT_CORRECTIONS
        for shares in calculator.corrections.values():
            assert sum(shares.values()) == pytest.approx(1, abs=1e-3)
