"""Sentiment calculation utilities for immigration rhetoric analysis."""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from immigration_rhetoric.stage09_analysis.plot_config import DataConfig

logger = logging.getLogger(__name__)


class SentimentCalculator:
    """Calculates sentiment scores over time periods, ignoring any snippets
    not about immigration.

    `corrections` maps each predicted label to the share of its snippets that have each true label, and is used
    to calculate corrected scores. It defaults to `DataConfig.COUNT_CORRECTIONS`.
    """

    def __init__(
        self,
        corrections: Optional[Dict[str, Dict[str, float]]] = None,
        data_config: Optional[DataConfig] = None,
    ):
        self.data_config = data_config or DataConfig()
        self.corrections = corrections or self.data_config.COUNT_CORRECTIONS

    def _get_sentiment_counts(
        self, series: pd.Series, corrected: bool = False
    ) -> Dict[str, float]:
        """Get immigration sentiment counts, estimating the true label counts from the predicted ones if
        `corrected`."""
        # Get counts for each label
        label_order = [
            self.data_config.POSITIVE_LABEL,
            self.data_config.NEUTRAL_LABEL,
            self.data_config.NEGATIVE_LABEL,
            self.data_config.NOT_ABOUT_LABEL,
        ]
        counts = np.array(
            [series.value_counts().get(label, 0) for label in label_order]
        )

        if corrected:

            # Build correction matrix (4x4: predicted x true)
            correction_matrix = np.array(
                [
                    [self.corrections[predicted][true] for true in label_order]
                    for predicted in label_order
                ]
            )

            # Apply correction matrix, transposing first to get the coefficients in the right place
            correction_matrix_T = correction_matrix.T
            corrected_counts = correction_matrix_T @ counts

        else:
            corrected_counts = counts.copy()

        corrected_pos, corrected_neu, corrected_neg, _ = corrected_counts

        return {
            "pos": corrected_pos,
            "neg": corrected_neg,
            "total": corrected_pos + corrected_neu + corrected_neg,
        }

    def percent_pos_minus_neg(
        self, series: pd.Series, corrected: bool = False
    ) -> float:
        """Calculate percentage of positive minus negative immigration sentiment,
        applying corrections if specified."""
        counts = self._get_sentiment_counts(series, corrected)
        if counts["total"] == 0:
            return np.nan
        return ((counts["pos"] - counts["neg"]) / counts["total"]) * 100
