"""Ternary plots of the sentiment of the terms used in snippets."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import ternary
from adjustText import adjust_text
from immigration_rhetoric.stage09_analysis.plot_config import TernaryStyleConfig
from immigration_rhetoric.stage09_analysis.ternary_processor import (
    TernaryDataProcessor,
)
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


class TernaryPlotter:
    """Plots each term at its shares of positive, neutral and negative TF-IDF score."""

    def __init__(
        self,
        processor: Optional[TernaryDataProcessor] = None,
        style_config: Optional[TernaryStyleConfig] = None,
    ):
        self.processor = processor or TernaryDataProcessor()
        self.style = style_config or TernaryStyleConfig()

    def plot_term_sentiment(
        self,
        df: pd.DataFrame,
        min_df: int,
        save_path: Union[Path, str],
        periods: Optional[List[Tuple[int, int]]] = None,
        min_total_score: float = 5.0,
    ) -> None:
        """Make a ternary plot of term sentiment.

        Args:
            df: DataFrame with 'date', 'label' and 'snippet_text' columns
            min_df: Minimum number of snippets a term must appear in
            save_path: Where to save the plot
            periods: If given, one plot per (start year, end year) period instead, saved with a
                `_{start}-{end}` suffix
            min_total_score: Terms with this total TF-IDF score or less aren't plotted
        """
        save_path = Path(save_path)

        if periods is None:
            scores = self.processor.compute_scores(df, min_df)
            self._plot_single(*scores, save_path, min_total_score)
            return

        years = df["date"].dt.year
        for start, end in periods:
            period_df = df[(years >= start) & (years <= end)]
            if period_df.empty:
                logger.warning(f"Skipping period {start}-{end}: no data")
                continue
            scores = self.processor.compute_scores(period_df, min_df)
            period_path = save_path.with_name(
                f"{save_path.stem}_{start}-{end}{save_path.suffix}"
            )
            self._plot_single(*scores, period_path, min_total_score)

    # Ternary plots use matplotlib's default style, not the style set for the other figures
    @plt.style.context("default")
    def _plot_single(
        self,
        positive: Dict[str, float],
        neutral: Dict[str, float],
        negative: Dict[str, float],
        doc_freqs: Dict[str, int],
        save_path: Path,
        min_total_score: float,
    ) -> None:
        """Plot each term's (positive, neutral, negative) shares of its total score, annotating the terms
        nearest each corner and the centre with the number of snippets they appear in.
        """
        points = []
        colours = []
        terms = []
        for word in sorted(set(positive) | set(neutral) | set(negative)):
            pos = positive.get(word, 0.0)
            neu = neutral.get(word, 0.0)
            neg = negative.get(word, 0.0)
            total = pos + neg + neu
            if total <= min_total_score:
                continue

            coord = (pos / total, neu / total, neg / total)
            points.append(coord)
            if coord[0] > coord[2] and coord[0] > coord[1]:
                colours.append(self.style.POSITIVE_COLOUR)
            elif coord[2] > coord[0] and coord[2] > coord[1]:
                colours.append(self.style.NEGATIVE_COLOUR)
            else:
                colours.append(self.style.NEUTRAL_COLOUR)
            terms.append({"word": word, "coord": coord})

        if not points:
            logger.warning("No terms passed thresholds for plotting; skipping figure.")
            return

        fig, tax = ternary.figure(scale=1.0)
        fig.set_size_inches(*self.style.FIGSIZE)
        tax.boundary(linewidth=1.0)
        tax.gridlines(multiple=0.2, color="blue", alpha=0.1)

        corner_label_style = {
            "fontsize": self.style.CORNER_LABEL_FONTSIZE,
            "offset": self.style.CORNER_LABEL_OFFSET,
            "fontweight": self.style.CORNER_LABEL_FONTWEIGHT,
        }
        tax.right_corner_label(
            "Positive", color=self.style.POSITIVE_COLOUR, **corner_label_style
        )
        tax.top_corner_label(
            "Neutral", color=self.style.NEUTRAL_COLOUR, **corner_label_style
        )
        tax.left_corner_label(
            "Negative", color=self.style.NEGATIVE_COLOUR, **corner_label_style
        )

        tax.scatter(
            points,
            marker="o",
            color=colours,
            s=self.style.SCATTER_SIZE,
            alpha=self.style.SCATTER_ALPHA,
        )

        # The most extreme terms near each corner
        corner_terms = []
        for i in range(3):
            corner_terms += sorted(
                [t for t in terms if t["coord"][i] > 0.5],
                key=lambda t: (-t["coord"][i], t["word"]),
            )[: self.style.CORNER_ANNOTATION_COUNT]
        corner_words = {t["word"] for t in corner_terms}

        # The most balanced of the remaining terms
        centre_terms = sorted(
            [t for t in terms if t["word"] not in corner_words],
            key=lambda t: (
                sum((c - 1 / 3) ** 2 for c in t["coord"]) ** 0.5,
                t["word"],
            ),
        )[: self.style.CENTRE_ANNOTATION_COUNT]

        for annotated, fontsize, colour, facecolour, alpha in [
            (
                corner_terms,
                self.style.CORNER_ANNOTATION_FONTSIZE,
                self.style.CORNER_ANNOTATION_COLOUR,
                "white",
                0.8,
            ),
            (
                centre_terms,
                self.style.CENTRE_ANNOTATION_FONTSIZE,
                self.style.CENTRE_ANNOTATION_COLOUR,
                "#f6f6f6",
                0.9,
            ),
        ]:
            for t in annotated:
                word = t["word"]
                label = f"{word} ({doc_freqs[word]})" if word in doc_freqs else word
                tax.annotate(
                    label,
                    t["coord"],
                    fontsize=fontsize,
                    fontweight="bold",
                    color=colour,
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor=facecolour,
                        edgecolor="none",
                        alpha=alpha,
                    ),
                    zorder=5,
                )

        # python-ternary's annotate doesn't return the annotations, and so far they're the axes' only texts
        ax = tax.get_axes()
        adjust_text(
            list(ax.texts),
            ax=ax,
            expand=self.style.ADJUST_TEXT_EXPAND,
            force_text=self.style.ADJUST_TEXT_FORCE,
            iter_lim=self.style.ADJUST_TEXT_ITER_LIM,
            only_move={"static": "y", "text": "xy"},
            arrowprops=dict(
                arrowstyle="-",
                color=self.style.ADJUST_TEXT_ARROW_COLOUR,
                lw=self.style.ADJUST_TEXT_ARROW_WIDTH,
                alpha=self.style.ADJUST_TEXT_ARROW_ALPHA,
            ),
        )

        tax.ticks(
            axis="lbr", linewidth=1, multiple=0.1, tick_formats="%.1f", offset=0.03
        )
        tax.clear_matplotlib_ticks()
        ax.axis("off")

        save_path.parent.mkdir(parents=True, exist_ok=True)
        tax.savefig(save_path, bbox_inches="tight", dpi=self.style.DPI)
        plt.close(fig)
        logger.info(f"Ternary plot saved to {save_path}")
