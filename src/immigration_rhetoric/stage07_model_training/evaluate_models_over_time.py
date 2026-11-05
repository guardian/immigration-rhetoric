"""Plot how the combined relevance + sentiment model's performance varies with the date of the snippet.

Plots, for each decade, the number of eval snippets and errors alongside the (macro averaged) F1 score, precision
and recall across all four labels. This is the paper's Figure A3.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from immigration_rhetoric.common.logging_utils import TqdmLoggingHandler
from immigration_rhetoric.config import STYLES_DIR
from immigration_rhetoric.stage07_model_training.evaluate_models import LABELS
from sklearn.metrics import f1_score, precision_score, recall_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[TqdmLoggingHandler()],
)

GUARDIAN_BROWN = "#a1845c"
GUARDIAN_PINK = "#bb3b80"
GUARDIAN_ORANGE = "#e05e00"

# (column in the scores DataFrame, legend label, line style), all plotted on the right-hand axis
SCORE_LINES = [
    ("FScore", "F1-Score", "-."),
    ("Precision", "Precision", "dotted"),
    ("Recall", "Recall", "--"),
]

# Nice maxima for the right-hand (score %) axis, the smallest that fits the data is used
SCORE_AXIS_MAXES = [10, 20, 25, 50, 60, 75, 80, 90, 100]


class PerformanceOverTimeEvaluator:
    """Plots the combined models' eval performance by the decade of each snippet."""

    def __init__(self):
        plt.style.use(
            [
                str(STYLES_DIR / "base.mplstyle"),
                str(STYLES_DIR / "immigration.mplstyle"),
            ]
        )

    @staticmethod
    def load_predictions(
        predictions_path: Path, min_year: int | None = None
    ) -> pd.DataFrame:
        """Load the actual-vs-predicted labels written by `ModelEvaluator`, adding year, decade and error columns.

        Snippets before `min_year` are left out, if given. The plot's x-axis starts at the beginning of
        that year's decade.
        """
        logging.info(f"Loading predictions from {predictions_path}")
        df = pd.read_json(predictions_path, lines=True)

        df["year"] = pd.to_datetime(df["date"]).dt.year
        df = df.dropna(subset=["year"]).astype({"year": int})
        if min_year is not None:
            df = df[df["year"] >= min_year].copy()
        df["decade"] = (df["year"] // 10) * 10
        df["is_error"] = df["predicted_combined_label"] != df["true_combined_label"]

        logging.info(
            f"Loaded {len(df)} predictions from {df['year'].min()} to {df['year'].max()}"
        )
        return df

    @staticmethod
    def calculate_scores(df: pd.DataFrame) -> dict[str, float]:
        """Macro averaged F1 score, precision and recall across the four labels, as percentages."""
        y_true = df["true_combined_label"]
        y_pred = df["predicted_combined_label"]
        return {
            name: metric(
                y_true, y_pred, labels=LABELS, average="macro", zero_division=0
            )
            * 100
            for name, metric in [
                ("FScore", f1_score),
                ("Precision", precision_score),
                ("Recall", recall_score),
            ]
        }

    def calculate_decade_metrics(
        self, df: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
        """Snippet count, error count and scores (as %) in each decade."""
        scores = pd.DataFrame.from_dict(
            {
                decade: self.calculate_scores(group)
                for decade, group in df.groupby("decade")
            },
            orient="index",
        )
        counts = df.groupby("decade").size()
        errors = df.groupby("decade")["is_error"].sum()
        return counts, errors, scores

    def plot_overall_by_decade(self, df: pd.DataFrame, save_path: Path) -> None:
        """Plot each decade's counts, errors and scores as flat lines spanning the decade, with the legend
        centred under the title."""
        counts, errors, scores = self.calculate_decade_metrics(df)
        decades = counts.index

        def flat(values: pd.Series) -> list:
            return [value for decade in decades for value in [values[decade]] * 2]

        flat_x = [year for decade in decades for year in (decade, decade + 9)]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax_scores = ax.twinx()
        ax.plot(
            flat_x,
            flat(counts),
            label="Total snippets",
            color=GUARDIAN_BROWN,
            linewidth=2,
        )
        ax.plot(flat_x, flat(errors), label="Errors", color=GUARDIAN_PINK, linewidth=2)
        for column, legend_label, linestyle in SCORE_LINES:
            ax_scores.plot(
                flat_x,
                flat(scores[column]),
                label=legend_label,
                color=GUARDIAN_ORANGE,
                linestyle=linestyle,
                linewidth=2,
            )

        ax.set_ylabel("Count")
        ax.set_xlabel("Year")
        ax_scores.set_ylabel("Score (%)", color=GUARDIAN_ORANGE)
        ax_scores.tick_params(
            axis="y", labelcolor=GUARDIAN_ORANGE, colors=GUARDIAN_ORANGE
        )
        self.align_y_axis_limits(ax, ax_scores)
        ax.grid(True, alpha=0.3)
        ax_scores.grid(False)

        # Start the x-axis at the first decade, with a tick at the start of each
        decade_starts = list(range(min(flat_x), max(flat_x) + 1, 10))
        ax.set_xticks(decade_starts)
        ax.set_xticklabels([str(year) for year in decade_starts])
        ax.set_xlim(decade_starts[0], max(flat_x))

        fig.suptitle(
            "Model performance over time: OVERALL (by decade)",
            x=0.0,
            ha="left",
            fontsize="medium",
            fontweight="medium",
        )
        count_lines, count_labels = ax.get_legend_handles_labels()
        score_lines, score_labels = ax_scores.get_legend_handles_labels()
        fig.legend(
            count_lines + score_lines,
            count_labels + score_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.95),
            ncol=5,
            frameon=False,
        )

        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"Performance over time plot saved to: {save_path}")

    @staticmethod
    def align_y_axis_limits(ax_counts: plt.Axes, ax_scores: plt.Axes) -> None:
        """Set both y-axes to start at 0 with the same number of ticks, so their tick marks line up."""

        def data_max(ax: plt.Axes) -> float:
            return max(
                line.get_ydata().max()
                for line in ax.get_lines()
                if len(line.get_ydata()) > 0
            )

        counts_max = max(data_max(ax_counts) * 1.1, 1)
        ax_counts.set_ylim(0, counts_max)
        ax_counts.yaxis.set_major_locator(
            ticker.MaxNLocator(nbins=6, integer=True, min_n_ticks=3)
        )
        # Draw so the locator's chosen ticks are available
        ax_counts.figure.canvas.draw()
        counts_ticks = sorted(
            {0, *(tick for tick in ax_counts.get_yticks() if 0 <= tick <= counts_max)}
        )
        num_intervals = len(counts_ticks) - 1

        scores_data_max = data_max(ax_scores) * 1.1
        scores_tick_max = next(
            (m for m in SCORE_AXIS_MAXES if m >= scores_data_max), 100
        )
        scores_ticks = [
            i * scores_tick_max / num_intervals for i in range(num_intervals + 1)
        ]
        # Scale the score axis by the same ratio as the count axis, so equal intervals line up
        ratio = scores_ticks[1] / counts_ticks[1]

        ax_counts.set_ylim(0, counts_max)
        ax_counts.set_yticks(counts_ticks)
        ax_scores.set_ylim(0, counts_max * ratio)
        ax_scores.set_yticks(scores_ticks)
        ax_scores.set_yticklabels(
            [
                f"{tick:.0f}%" if tick == int(tick) else f"{tick:.1f}%"
                for tick in scores_ticks
            ]
        )
