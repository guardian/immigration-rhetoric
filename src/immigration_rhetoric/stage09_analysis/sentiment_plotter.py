"""Main sentiment plotting functionality."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from immigration_rhetoric.stage09_analysis.date_utils import DateTimeUtils
from immigration_rhetoric.stage09_analysis.plot_config import (
    DataConfig,
    PlotConfig,
    StyleConfig,
)
from immigration_rhetoric.stage09_analysis.plot_styler import PlotStyler
from immigration_rhetoric.stage09_analysis.sentiment_calculator import (
    SentimentCalculator,
)
from immigration_rhetoric.stage09_analysis.sentiment_processor import (
    SentimentDataProcessor,
)

logger = logging.getLogger(__name__)


class SentimentPlotter:
    """Main class for creating sentiment plots.

    `corrections` default to `DataConfig.COUNT_CORRECTIONS`, and `cache_dir` is where the sensitivity analysis
    caches its term presence matrices.
    """

    def __init__(
        self,
        corrections: Optional[Dict[str, Dict[str, float]]] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.data_config = DataConfig()
        self.processor = SentimentDataProcessor(
            calculator=SentimentCalculator(
                corrections=corrections, data_config=self.data_config
            ),
            data_config=self.data_config,
            cache_dir=cache_dir,
        )
        self.styler = PlotStyler()
        self.style_config = StyleConfig()

    def _sentiment_plot_items(
        self,
        df: pd.DataFrame,
        parties_of_interest: Optional[List[str]],
        timeseries: str,
        include_uncorrected: bool = False,
        label_suffix: str = "",
    ) -> List[dict]:
        """Sentiment over time overall, then for each party, as plot items with a 'value' column to plot."""
        party_data = self.processor.get_party_sentiment_over_time(
            df=df,
            parties_of_interest=parties_of_interest,
            timeseries=timeseries,
            include_uncorrected=include_uncorrected,
        )
        overall = party_data.pop("Overall", pd.DataFrame())
        if overall.empty:
            raise ValueError("Overall sentiment data is empty")

        plot_items = []
        for label, colour_key, dat in [("All Parties", "Other", overall)] + [
            (party, party, dat) for party, dat in party_data.items()
        ]:
            dat = DateTimeUtils.create_datetime_column(
                df=dat, source_col=timeseries, target_col="timeseries_dt"
            )
            dat["value"] = dat["sentiment"].copy()
            plot_items.append(
                {
                    "label": label + label_suffix,
                    "data": dat,
                    "color": self.style_config.get_party_colour(colour_key),
                }
            )
        return plot_items

    def _prepare_timeseries_plot_data(
        self,
        df: pd.DataFrame,
        config: PlotConfig,
        parties_of_interest: Optional[List[str]] = None,
        timeseries: str = "year",
        include_uncorrected: bool = False,
    ) -> dict:
        """Prepare all data needed for timeseries plotting."""
        plot_items = self._sentiment_plot_items(df, parties_of_interest, timeseries)
        overall = plot_items[0]["data"]

        x0, x1 = config.get_xlim_timestamps(
            default_x0=overall.timeseries_dt.min(),
            default_x1=overall.timeseries_dt.max(),
        )

        plot_items_uncorrected = (
            self._sentiment_plot_items(
                df,
                parties_of_interest,
                timeseries,
                include_uncorrected=True,
                label_suffix=" (Uncorrected)",
            )
            if include_uncorrected
            else []
        )

        return {
            "plot_items": plot_items,
            "plot_items_uncorrected": plot_items_uncorrected,
            "x0": x0,
            "x1": x1,
        }

    def _jackknife_ribbon(
        self,
        df: pd.DataFrame,
        all_terms: list,
        timeseries: str,
        drop_percentage: int,
        colour_key: str,
    ) -> dict:
        """Overall sentiment of `df` with all trigger terms, and its min/max across the jackknife runs."""
        runs = self.processor.calculate_term_sensitivity(
            df=df, all_terms=all_terms, drop_percentage=drop_percentage
        )
        baseline, *dropped = [
            self._sentiment_plot_items(run_df, None, timeseries)[0]["data"]
            for run_df in runs.values()
        ]
        values_matrix = np.array([run["value"].values for run in dropped])
        return {
            "data": baseline,
            "min_values": np.min(values_matrix, axis=0),
            "max_values": np.max(values_matrix, axis=0),
            "color": self.style_config.get_party_colour(colour_key),
        }

    def _calculate_jackknife_ribbons(
        self,
        df: pd.DataFrame,
        all_terms: list,
        timeseries: str,
        parties_of_interest: Optional[List[str]],
        drop_percentage: int,
    ) -> dict:
        """Jackknife ribbons overall and for each party, by dropping groups of trigger terms in turn."""
        ribbon_data = {
            "Overall": self._jackknife_ribbon(
                df, all_terms, timeseries, drop_percentage, "Other"
            )
        }
        for party in parties_of_interest or []:
            ribbon_data[party] = self._jackknife_ribbon(
                df[df["party_label"] == party],
                all_terms,
                timeseries,
                drop_percentage,
                party,
            )
        return ribbon_data

    def _create_stacked_bar_plot(
        self,
        plot_data: dict,
        title: str,
        config: PlotConfig,
    ) -> Tuple:
        """Create figure and plot the stacked bars."""
        fig, ax = config.create_figure()

        items = plot_data["plot_items"]
        first_item = items[0]
        overall_dat = first_item["data"]
        overall_label = first_item["label"]
        overall_colour = first_item["color"]
        x = DateTimeUtils.localise_datetime_series(overall_dat["timeseries_dt"])

        # Calculate a sensible width for the bars based on the most common time difference
        time_diffs = [(x.iloc[i + 1] - x.iloc[i]).days for i in range(len(x) - 1)]
        median_diff = pd.Series(time_diffs).median()
        width = int(median_diff * 0.8)

        bottom = pd.Series(0.0, index=overall_dat.index)
        if len(plot_data["plot_items"]) > 1:
            parties = items[1:]
            for item in parties:
                dat = item["data"]
                y = dat["value"]
                ax.bar(
                    x,
                    y,
                    bottom=bottom,
                    color=item.get("color"),
                    label=item.get("label"),
                    alpha=config.bar_alpha,
                    width=width,
                )
                # Update bottom for next stack
                bottom += y
            # Add whatever is left as an additional bar on top
            y = overall_dat["value"]
            remainder = y - bottom
            ax.bar(
                x,
                remainder,
                bottom=bottom,
                color=overall_colour,
                label="Other Parties",
                alpha=config.bar_alpha,
                width=width,
            )
        else:
            # If only overall data is present, plot it as a single bar
            y = overall_dat["value"]
            ax.bar(
                x,
                y,
                color=overall_colour,
                label=overall_label,
                alpha=config.bar_alpha,
                width=width,
            )

        # Apply standard styling
        padding = pd.Timedelta(days=width / 2.0)  # Half bar width as padding
        self.styler.setup_axes(
            ax, plot_data["x0"] - padding, plot_data["x1"] + padding, title
        )

        return fig, ax

    def _create_timeseries_plot(
        self,
        plot_data: dict,
        title: Optional[str],
        config: PlotConfig,
    ) -> Tuple:
        """Create figure and plot the sentiment lines.

        Args:
            plot_data: Dictionary containing plot items and x-axis limits
            title: Plot title, or None for no title
            config: Plot configuration
        """
        fig, ax = config.create_figure()

        # Plot sentiment lines with error bands
        for item in plot_data["plot_items"]:
            dat = item.get("data", None)
            if dat is None:
                raise ValueError("Plot item is missing data")

            x = DateTimeUtils.localise_datetime_series(dat["timeseries_dt"]).values

            if item.get("is_ribbon", False):
                ax.fill_between(
                    x,
                    item["min_values"],
                    item["max_values"],
                    color=item.get("color"),
                    alpha=item.get("ribbon_alpha", 0.25),
                    label="_nolegend_",
                    edgecolor="none",
                )
            else:
                # Plot regular line
                ax.plot(
                    x,
                    dat["value"].astype(float),
                    color=item.get("color", self.style_config.ZERO_LINE_COLOUR),
                    lw=item.get("linewidth", self.style_config.ZERO_LINE_WIDTH),
                    label=item.get("label", None),
                    linestyle=item.get("linestyle", self.style_config.BASELINE_STYLE),
                )

        if plot_data.get("plot_items_uncorrected"):
            added_uncorrected_legend = False
            for item in plot_data["plot_items_uncorrected"]:
                dat = item["data"]
                colour = item["color"]
                x_unc = DateTimeUtils.localise_datetime_series(
                    dat["timeseries_dt"]
                ).values
                y_unc = dat["value"].astype(float)
                # Only add a single legend entry to key all uncorrected lines
                legend_label = (
                    "Uncorrected" if not added_uncorrected_legend else "_nolegend_"
                )
                ax.plot(
                    x_unc,
                    y_unc,
                    color=colour,
                    lw=1.25,
                    linestyle="--",
                    alpha=0.9,
                    label=legend_label,
                )
                if not added_uncorrected_legend:
                    added_uncorrected_legend = True

        # Apply standard styling
        self.styler.setup_axes(ax, plot_data["x0"], plot_data["x1"], title)

        return fig, ax

    def _add_plot_overlays(
        self,
        ax,
        x0: pd.Timestamp,
        x1: pd.Timestamp,
        events: Optional[Dict[str, str]] = None,
        shaded_block: Optional[Dict[str, str]] = None,
    ) -> None:
        """Add an optional shaded block and event markers to the plot."""

        # Add a shaded block (e.g. historical period) underneath other overlays
        if shaded_block:
            self.styler.add_shaded_region(
                ax=ax,
                start=shaded_block.get("start"),
                end=shaded_block.get("end"),
                colour=shaded_block.get("color", None),
                alpha=shaded_block.get("alpha", None),
            )

        # Add event markers
        if events:
            self.styler.add_event_markers(ax=ax, events=events, x0=x0, x1=x1)

    def _finalize_plot(
        self, fig, save_path: Optional[Union[Path, str]], config: PlotConfig
    ) -> None:
        """Apply final formatting and save or show the plot."""
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=config.dpi)
            logger.info(f"Figure saved to {save_path}")
        else:
            plt.show()

    def plot_sentiment_over_time(
        self,
        df: pd.DataFrame,
        title: Optional[str],
        events: Optional[Dict[str, str]] = None,
        parties_of_interest: Optional[List[str]] = None,
        config: Optional[PlotConfig] = None,
        timeseries: str = "year",
        include_uncorrected: bool = False,
        save_path: Optional[Union[Path, str]] = None,
        shaded_block: Optional[Dict[str, str]] = None,
        show_ribbons: bool = False,
        jackknife_drop_percentage: int = 10,
        jackknife_trigger_terms: Optional[list] = None,
        ribbon_alpha: float = 0.25,
    ) -> plt.Figure:
        """
        Create a sentiment plot for political parties at timeseries-level.
        This applies an aggregate correction to the sentiment proportions
        based on misclassification rates between classes.

        Args:
            df: DataFrame with columns ['date', 'label', 'party_label']
            title: Plot title, or None for no title
            events: Dict of events to mark on plot
            parties_of_interest: List of party names to plot
            config: Plot configuration (figsize, dpi, xlim)
            timeseries: Column to use as timeseries for grouping, e.g. "year"
            include_uncorrected: Whether to include uncorrected sentiment data in the plot
            save_path: Optional path to save the figure
            shaded_block: Optional dictionary with 'start', 'end', 'color', 'alpha' for shaded region
            show_ribbons: Whether to show jackknife ribbons
            jackknife_drop_percentage: Percentage of terms to drop for jackknife (default: 10)
            jackknife_trigger_terms: List of trigger terms, needed for jackknife ribbons
            ribbon_alpha: Transparency of ribbons (0-1, default: 0.25)

        Returns:
            matplotlib Figure object
        """
        if not config:
            config = PlotConfig()

        # Assert that config is not None for type checking
        assert config is not None

        # Trigger terms are needed for jackknife
        if show_ribbons and jackknife_trigger_terms is None:
            raise ValueError(
                "jackknife_trigger_terms must be provided when show_ribbons is True"
            )

        # Prepare and process data
        plot_data = self._prepare_timeseries_plot_data(
            df=df,
            config=config,
            parties_of_interest=parties_of_interest,
            timeseries=timeseries,
            include_uncorrected=include_uncorrected,
        )

        # Calculate ribbons if requested
        if show_ribbons:
            ribbon_data = self._calculate_jackknife_ribbons(
                df=df,
                all_terms=jackknife_trigger_terms,
                timeseries=timeseries,
                parties_of_interest=parties_of_interest,
                drop_percentage=jackknife_drop_percentage,
            )

            # Add ribbon items to plot_data (insert at beginning so they're behind lines)
            ribbon_items = []
            for label, ribbon_info in ribbon_data.items():
                ribbon_item = {
                    "label": f"{label} (Jackknife)",
                    "data": ribbon_info["data"],
                    "color": ribbon_info["color"],
                    "linewidth": 0,
                    "linestyle": "",
                    "is_ribbon": True,
                    "min_values": ribbon_info["min_values"],
                    "max_values": ribbon_info["max_values"],
                    "ribbon_alpha": ribbon_alpha,
                }
                ribbon_items.append(ribbon_item)

            # Insert ribbons at the beginning
            plot_data["plot_items"] = ribbon_items + plot_data["plot_items"]

        # Create and configure plot
        fig, ax = self._create_timeseries_plot(
            plot_data=plot_data,
            title=title,
            config=config,
        )

        # Add optional overlays
        self._add_plot_overlays(
            ax=ax,
            events=events,
            x0=plot_data["x0"],
            x1=plot_data["x1"],
            shaded_block=shaded_block,
        )

        # Finalise plot
        self._finalize_plot(fig=fig, save_path=save_path, config=config)

        return fig

    def plot_sentiment_per_term(
        self,
        df: pd.DataFrame,
        title: str,
        x_label: str,
        trigger_terms: List[str],
        text_col: str = "snippet_text",
        min_occurrences: int = 50,
        config: Optional[PlotConfig] = None,
        figsize: Tuple[int, int] = (20, 20),
        save_path: Optional[Union[Path, str]] = None,
    ):
        """
        Create a lollipop plot showing sentiment for trigger terms. This
        does not apply an aggregate correction to the sentiment counts,
        because it is likely that rates vary across the vocabulary, with terms
        associated with more extreme sentiment less likely to be misclassified.

        Args:
            df: DataFrame with sentiment data
            trigger_terms: List of terms to analyze
            text_col: Column containing text to search
            min_occurrences: Minimum occurrences for a term to be included
            config: Plot configuration object
            figsize: Figure size tuple
            save_path: Optional path to save the plot
        """

        # Get sentiment data for trigger terms
        sorted_qualifiers = self.processor.sentiment_per_trigger(
            df=df,
            trigger_terms=trigger_terms,
            text_col=text_col,
            min=min_occurrences,
        )

        if sorted_qualifiers.empty:
            logger.warning(
                "No trigger terms found meeting minimum occurrence threshold"
            )
            return

        # Create the plot
        if not config:
            config = PlotConfig()
        assert config is not None  # for type checking

        # Override figsize because this is a tall plot
        fig, ax = config.create_figure(figsize=figsize)

        # Split into positive and negative
        most_negative = sorted_qualifiers[sorted_qualifiers["sentiment"] < 0]
        most_positive = sorted_qualifiers[sorted_qualifiers["sentiment"] >= 0]
        x_positions_neg = most_negative["sentiment"]
        x_positions_pos = most_positive["sentiment"]
        ylabel_offset = 1

        # Plot the negative values
        y_positions_neg = range(len(most_negative))
        ax.hlines(
            y_positions_neg,
            0,
            x_positions_neg,
            color="#506991",
            linewidth=1,
        )
        ax.plot(x_positions_neg, y_positions_neg, "o", color="#506991", markersize=6)

        # Plot the positive values
        y_positions_pos = range(
            len(most_negative), len(most_negative) + len(most_positive)
        )
        ax.hlines(
            y_positions_pos,
            0,
            x_positions_pos,
            color="#506991",
            linewidth=1,
        )
        ax.plot(x_positions_pos, y_positions_pos, "o", color="#506991", markersize=6)

        # Add text labels at the end of each bar
        for i, (label, _) in enumerate(zip(most_negative.trigger, x_positions_neg)):
            ax.text(ylabel_offset, i, label, va="center", ha="left", fontsize="small")

        for i, (label, _) in enumerate(zip(most_positive.trigger, x_positions_pos)):
            ax.text(
                -ylabel_offset,
                i + len(most_negative),
                label,
                va="center",
                ha="right",
                fontsize="small",
            )

        # Configure the plot
        ax.set_yticks([])

        # Set tight y-limits
        total_bars = len(most_negative) + len(most_positive)
        ax.set_ylim(-0.75, total_bars - 0.5)

        # Set x-limits with some padding
        xmax = max(x_positions_pos.max(), abs(x_positions_neg.min()))
        ax.set_xlim(-1.1 * xmax, 1.1 * xmax)

        # Add vertical line at x=0
        ax.axvline(x=0, color="#121212", linewidth=0.8, alpha=0.7)

        # Style the plot
        ax.tick_params(
            axis="both",
            which="both",
            direction="in",
            right=False,
            top=False,
        )

        # Labels and title
        ax.set_xlabel(x_label)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", alpha=0.7)

        self._finalize_plot(fig=fig, save_path=save_path, config=config)

    def _prepare_prevalence_data(
        self,
        df_trigger_speeches: pd.DataFrame,
        df_all_speeches: pd.DataFrame,
        config: PlotConfig,
        parties_of_interest: Optional[List[str]] = None,
        timeseries: str = "year",
        percentage: bool = False,
        denominator_col: str = "total_count",
    ) -> dict:
        """Prepare data for prevalence plotting."""

        # Get prevalence data
        prevalence_data = self.processor.get_prevalence_over_time(
            df_trigger_speeches=df_trigger_speeches,
            df_all_speeches=df_all_speeches,
            parties_of_interest=parties_of_interest,
            timeseries=timeseries,
        )

        overall = prevalence_data["Overall"]

        parties = prevalence_data.copy()
        parties.pop("Overall", None)

        if overall.empty:
            raise ValueError("Prevalence data is empty")

        # Add datetime columns
        overall = DateTimeUtils.create_datetime_column(
            df=overall, source_col=timeseries, target_col="timeseries_dt"
        )
        for party_name, dat in parties.items():
            parties[party_name] = DateTimeUtils.create_datetime_column(
                df=dat, source_col=timeseries, target_col="timeseries_dt"
            )

        # Set plot limits
        x0, x1 = config.get_xlim_timestamps(
            default_x0=overall.timeseries_dt.min(),
            default_x1=overall.timeseries_dt.max(),
        )

        # Prepare plot items with colors
        plot_items = [
            {
                "label": "All Parties",
                "data": overall.copy(),
                "color": self.style_config.get_party_colour("Other"),
            }
        ]
        for p, dat in parties.items():
            plot_items.append(
                {
                    "label": p,
                    "data": dat.copy(),
                    "color": self.style_config.get_party_colour(p),
                }
            )

        # Prepare plot_data with 'value' column for plotting
        for item in plot_items:
            dat_copy = item["data"]
            if percentage:
                # compute percentage using overall denominator series
                denom = overall[denominator_col].astype(float)
                item["data"] = dat_copy.assign(
                    value=(dat_copy["about_count"].astype(float) / denom * 100)
                )
            else:
                item["data"] = dat_copy.assign(value=dat_copy["about_count"].copy())

        return {
            "plot_items": plot_items,
            "x0": x0,
            "x1": x1,
        }

    def plot_prevalence_over_time(
        self,
        df_trigger_speeches: pd.DataFrame,
        df_all_speeches: pd.DataFrame,
        title: str,
        parties_of_interest: Optional[List[str]] = None,
        config: Optional[PlotConfig] = None,
        timeseries: str = "year",
        percentage: bool = False,
        per_about: bool = False,
        save_path: Optional[Union[Path, str]] = None,
    ) -> None:
        """
        Create a plot showing prevalence of immigration-related speeches over
        time. This does not apply any correction to prevalence proportions,
        because (1) sentiment corrections are aggregate at class level, so we
        cannot get individual snippets reclassified in order to reclassify
        speeches; (2) the missclassification rate for NOT_ABOUT_IMMIGRATION is
        relatively low in any case, so (3) the likelihood that all snippets
        from a speech would flip class is even lower.
        """
        if not config:
            config = PlotConfig()

        # Assert that config is not None for type checking
        assert config is not None

        # Prepare data
        plot_data = self._prepare_prevalence_data(
            df_trigger_speeches=df_trigger_speeches,
            df_all_speeches=df_all_speeches,
            config=config,
            parties_of_interest=parties_of_interest,
            timeseries=timeseries,
            percentage=percentage,
            denominator_col="about_count" if per_about else "total_count",
        )

        # Create and configure plot
        fig, _ = self._create_stacked_bar_plot(
            plot_data=plot_data,
            title=title,
            config=config,
        )

        # Finalize plot
        self._finalize_plot(fig=fig, save_path=save_path, config=config)
