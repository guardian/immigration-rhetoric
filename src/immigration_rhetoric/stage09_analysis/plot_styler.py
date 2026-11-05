"""Plot styling utilities for sentiment analysis visualizations."""

import logging
from typing import Dict, Optional

import pandas as pd
from immigration_rhetoric.stage09_analysis.plot_config import PlotConfig, StyleConfig
from matplotlib.dates import DateFormatter, YearLocator

logger = logging.getLogger(__name__)


class PlotStyler:
    """Handles plot styling and formatting."""

    def __init__(
        self,
        plot_config: Optional[PlotConfig] = None,
        style_config: Optional[StyleConfig] = None,
    ):
        self.plot_config = plot_config or PlotConfig()
        self.style_config = style_config or StyleConfig()
        self.party_colours = self.style_config.PARTY_COLOURS

    def setup_axes(
        self, ax, x0: pd.Timestamp, x1: pd.Timestamp, ylabel: Optional[str]
    ) -> None:
        """Apply standard styling to plot axes."""
        ax.set_xlim(x0, x1)
        ax.axhline(
            0,
            color=self.style_config.ZERO_LINE_COLOUR,
            linewidth=self.style_config.ZERO_LINE_WIDTH,
            zorder=self.style_config.ZERO_LINE_ZORDER,
        )  # Put behind plot lines
        if ylabel is not None:
            ax.set_title(ylabel, y=1.0, pad=50)
        ax.set_xlabel("Year")
        ax.xaxis.set_major_locator(YearLocator(5))
        ax.xaxis.set_major_formatter(DateFormatter("%Y"))
        ax.legend(
            loc=self.plot_config.legend_location,
            bbox_to_anchor=self.plot_config.legend_bbox,
            ncol=self.plot_config.legend_columns,
        )
        ax.grid(True, alpha=self.plot_config.grid_alpha)

    def add_event_markers(
        self,
        ax,
        events: Dict[str, str],
        x0: pd.Timestamp,
        x1: pd.Timestamp,
    ) -> None:
        """Add vertical lines and labels for events."""
        # Filter for events within the x-axis range
        filtered_events = {
            dt: nm
            for dt, nm in events.items()
            if x0 <= pd.Timestamp(dt, tz="UTC") <= x1
        }

        for i, (dt, nm) in enumerate(filtered_events.items(), 1):
            try:
                event_dt = pd.Timestamp(dt, tz="UTC")
                ax.axvline(
                    event_dt,
                    color=self.party_colours["Other"],
                    linestyle=self.style_config.EVENT_LINE_STYLE,
                    lw=self.style_config.EVENT_LINE_WIDTH,
                    alpha=self.style_config.EVENT_LINE_ALPHA,
                )
                ax.text(
                    event_dt,
                    ax.get_ylim()[1] * 1.01,
                    str(i),
                    ha="center",
                    fontsize=self.style_config.EVENT_TEXT_FONTSIZE,
                    fontweight=self.style_config.EVENT_TEXT_FONTWEIGHT,
                )
            except Exception as e:
                logger.warning(f"Could not plot event {nm}: {e}")

    def add_shaded_region(
        self,
        ax,
        start: pd.Timestamp,
        end: pd.Timestamp,
        colour: str = "#808080",
        alpha: float = 0.25,
    ) -> None:
        """Add shaded region to chart."""

        try:
            x_start = pd.to_datetime(start, utc=True, errors="coerce")
            x_end = pd.to_datetime(end, utc=True, errors="coerce")
            if start <= end:
                ax.axvspan(x_start, x_end, color=colour, alpha=alpha, zorder=0)
            else:
                logger.warning(
                    f"Start date {start} is after end date {end}. Skipping shaded region."
                )

        except Exception as e:
            logger.error(f"Error adding shaded region: {e}")
