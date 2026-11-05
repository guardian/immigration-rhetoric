"""Plotter for frames analysis visualizations."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from immigration_rhetoric.stage09_analysis.date_utils import DateTimeUtils
from immigration_rhetoric.stage09_analysis.frames_processor import FramesDataProcessor
from immigration_rhetoric.stage09_analysis.plot_config import PlotConfig, StyleConfig
from immigration_rhetoric.stage09_analysis.plot_styler import PlotStyler
from immigration_rhetoric.stage09_analysis.sentiment_plotter import SentimentPlotter

logger = logging.getLogger(__name__)


class FramesPlotter:
    """Plotter for frames time-series percentages (single axis, any number of frames)."""

    def __init__(self):
        self.processor = FramesDataProcessor()
        self.styler = PlotStyler()
        # Reuse finalization from SentimentPlotter for consistent saving/showing
        self._finalize_plot = SentimentPlotter()._finalize_plot

    def plot_frames(
        self,
        df: pd.DataFrame,
        title: str,
        frames_dict: Dict[str, List[str]],
        frames: Optional[List[str]] = None,
        config: Optional[PlotConfig] = None,
        timeseries: str = "year",
        save_path: Optional[Union[Path, str]] = None,
    ) -> None:
        """Plot yearly percentage of snippets containing one or more frames on a single axis.

        Args:
            df: DataFrame with 'date', 'snippet_text', 'relevance_label'.
            title: Plot title.
            frames_dict: Raw Card et al. frames definitions.
            frames: Frame names to plot. Defaults to all.
            config: Plot configuration.
            timeseries: Time grouping (typically 'year').
            save_path: Where to save the figure.
        """

        if not config:
            config = PlotConfig()
        assert config is not None

        if frames is None:
            frames = list(frames_dict.keys())
        if len(frames) == 0:
            raise ValueError("No frames provided or available in frames_dict")

        # Clean frames dictionary (split/trim; stemming handled in processor)
        cleaned = self.processor.clean_frames_dict(frames_dict)

        # Compute percentages over time (forced stemming inside find_frames)
        df_long = self.processor.frames_share_over_time(
            df=df,
            frames=frames,
            all_frames=cleaned,
            timeseries=timeseries,
        )

        if df_long.empty:
            logger.warning("No frame matches found in the provided data")
            return

        # Create figure
        fig, ax = config.create_figure()

        # Colour cycle
        color_list = list(StyleConfig.FRAME_COLOURS.values())

        # Plot all frames on a single y-axis
        for i, frame_name in enumerate(frames):
            df_f = df_long[df_long["Frame"] == frame_name].sort_values(timeseries)
            ax.plot(
                df_f[timeseries].values,
                df_f["Percentage"].values,
                label=frame_name,
                color=color_list[i % len(color_list)],
                linewidth=config.line_width,
            )
        ax.set_ylabel("Percentage of snippets (%)")

        # Axes formatting using shared styler for consistency
        dated = DateTimeUtils.create_datetime_column(
            df=df_long[[timeseries]].drop_duplicates().sort_values(timeseries),
            source_col=timeseries,
            target_col="timeseries_dt",
        )
        x0, x1 = config.get_xlim_timestamps(
            default_x0=dated.timeseries_dt.min(),
            default_x1=dated.timeseries_dt.max(),
        )
        self.styler.setup_axes(ax=ax, x0=x0, x1=x1, ylabel=title)

        # Finalise
        self._finalize_plot(fig=fig, save_path=save_path, config=config)
