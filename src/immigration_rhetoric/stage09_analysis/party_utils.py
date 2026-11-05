"""Party utilities."""

import logging
from typing import Dict

import pandas as pd
from immigration_rhetoric.stage09_analysis.plot_config import DataConfig

logger = logging.getLogger(__name__)


class PartyUtils:

    @staticmethod
    def _is_party_plotted_at_date(
        party_name: str, date: str, party_windows: Dict[str, Dict[str, str]]
    ) -> bool:
        """
        Check if a party should be plotted at a given date based on specified start/end dates.

        Args:
            party_name: Name of the party to check
            date: Date to check
            party_windows: Dictionary defining start/end dates for parties

        Returns:
            bool: True if party should be plotted at this date, False otherwise
        """
        # If party not in windows dict, it's always valid (default behavior)
        if party_name not in party_windows:
            return True

        window = party_windows[party_name]
        check_date = pd.to_datetime(date)

        # Check start date constraint
        if "start" in window and window["start"]:
            start_date = pd.to_datetime(window["start"])
            if check_date < start_date:
                return False

        # Check end date constraint
        if "end" in window and window["end"]:
            end_date = pd.to_datetime(window["end"])
            if check_date > end_date:
                return False

        return True

    def filter_dataframe_by_party_windows(
        self,
        df: pd.DataFrame,
        date_column: str,
        party_column: str,
        party_windows: Dict[str, Dict[str, str]],
    ) -> pd.DataFrame:
        """
        Mask party labels for rows outside their defined time windows.

        Sets party_label to None for parties outside their start/end date ranges,
        preserving all rows for overall calculations while excluding them from
        party-specific analysis.

        Args:
            df: Input dataframe
            date_column: Name of column containing dates
            party_column: Name of column containing party names
            party_windows: Dictionary mapping party names to dicts with optional
                'start' and 'end' date strings (YYYY-MM-DD)

        Returns:
            pd.DataFrame: Dataframe with party_label set to None where out of window
        """

        if not party_windows:
            return df  # No filtering if no windows defined

        mask = df.apply(
            lambda row: self._is_party_plotted_at_date(
                party_name=row[party_column],
                date=row[date_column],
                party_windows=party_windows,
            ),
            axis=1,
        )

        # Set party_label to None where mask is False to preserve data for
        # overall-level calculations while excluding from party-specific analysis
        df.loc[~mask, "party_label"] = None

        return df

    @staticmethod
    def apply_party_name_mapping(party_label: str) -> str:
        return DataConfig.PARTY_MAPPING.get(party_label, party_label)
