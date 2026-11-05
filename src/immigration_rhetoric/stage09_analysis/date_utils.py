"""Datetime utilities."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DateTimeUtils:
    """Utilities for datetime operations."""

    @staticmethod
    def create_datetime_column(
        df: pd.DataFrame, source_col: str, target_col: str, utc: bool = True
    ) -> pd.DataFrame:
        """Create a datetime column (`target_col`) from a source column that may be a string or an integer."""
        try:
            df_copy = df.copy()

            if source_col not in df_copy.columns:
                raise ValueError(f"Source column '{source_col}' not found in DataFrame")

            if source_col == "year":
                # Convert to string if necessary, so conversion to datetime works
                df_copy[target_col] = pd.to_datetime(
                    df_copy[source_col].astype(str), utc=utc
                )
            else:
                df_copy[target_col] = pd.to_datetime(df_copy[source_col], utc=utc)

            logger.debug(f"Datetime column '{target_col}' created successfully.")
            return df_copy

        except pd.errors.OutOfBoundsDatetime as e:
            logger.error(f"Date out of bounds error when creating '{target_col}': {e}")
            raise
        except (ValueError, TypeError) as e:
            logger.error(f"Error converting '{source_col}' to datetime: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error creating datetime column '{target_col}': {e}"
            )
            raise

    @staticmethod
    def localise_datetime_series(series: pd.Series) -> pd.Series:
        """Remove timezone from datetime series for plotting."""
        logger.debug("Localising datetime series (removing timezone).")
        return series.dt.tz_localize(None)
