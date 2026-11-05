"""Data processing for frames analysis."""

import logging
import re
from typing import List

import pandas as pd
from immigration_rhetoric.stage09_analysis.sentiment_processor import (
    SentimentDataProcessor,
)
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)


class FramesDataProcessor:
    """Processes frames for matching and aggregation.

    Provides utilities to clean/stem raw frame term definitions and prepare
    dataframes for plotting shares over time.
    """

    def __init__(self):
        self._stemmer = PorterStemmer()
        self._tokenize = word_tokenize
        self._punkt_validated = False

    def _get_stemmer_and_tokenize(self):
        """Return a Porter stemmer and a tokenizer function.

        Requires NLTK's `punkt_tab` tokenizer resource to be available.
        """
        if not self._punkt_validated:
            try:
                self._tokenize("validation_check_for_punkt")
            except LookupError as e:
                raise RuntimeError(
                    "NLTK 'punkt_tab' tokenizer resource not found. "
                    "Run: python -m nltk.downloader punkt_tab"
                ) from e
            self._punkt_validated = True

        return self._stemmer, self._tokenize

    def stem_terms(self, terms: List[str]) -> List[str]:
        """Stem a list of terms using PorterStemmer if available."""
        stemmer, _ = self._get_stemmer_and_tokenize()
        if not terms:
            return []
        return [stemmer.stem(t) for t in terms if t]

    def stem_text_column(
        self,
        df: pd.DataFrame,
        text_col: str = "snippet_text",
        output_col: str = "stemmed_snippet_text",
    ) -> pd.DataFrame:
        """Create a stemmed text column.

        Tokenizes each snippet and stems each token using PorterStemmer,
        joining them back into a space-separated string.
        """
        if text_col not in df.columns:
            raise ValueError(f"Missing required text column: {text_col}")

        stemmer, tokenize = self._get_stemmer_and_tokenize()

        def _stem_text(x: str) -> str:
            if not isinstance(x, str):
                return ""
            tokens = tokenize(x)
            return " ".join(stemmer.stem(w) for w in tokens)

        out = df.copy()
        out[output_col] = out[text_col].apply(_stem_text)
        return out

    def clean_frames_dict(self, frames_dict: dict) -> dict:
        """Clean and stem raw frames definitions.

        The input frames_dict typically has values as single long strings with
        comma-separated terms and part-of-speech annotations in parentheses,
        e.g., "absconder (n), apprehend (v), ...".

        Steps:
        - Remove trailing POS annotations using a simple regex
        - Split by comma and strip whitespace
        - Flatten to a single list of tokens
        - Stem each token with PorterStemmer
        - Drop empty strings

        Args:
            frames_dict: Dict[str, List[str]] where each value list usually
                         contains one long comma-separated string of terms.

        Returns:
            Dict[str, List[str]] with cleaned and stemmed terms per frame.
        """
        if not isinstance(frames_dict, dict):
            raise ValueError("frames_dict must be a dictionary of frames -> terms")

        cleaned_frames: dict = {}

        for frame, terms in frames_dict.items():
            # Ensure terms is iterable
            if terms is None:
                cleaned_frames[frame] = []
                continue

            # Collect raw tokens after removing POS markers and splitting by comma
            raw_tokens: List[str] = []
            for s in terms:
                if not isinstance(s, str):
                    continue
                # Remove POS markers like " (n)", " (v)", " (a)"
                s_no_pos = re.sub(r" \(.\)", "", s)
                # Split by comma and strip whitespace
                pieces = [t.strip() for t in s_no_pos.split(",")]
                raw_tokens.extend(pieces)

            # Filter empties (stemming can be applied in a later step)
            cleaned_frames[frame] = [t for t in raw_tokens if t]

        return cleaned_frames

    def find_frames(
        self,
        df: pd.DataFrame,
        frames: List[str],
        all_frames: dict,
        text_col: str = "snippet_text",
    ) -> pd.DataFrame:
        """Add binary columns indicating presence of selected frames in snippet text.

        Args:
            df: Input dataframe containing a text column (default 'snippet_text').
            frames: List of frame names to search for (keys in all_frames).
            all_frames: Dictionary mapping frame name -> list of cleaned terms.
            text_col: Column name containing the snippet text.

        Returns:
            DataFrame copy with one binary column per frame in `frames`.
        """
        if df.empty:
            raise ValueError("Dataframe is empty")

        if text_col not in df.columns:
            raise ValueError(f"Missing required text column: {text_col}")

        # Work on a copy
        out = df.copy()

        # Always filter to immigration-related snippets
        if "relevance_label" in out.columns:
            out = out[out["relevance_label"] == "ABOUT_IMMIGRATION"]

        # Create a stemmed text column
        out = self.stem_text_column(
            out, text_col=text_col, output_col="stemmed_snippet_text"
        )

        # Build patterns per selected frame using stemmed terms and mark presence via str.contains
        for frame in frames:
            if frame not in all_frames:
                # If frame not present in dict, create a zero column
                out[frame] = 0
                continue

            # Stem the frame terms
            raw_terms = [t for t in all_frames.get(frame, []) if t]
            terms = self.stem_terms(raw_terms)
            if not terms:
                out[frame] = 0
                continue

            # Escape each term and add word boundaries; join with |
            escaped = [re.escape(t.lower()) for t in terms]
            # Use word boundary; allow simple substring if term has non-word chars
            parts = [rf"\b{e}\b" for e in escaped]
            pattern = "|".join(parts)

            out[frame] = (
                out["stemmed_snippet_text"]
                .astype(str)
                .str.contains(pattern, case=False, na=False, regex=True)
                .astype(int)
            )

        # Optional: drop the stemmed helper column to keep output tidy
        out = out.drop(columns=["stemmed_snippet_text"], errors="ignore")

        return out

    def frames_share_over_time(
        self,
        df: pd.DataFrame,
        frames: List[str],
        all_frames: dict,
        timeseries: str = "year",
        text_col: str = "snippet_text",
    ) -> pd.DataFrame:
        """Compute percentage share of snippets containing each frame over time.

        Steps:
        - Force stemming of text and terms via find_frames()
        - Create a total counter per snippet
        - Group by the requested timeseries (default 'year') and sum
        - Convert sums to percentages and return a long-form DataFrame

        Args:
            df: Input dataframe of snippets.
            frames: List of frame names to include.
            all_frames: Dictionary of all frame term lists.
            timeseries: Column name for time grouping; typically 'year'.
            text_col: Column containing snippet text.

        Returns:
            DataFrame with columns [timeseries, 'Frame', 'Percentage'].
        """
        if df.empty:
            return pd.DataFrame(columns=[timeseries, "Frame", "Percentage"])

        # Prepare time columns (adds 'year')
        df_time = df.copy()
        if timeseries == "year":
            # Ensure datetime and time buckets
            if "date" not in df_time.columns:
                raise ValueError("Missing required 'date' column for time grouping")
            df_time = SentimentDataProcessor()._prepare_time_columns(df_time)

        # Find frames with forced stemming and optional ABOUT_IMMIGRATION filter
        df_marked = self.find_frames(
            df=df_time,
            frames=frames,
            all_frames=all_frames,
            text_col=text_col,
        )

        # Total counter per snippet
        df_marked["snippet_id_total"] = 1

        # Group and sum
        group_cols = [timeseries]
        sum_cols = frames + ["snippet_id_total"]
        df_total = df_marked.groupby(group_cols, as_index=False)[sum_cols].sum()

        # Convert to percentage per frame
        for frame in frames:
            df_total[frame] = df_total[frame] / df_total["snippet_id_total"] * 100.0

        # Melt to long form
        df_long = df_total.melt(
            id_vars=[timeseries, "snippet_id_total"],
            value_vars=frames,
            var_name="Frame",
            value_name="Percentage",
        )

        return df_long[[timeseries, "Frame", "Percentage"]]
