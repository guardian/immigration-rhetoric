"""Data processing classes for sentiment analysis."""

import hashlib
import json
import logging
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from immigration_rhetoric.stage09_analysis.party_utils import PartyUtils
from immigration_rhetoric.stage09_analysis.plot_config import DataConfig
from immigration_rhetoric.stage09_analysis.sentiment_calculator import (
    SentimentCalculator,
)

logger = logging.getLogger(__name__)


class SentimentDataProcessor:
    """Processes sentiment data over time periods.

    If `cache_dir` is set, the term presence matrix used by the sensitivity analysis is cached there.
    """

    def __init__(
        self,
        calculator: Optional[SentimentCalculator] = None,
        data_config: Optional[DataConfig] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.calculator = calculator or SentimentCalculator()
        self.data_config = data_config or DataConfig()
        self.cache_dir = cache_dir

    def _prepare_dataframe(
        self, df: pd.DataFrame, speeches: bool = False
    ) -> pd.DataFrame:
        """Prepare input dataframe for further calculations. Preparation
        determined by whether we are processing snippets for sentiment
        analysis (`speeches = False`) or speeches for prevalence analysis (`speeches = True`)
        """
        if df.empty:
            raise ValueError("Dataframe is empty")

        required_cols = (
            ["date", "party_label"] if speeches else ["date", "label", "party_label"]
        )

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        df = self._prepare_party_columns(df)

        df = self._prepare_time_columns(df)

        return df

    def _prepare_party_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to keep party labels for speeches by parties of interest
        and apply any name mappings. E.g. 'Green Party of England and Wales. --> 'Green Party'
        """

        party_utils = PartyUtils()

        df = party_utils.filter_dataframe_by_party_windows(
            df=df,
            date_column="date",
            party_column="party_label",
            party_windows=DataConfig.get_party_windows(),
        )

        df["party_label"] = df["party_label"].apply(
            party_utils.apply_party_name_mapping
        )

        return df

    def _prepare_time_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a year column to dataframe."""
        df = df.copy()
        if df["date"].dtype != "datetime64[ns]":
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.to_period("Y").dt.to_timestamp()
        return df

    def _get_sentiment_by_group(
        self, df: pd.DataFrame, groupby: str, include_uncorrected: bool = False
    ) -> pd.DataFrame:
        """Calculate sentiment over time grouped by specified dimension, e.g.
        time periods or triggers. This applies count corrections when grouping
        by time period, however this assumes that the misclassification rates are
        uniform over the dimension. We do not believe this assumptions to hold
        for trigger terms because (1) language changes over time, and (2) it
        is likely that some triggers are associated with more extreme
        sentiment and the more extreme the sentiment, the less likely it is to
        be misclassified.
        """
        corrected = groupby != "trigger" and not include_uncorrected

        sentiment_df = pd.DataFrame(
            [
                {
                    groupby: name,
                    "sentiment": self.calculator.percent_pos_minus_neg(
                        group["label"], corrected=corrected
                    ),
                }
                for name, group in df.groupby(groupby)
            ],
            columns=[groupby, "sentiment"],
        )

        # Groups with no snippets about immigration take the previous group's sentiment
        sentiment_df["sentiment"] = sentiment_df["sentiment"].ffill()

        return sentiment_df.dropna().reset_index(drop=True)

    def get_party_sentiment_over_time(
        self,
        df: pd.DataFrame,
        timeseries: str = "year",
        parties_of_interest: Optional[List[str]] = None,
        include_uncorrected: bool = False,
    ) -> dict:
        """Get sentiment data overall and optionally by party.

        Args:
            df: Input dataframe
            timeseries: Column to use as timeseries, e.g. 'year'
            parties_of_interest: List of parties to include

        Returns:
            Dictionary with party names as keys and sentiment dataframes as values
        """

        parties = {}

        # Map party names to standard forms and add a year column
        df = self._prepare_dataframe(df)

        # Overall sentiment
        overall = self._get_sentiment_by_group(
            df=df,
            groupby=timeseries,
            include_uncorrected=include_uncorrected,
        )
        if not overall.empty:
            parties["Overall"] = overall

        # Party-specific sentiment
        if parties_of_interest:
            for party in parties_of_interest:
                party_data = df[df.party_label == party]
                if not party_data.empty:
                    party_sentiment = self._get_sentiment_by_group(
                        df=party_data,
                        groupby=timeseries,
                        include_uncorrected=include_uncorrected,
                    )
                    if not party_sentiment.empty:
                        parties[party] = party_sentiment
                    else:
                        logger.warning(f"No data available for party: {party}")

        return parties

    @staticmethod
    def _find_matching_words(text, word_list, return_unique=False):
        """Counter for trigger words within a snippet, optimised for performance"""
        if not isinstance(text, str):
            return [] if not return_unique else set()

        # Compile all patterns at once
        text_lower = text.lower()
        matching_words = []

        # Use a set for faster lookups
        text_words = set(re.findall(r"\b\w+\b", text_lower))

        # Check which words from our list appear in the text
        for word in word_list:
            word_lower = word.lower()
            if word_lower in text_words or word_lower in text_lower:
                # Double-check with proper word boundary
                pattern = re.compile(r"\b" + re.escape(word_lower) + r"\b")
                matches = pattern.findall(text_lower)
                matching_words.extend(matches)

        if return_unique:
            return set(matching_words)
        else:
            return matching_words

    def sentiment_per_trigger(
        self,
        df: pd.DataFrame,
        trigger_terms: List,
        text_col: str = "snippet_text",
        min: int = 50,
    ) -> pd.DataFrame:
        """Calculate sentiment per trigger/topic."""
        if text_col not in df.columns:
            logger.warning(f"No '{text_col}' column found in dataframe")
            return pd.DataFrame()

        df_triggers = df.copy()

        # Record which trigger terms appear in each snippet (if any)
        df_triggers["trigger"] = df_triggers[text_col].apply(
            lambda x: self._find_matching_words(
                text=x, word_list=trigger_terms, return_unique=True
            )
        )

        # One row per term
        df_triggers = df_triggers.explode(column="trigger")

        # Filter the dataframe based on the minimum count of speeches the trigger appeared in
        # to ensure low frequency qualifiers dont skew the dataset
        value_counts = df_triggers["trigger"].value_counts()
        filtered_df = df_triggers[
            df_triggers["trigger"].isin(value_counts[value_counts >= min].index)
        ]

        # Get sentiment grouped by qualifier
        qualifier_sentiment = self._get_sentiment_by_group(
            df=filtered_df, groupby="trigger"
        )

        return qualifier_sentiment.sort_values("sentiment")

    def _get_prevalence_by_group(
        self,
        df_trigger_speeches: pd.DataFrame,
        df_all_speeches: pd.DataFrame,
        groupby: str,
    ) -> pd.DataFrame:
        """Calculate prevalence over time grouped by specified dimension, e.g. time periods or triggers."""

        # Get counts of speeches about immigration
        about_counts = (
            df_trigger_speeches[df_trigger_speeches.n_snippets_about > 0]
            .groupby(groupby)
            .size()
            .reset_index(name="about_count")
        )

        # Get counts of all speeches
        total_counts = (
            df_all_speeches.groupby(groupby).size().reset_index(name="total_count")
        )

        # Join the dataframes
        return pd.merge(total_counts, about_counts, on=groupby, how="left").fillna(0)

    def get_prevalence_over_time(
        self,
        df_trigger_speeches: pd.DataFrame,
        df_all_speeches: pd.DataFrame,
        timeseries: str = "year",
        parties_of_interest: Optional[List[str]] = None,
    ) -> dict:
        """Get prevalence data overall and optionally by party.

        Args:
            df: Input dataframe with snippets containing trigger terms
            df_all_speeches: Input dataframe with all speeches
            timeseries: Column to use as timeseries, e.g. 'year'
            parties_of_interest: List of parties to include

        Returns:
            Dictionary with party names as keys and prevalence dataframes as values
        """

        parties = {}

        # Map party names to standard forms and add a year column
        df_trigger_speeches = self._prepare_dataframe(
            df_trigger_speeches, speeches=True
        )
        df_all_speeches = self._prepare_dataframe(df_all_speeches, speeches=True)

        # Get the full date range from all data
        full_date_range = sorted(
            set(df_trigger_speeches[timeseries].unique())
            | set(df_all_speeches[timeseries].unique())
        )

        # Create a template DataFrame with all dates and zero values
        template_df = pd.DataFrame(
            {
                timeseries: full_date_range,
            }
        )

        # Overall prevalence
        overall = self._get_prevalence_by_group(
            df_trigger_speeches=df_trigger_speeches,
            df_all_speeches=df_all_speeches,
            groupby=timeseries,
        )
        if not overall.empty:
            parties["Overall"] = overall

        # Party-specific prevalence
        if parties_of_interest:
            for party in parties_of_interest:
                party_data = df_trigger_speeches[
                    df_trigger_speeches.party_label == party
                ]
                party_data_all = df_all_speeches[df_all_speeches.party_label == party]

                party_prevalence = self._get_prevalence_by_group(
                    df_trigger_speeches=party_data,
                    df_all_speeches=party_data_all,
                    groupby=timeseries,
                )

                if not party_prevalence.empty:
                    # Merge with template to fill missing dates with zeros
                    party_filled = template_df.merge(
                        party_prevalence, on=timeseries, how="left"
                    )
                    # Fill NaN values with zeros from template
                    party_filled = party_filled.fillna(0)
                    parties[party] = party_filled
                else:
                    logger.warning(
                        f"No data available for party: {party}, using zeros for all dates"
                    )

        return parties

    @staticmethod
    def _stable_terms_hash(terms):
        """Generate a stable hash for a list of terms, independent of order."""
        sorted_terms = sorted(terms)
        serialized = json.dumps(sorted_terms).encode("utf-8")
        return hashlib.md5(serialized).hexdigest()

    def _load_or_create_term_presence_matrix(
        self, df, all_terms, text_col="snippet_text"
    ):
        """Create a binary matrix showing which snippets contain which terms."""

        # Path for a cache file that's uniquely named for a particular set of terms and snippets
        cache_file = None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            snippets_hash = hashlib.md5(
                "\n".join(df["snippet_id"].astype(str)).encode("utf-8")
            ).hexdigest()
            cache_file = (
                self.cache_dir
                / f"presence_matrix_{self._stable_terms_hash(all_terms)}_{snippets_hash}.pkl"
            )

        # Try to load from cache...
        if cache_file and cache_file.exists():
            logger.info("Loading term presence matrix from cache: " + str(cache_file))
            with open(cache_file, "rb") as f:
                presence_matrix = pickle.load(f)
            return presence_matrix

        # ...otherwise, construct and cache
        logger.info("Creating term presence matrix for this set of terms.")
        presence_matrix = pd.DataFrame(0, index=df.index, columns=all_terms)
        for term in all_terms:
            pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
            presence_matrix[term] = df[text_col].apply(
                lambda x: 1 if isinstance(x, str) and pattern.search(x) else 0
            )
        presence_matrix["speech_id"] = df["speech_id"]
        presence_matrix["snippet_id"] = df["snippet_id"]

        if cache_file:
            with open(cache_file, "wb") as f:
                pickle.dump(presence_matrix, f)

        return presence_matrix

    def calculate_term_sensitivity(
        self,
        df: pd.DataFrame,
        all_terms: list,
        drop_percentage: int = 10,
        text_col: str = "snippet_text",
    ):
        """
        Run sensitivity analysis by randomly dropping terms and calculating
        yearly sentiment.

        Returns: dict of the snippets kept in the baseline run (all of them) and each sensitivity run
        """

        results: Dict[str, pd.DataFrame] = {}

        df = df.copy()
        logger.info(
            f"Baseline sentiment calculated with all {len(all_terms)} terms, "
            f"{len(df)} snippets and {df['speech_id'].nunique()} speeches."
        )
        results["baseline"] = df.copy()

        # Create term presence matrix for efficiently dropped terms in sensitivity analysis
        term_matrix = self._load_or_create_term_presence_matrix(df, all_terms, text_col)

        # Divide the trigger terms into m sets, each of which drops a `drop_percentage` of terms
        # (if the list doesn't divide evenly, the last set has fewer terms)
        n_drop = max(1, int(len(all_terms) * drop_percentage / 100))
        m = int(np.ceil(len(all_terms) / n_drop))
        np.random.seed(42)
        shuffled_terms = np.random.permutation(all_terms)
        term_groups = np.array_split(shuffled_terms, m)
        term_groups = [group.tolist() for group in term_groups]

        # Iterate through each set of dropped terms
        for i in range(m):
            run_name = f"run_{i + 1}"
            logger.info(
                f"Starting sensitivity analysis: {run_name}, dropping the following terms: {term_groups[i]}"
            )

            # Identify dropped and remaining terms
            dropped_terms = term_groups[i]
            remaining_terms = list(set(all_terms) - set(dropped_terms))

            # For each speech, check if any snippet contains any remaining term
            speech_has_remaining = (
                term_matrix.groupby("speech_id")[remaining_terms].sum().sum(axis=1) > 0
            )

            # Keep any speeches that have at least one snippet with a remaining term
            speech_ids_to_keep = speech_has_remaining[speech_has_remaining].index
            filtered_df = df[df["speech_id"].isin(speech_ids_to_keep)]
            logger.info(
                f"After dropping terms, {len(filtered_df)} snippets and {filtered_df['speech_id'].nunique()} speeches remain for analysis."
            )

            results[run_name] = filtered_df.copy()

        return results
