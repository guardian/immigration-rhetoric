import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
from immigration_rhetoric.stage08_inference.classify_snippets import (
    NOT_RELEVANT_LABEL,
    TQDM_DISABLE,
)
from tqdm import tqdm

COLUMN_NAMES = {
    "id": "speech_id",
    "text": "speech_text",
    "constituencyLabel": "constituency_label",
    "partyLabel": "party_label",
    "personLabel": "person_label",
    "snippet": "snippet_text",
}

SPEECH_COLUMNS = [
    "speech_id",
    "speech_text",
    "constituency_label",
    "party_label",
    "person_label",
    "date",
]

LABEL_COUNT_COLUMNS = {
    "NEGATIVE_TOWARDS_IMMIGRATION": "n_snippets_negative",
    "NEUTRAL_TOWARDS_IMMIGRATION": "n_snippets_neutral",
    "NOT_ABOUT_IMMIGRATION": "n_snippets_not_about",
    "POSITIVE_TOWARDS_IMMIGRATION": "n_snippets_positive",
}

SNIPPET_COLUMNS = [
    "snippet_id",
    "snippet_text",
    "speech_id",
    "date",
    "party_label",
    "person_label",
    "relevance_label",
    "relevance_label_scores",
    "sentiment_label",
    "sentiment_label_scores",
    "label",
]


class InferenceOutputCombiner:
    """Combines the yearly labelled snippets into one snippet-level and one speech-level dataset."""

    @staticmethod
    def load_labelled_snippets(file_path: Path) -> pd.DataFrame:
        """Loads one year's labelled snippets with renamed columns, dropping any without a known speaker."""
        df = pd.read_json(file_path, lines=True, convert_dates=False)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df.rename(columns=COLUMN_NAMES)

        no_speaker = df["person_label"].isna()
        if no_speaker.any():
            logging.info(
                f"Dropping {no_speaker.sum()} snippets without a speaker from {file_path.name}"
            )
        return df[~no_speaker].reset_index(drop=True)

    @staticmethod
    def sentiment_score(labels: pd.Series) -> float:
        """(positive - negative) / number of snippets about immigration, or 0 if there are none."""
        n_about = (labels != NOT_RELEVANT_LABEL).sum()
        if n_about == 0:
            return 0
        n_positive = (labels == "POSITIVE_TOWARDS_IMMIGRATION").sum()
        n_negative = (labels == "NEGATIVE_TOWARDS_IMMIGRATION").sum()
        return (n_positive - n_negative) / n_about

    @classmethod
    def summarise_speeches(cls, df: pd.DataFrame) -> pd.DataFrame:
        """One row per speech, with its snippet ids, sentiment score and the number of snippets per label."""
        label_counts = (
            df.groupby(["speech_id", "label"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=list(LABEL_COUNT_COLUMNS), fill_value=0)
            .rename(columns=LABEL_COUNT_COLUMNS)
            .reset_index()
        )
        speeches = (
            df.groupby(SPEECH_COLUMNS, dropna=False)
            .agg(
                snippet_ids=("snippet_id", list),
                sentiment_score_positive_minus_negative=(
                    "label",
                    cls.sentiment_score,
                ),
            )
            .reset_index()
            .merge(label_counts, on="speech_id", how="left")
        )
        speeches["n_snippets_total"] = speeches["snippet_ids"].apply(len)
        speeches["n_snippets_about"] = (
            speeches["n_snippets_total"] - speeches["n_snippets_not_about"]
        )
        return speeches

    def combine(self, input_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Returns (speeches, snippets) combined across every labelled_snippets_{year}.jsonl in input_dir."""
        all_speeches, all_snippets = [], []
        for file_path in tqdm(
            sorted(input_dir.glob("labelled_snippets_*.jsonl")),
            desc="Combining labelled snippets",
            disable=TQDM_DISABLE,
        ):
            df = self.load_labelled_snippets(file_path)
            all_speeches.append(self.summarise_speeches(df))
            all_snippets.append(df[SNIPPET_COLUMNS])
            logging.info(f"Loaded {len(df)} snippets from {file_path.name}")

        speeches = pd.concat(all_speeches, ignore_index=True)
        snippets = pd.concat(all_snippets, ignore_index=True)
        return speeches, snippets
