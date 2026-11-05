import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple

import pandas as pd
from immigration_rhetoric.common.utils import text_after_last_slash

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class SpeakerMetadataPreprocessor:
    """Prepares raw speaker metadata (TheyWorkForYou people data and Wikidata MP data) for joining to speeches."""

    def __init__(self, input_dir: str | Path):
        self.input_dir = Path(input_dir)

    def load_people_json(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads the content of people.json into three dataframes: people, memberships, posts."""
        filepath = self.input_dir / "people.json"
        logging.info(f"Reading from {filepath}")
        with open(filepath, "r") as json_data:
            people_data = json.load(json_data)

        data_df = pd.json_normalize(people_data)

        people = pd.DataFrame(data_df.at[0, "persons"])
        memberships = pd.DataFrame(data_df.at[0, "memberships"])
        posts = pd.DataFrame(data_df.at[0, "posts"])

        return (people, memberships, posts)

    @staticmethod
    def fill_redirected_ids(df: pd.DataFrame) -> pd.DataFrame:
        """
        Some rows in people and memberships have NaN for all rows except redirect.
        This happens when the id has been redirected to another id.
        Join the data from the redirected id onto these rows.

        Returns: A DataFrame with no rows that have NaNs for other_names
        """
        # filter to rows that have been redirected
        redirected = df[df["redirect"].notna()]

        # drop unnecessary cols and rename
        redirected = redirected[["id", "redirect"]]
        redirected = redirected.rename(columns={"id": "original_id"})

        merged = pd.merge(redirected, df, left_on="redirect", right_on="id", how="left")

        # drop the redirected ids we just joined on and relabel
        merged = merged.drop(columns=["redirect_x", "id"])
        # this results in rows where id is the original id, but the other columns have data from the redirected id
        redirect_rows_with_info = merged.rename(columns={"original_id": "id"})

        # filter the df to remove the redirected rows
        no_redirects = df[df["redirect"].isna()]

        # join the ones with proper info back on
        return pd.concat([no_redirects, redirect_rows_with_info])

    def load_wikidata(self) -> pd.DataFrame:
        """Loads metadata from wikidata into a pandas DataFrame."""
        return pd.read_csv(self.input_dir / "mps_wikidata.csv")

    def preprocess_wikidata(self, wikidata: pd.DataFrame) -> pd.DataFrame:
        """Prepares the wikidata dataset for joining to speeches."""
        num_rows = len(wikidata)
        num_mps = wikidata.theyWorkForYouID.nunique()

        # replace NaN party end dates with today's date
        num_missing_end_date = wikidata["partyEndDate"].isna().sum()
        wikidata["partyEndDate"] = wikidata["partyEndDate"].fillna(datetime.today())

        # convert to datetimes
        wikidata["partyStartDate"] = pd.to_datetime(
            wikidata["partyStartDate"], format="%Y-%m-%dT%H:%M:%SZ"
        )
        wikidata["partyEndDate"] = pd.to_datetime(
            wikidata["partyEndDate"], format="%Y-%m-%dT%H:%M:%SZ"
        )

        # drop NaN ids
        num_missing_id = wikidata.theyWorkForYouID.isna().sum()
        wikidata = wikidata[wikidata.theyWorkForYouID.notna()]

        # convert ids to int
        wikidata["theyWorkForYouID"] = pd.to_numeric(
            wikidata["theyWorkForYouID"]
        ).astype(int)

        logging.info(
            f"Preprocessed wikidata: {num_rows} rows, {num_mps} unique MPs "
            f"({num_missing_end_date} missing partyEndDate filled with today's date, "
            f"{num_missing_id} dropped for missing theyWorkForYouID)"
        )

        return wikidata

    def preprocess_memberships(self, memberships: pd.DataFrame) -> pd.DataFrame:
        """Prepare the memberships dataframe for joining membership id to person id."""
        memberships_filled = self.fill_redirected_ids(memberships)

        # make the id fields numeric
        memberships_filled["id"] = (
            memberships_filled["id"].apply(text_after_last_slash).astype(int)
        )

        # extract the numeric value for person_id
        memberships_filled["person_id"] = (
            memberships_filled["person_id"].apply(text_after_last_slash).astype(int)
        )

        logging.info(
            f"Preprocessed memberships: {memberships_filled.shape[0]} rows, "
            f"{memberships_filled.id.nunique()} unique ids"
        )

        # only need to return the two id columns
        return memberships_filled[["id", "person_id"]]
