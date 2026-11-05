import logging
import re
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
from immigration_rhetoric.common.utils import (
    extract_numbers_as_int,
    text_after_last_slash,
)
from tqdm import tqdm

# When output is redirected to a file/log (not an interactive terminal), tqdm can't overwrite its own
# progress line, so every refresh becomes a separate line. Disable the bar in that case.
TQDM_DISABLE = not sys.stderr.isatty()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class SpeechMetadataJoiner:
    """Joins raw speeches to speaker metadata (Wikidata MP data), matching on person ID, membership ID or name."""

    def __init__(self, raw_speeches_dir: str | Path, output_dir: str | Path):
        self.raw_speeches_dir = Path(raw_speeches_dir)
        self.output_dir = Path(output_dir)

    def load_speech_csv(self, min_year: int, max_year: int) -> pd.DataFrame:
        """Loads and combines the raw speech csvs for years between min_year and max_year (inclusive)."""
        files = [f.name for f in self.raw_speeches_dir.iterdir() if f.is_file()]

        filtered_files = [
            file
            for file in files
            if (
                file.endswith(".csv")
                and (extract_numbers_as_int(file) >= min_year)
                and (max_year >= extract_numbers_as_int(file))
            )
        ]

        dataframes = []
        for filename in tqdm(
            filtered_files, desc="Loading raw speech csv", disable=TQDM_DISABLE
        ):
            df = pd.read_csv(self.raw_speeches_dir / filename)
            dataframes.append(df)

        combined_df = pd.concat(dataframes, ignore_index=True)
        logging.info(
            f"Loaded {len(combined_df)} speeches from {len(dataframes)} files ({min_year}-{max_year})"
        )

        return combined_df

    @staticmethod
    def prepare_speeches_for_join(all_speeches: pd.DataFrame) -> pd.DataFrame:
        """Does some simple preprocessing to prepare the speeches for joining."""
        num_before = len(all_speeches)

        if "person_id" in all_speeches.columns:
            all_speeches["person_id"] = all_speeches["person_id"].replace(
                "unknown", None
            )

        if "speakerid" in all_speeches.columns:
            all_speeches.rename(columns={"speakerid": "membership_id"}, inplace=True)
            all_speeches["membership_id"] = all_speeches["membership_id"].replace(
                "unknown", None
            )

        # convert date to datetime object (needs to be done after loading)
        all_speeches["date"] = pd.to_datetime(all_speeches["date"])

        num_missing_text = all_speeches.text.isna().sum()
        all_speeches = all_speeches[all_speeches.text.notna()]

        num_duplicate_ids = all_speeches.duplicated(subset=["id"]).sum()
        all_speeches = all_speeches.drop_duplicates(subset=["id"])

        # drop speeches where the three identifiable fields are nan
        identifiable_fields = ["speakername", "membership_id", "person_id"]
        identifiable_fields_in_these_speeches = all_speeches.columns.intersection(
            identifiable_fields
        )  # don't look for columns that aren't in the speeches
        num_before_nan_drop = len(all_speeches)
        all_speeches = all_speeches.dropna(
            subset=identifiable_fields_in_these_speeches, how="all"
        )
        num_no_identifiable_fields = num_before_nan_drop - len(all_speeches)

        logging.info(
            f"Cleaned {num_before} speeches: dropped {num_missing_text} with missing text, "
            f"{num_duplicate_ids} duplicate ids, {num_no_identifiable_fields} with no identifiable fields "
            f"-> {len(all_speeches)} remaining"
        )

        return all_speeches

    @staticmethod
    def split_speeches_into_join_stages(
        speeches: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits the speeches up by which fields they can use to join on, the stages are defined as follows:

        Stage 1. person_id is not null. Here we can join directly to wikidata

        Stage 2. person_id is null but membership_id is not. Here we can use the memberships dataframe to get the
        person_id, then join as in stage 1

        Stage 3. person_id and speaker are null, here we require name matching to do the joins
        """
        total = speeches.shape[0]

        if "person_id" in speeches.columns:
            stage1 = speeches[speeches.person_id.notnull()]
        else:
            stage1 = pd.DataFrame(columns=speeches.columns)

        # filter out the stage 1 speeches
        remove_stage1 = speeches[~speeches.id.isin(stage1.id)]

        if "membership_id" in remove_stage1.columns:
            stage2 = remove_stage1[remove_stage1.membership_id.notnull()]
        else:
            stage2 = pd.DataFrame(columns=speeches.columns)

        # after removing stage 1 and 2, everything left is stage 3
        stage3 = remove_stage1[~remove_stage1.id.isin(stage2.id)]

        logging.info(
            f"Split {total} speeches into join stages: {stage1.shape[0]} by person_id "
            f"({stage1.shape[0] / total * 100:.1f}%), {stage2.shape[0]} by membership_id "
            f"({stage2.shape[0] / total * 100:.1f}%), {stage3.shape[0]} by name matching "
            f"({stage3.shape[0] / total * 100:.1f}%)"
        )

        return stage1, stage2, stage3

    @staticmethod
    def make_id_numeric(speeches: pd.DataFrame, id_col: str) -> pd.DataFrame:
        """
        Method to extract <number> from URLs of the form uk.org.publicwhip/<id_col>/<number>
        to the <id_col> column
        """
        original_cols = len(speeches.columns)

        # get text after the last slash
        speeches["id_numeric"] = speeches[id_col].apply(text_after_last_slash)

        speeches[id_col] = pd.to_numeric(speeches["id_numeric"]).astype(int)

        speeches_drop = speeches.drop(columns="id_numeric")

        if len(speeches_drop.columns) != original_cols:
            raise ValueError("Added an extra column when preparing stage 1 speeches")

        return speeches_drop

    def prepare_stage1(self, stage1: pd.DataFrame) -> pd.DataFrame:
        """Only preparation required for stage 1 is extracting the person_id number from the column."""
        try:
            stage1_prepared = self.make_id_numeric(stage1, "person_id")
            return stage1_prepared

        except KeyError:
            return stage1

    def prepare_stage2(self, stage2: pd.DataFrame) -> pd.DataFrame:
        """Prepare the stage 2 speeches for the join on person_id."""
        try:
            stage2_prepared = self.make_id_numeric(stage2, "membership_id")
            return stage2_prepared
        except KeyError:
            return stage2

    @staticmethod
    def clean_name(name: str) -> str:
        try:
            punctuations_with_titles = [".", ",", ":", ";", ""]
            remove_titles_from_name = ["Mrs", "Mr", "Miss", "Dr", "Hon"]

            # if the name starts with `The` and has brackets then grab name from within brackets
            name = name.strip()
            if (
                name.startswith("The")
                or name.lower()
                .replace("mr.", "mr")
                .replace("mr,", "mr")
                .replace("mr:", "mr")
                .replace("mr;", "mr")
                .startswith("mr speaker")
                or name.lower()
                .replace("mr.", "mr")
                .replace("mr,", "mr")
                .replace("mr:", "mr")
                .replace("mr;", "mr")
                .startswith("mr deputy speaker")
                or name.lower()
                .replace("mr.", "mr")
                .replace("mr,", "mr")
                .replace("mr:", "mr")
                .replace("mr;", "mr")
                .startswith("mr deputy-speaker")
                or name.lower()
                .replace("mr.", "mr")
                .replace("mr,", "mr")
                .replace("mr:", "mr")
                .replace("mr;", "mr")
                .startswith("mr deputy - speaker")
            ) and "(" in name:
                match = re.search(r"\(([^)]+)", name)
                if match:
                    name = match.group(1)

            if "(" in name:
                # if there is still `(` in name then get name till `(` e.g., "Mr. R. S.Hudson (Secretary,
                # Overseas Trade Department)" or "Mr. Patrick Cormack (Staffordshire)"
                name = name.split("(")[0].strip()

            # remove titles (with different punctuations) from name
            for title in remove_titles_from_name:
                for punct in punctuations_with_titles:
                    title_with_punct = title + punct
                    name = name.replace(title_with_punct, "")
                    name = name.replace(title_with_punct.lower(), "")
                    name = name.replace(title_with_punct.upper(), "")

            # replace multiple spaces any where with single space and remove leading and trailing spaces
            name = re.sub(r"\s+", " ", name).strip()
            return name.strip()

        except (AttributeError, TypeError) as e:
            logging.error(f"name: {name}")
            raise e from None

    def prepare_stage3(self, stage3: pd.DataFrame) -> pd.DataFrame:
        """Prepare the stage 3 speeches for name matching."""

        # lowercase names
        stage3["speakername"] = stage3["speakername"].str.lower()

        # filter out speeches made by 'members' or any variant thereof
        member_not_in_speakername = stage3[~stage3.speakername.str.contains("member")]

        # apply name cleaning
        member_not_in_speakername["clean_name"] = member_not_in_speakername[
            "speakername"
        ].apply(self.clean_name)

        logging.info(
            f"Dropped {stage3.shape[0] - member_not_in_speakername.shape[0]} speeches referring to "
            "'member' by name, cleaned remaining speakernames"
        )

        return member_not_in_speakername

    @staticmethod
    def join_speeches_on_person_id(
        speeches: pd.DataFrame, wikidata: pd.DataFrame
    ) -> pd.DataFrame:
        initial_num = len(speeches)

        # look for matches
        matches = speeches[speeches["person_id"].isin(wikidata["theyWorkForYouID"])]
        num_non_matches = initial_num - len(matches)

        # merge the dataframes, this is a many-to-many merge, so have lots of duplicates we need to filter
        # note we could preprocess the wikidata table more so that this is many-to-one
        merged = pd.merge(
            matches, wikidata, left_on="person_id", right_on="theyWorkForYouID"
        )

        # can drop matches where the wikidata date range doesn't fit with the date of the speech
        dropped_invalid_dates = merged[
            (merged.date >= merged.partyStartDate)
            & (merged.date <= merged.partyEndDate)
        ]

        # if the only difference is the altLabel, we can safely drop it
        columns_except_altLabel = dropped_invalid_dates.columns.difference(
            ["altLabel"]
        ).tolist()
        drop_different_altlabels = dropped_invalid_dates.drop_duplicates(
            subset=columns_except_altLabel
        )

        # these are caused when someone changes party and makes a speech on the same date
        extra_speeches = drop_different_altlabels[
            drop_different_altlabels.duplicated(subset=["id"], keep=False)
        ]
        # so we can keep the newer party
        dededuplicated_extras = extra_speeches[
            extra_speeches.date == extra_speeches.partyStartDate
        ]

        # initially remove all duplicated ids, then add the deduplicated ones back on
        drop_duplicate_ids = drop_different_altlabels[
            ~drop_different_altlabels.duplicated(subset=["id"], keep=False)
        ]

        add_deduplicated_ids = pd.concat([drop_duplicate_ids, dededuplicated_extras])

        # drop some columns we no longer need
        dropped_cols = add_deduplicated_ids.drop(
            columns=[
                "parliament_term",
                "parliament_term_id",
                "person",
                "altLabel",
                "partyStartDate",
                "partyEndDate",
                "mp_wikiID",
            ]
        )

        # figure out how many we lost due to deduplicating alone, so add on the ones we lost due to non-matches
        deduplicating_losses = len(drop_different_altlabels) - len(dropped_cols)
        logging.info(
            f"Joined {initial_num} speeches to wikidata by person_id -> {len(dropped_cols)} speeches "
            f"({num_non_matches} had no matching person_id, {deduplicating_losses} were duplicate matches)"
        )

        return dropped_cols

    def join_speeches_on_membership_id(
        self, speeches: pd.DataFrame, memberships: pd.DataFrame, wikidata: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Stage 2 of the speech join. Done in two stages, first membership_id is joined to get person_id.
        Then the speeches can be joined to the wikidata table as per join_speeches_on_person_id.
        """
        initial_num = len(speeches)

        matches = speeches[speeches.membership_id.isin(memberships.id)]
        logging.info(
            f"Matched {len(matches)}/{initial_num} speeches to a membership_id "
            f"({initial_num - len(matches)} had no matching membership)"
        )

        speeches_with_person_id = pd.merge(
            matches, memberships, left_on="membership_id", right_on="id"
        )

        # pandas only suffixes columns present in both frames. `speeches` may have no person_id column at
        # all, in which case memberships' person_id carries through unsuffixed and needs no cleanup here.
        if "person_id_x" in speeches_with_person_id.columns:
            speeches_with_person_id = speeches_with_person_id.drop(
                columns=["person_id_x"]
            ).rename(columns={"person_id_y": "person_id"})

        speeches_with_person_id = speeches_with_person_id.drop(columns=["id_y"]).rename(
            columns={"id_x": "id"}
        )

        # can now join person_id to wikidata
        speeches_with_metadata = self.join_speeches_on_person_id(
            speeches_with_person_id, wikidata
        )

        return speeches_with_metadata

    def join_speeches_on_name(
        self,
        speeches: pd.DataFrame,
        wikidata: pd.DataFrame,
        speech_name_col: str,
        wikidata_name_col: str,
    ) -> pd.DataFrame:
        """Function to join speeches on a name column to the wikidata table."""

        # lowercase names for consistent matching
        wikidata[wikidata_name_col] = wikidata[wikidata_name_col].str.lower()

        # filter to just matching speeches
        matches = speeches[speeches[speech_name_col].isin(wikidata[wikidata_name_col])]

        # do the merge
        merged = pd.merge(
            matches, wikidata, left_on=speech_name_col, right_on=wikidata_name_col
        )

        # remove invalid dates
        drop_invalid_dates = merged[
            (merged.date >= merged.partyStartDate)
            & (merged.date <= merged.partyEndDate)
        ]

        # remove dupes that differ only by the altLabel
        columns_except_altLabel = drop_invalid_dates.columns.difference(
            ["altLabel"]
        ).tolist()

        drop_different_altlabels = drop_invalid_dates.drop_duplicates(
            subset=columns_except_altLabel
        )

        # remove all speeches that have a duplicate, we will add the deduplicated ones back on later
        drop_duplicate_ids = drop_different_altlabels[
            ~drop_different_altlabels.duplicated(subset=["id"], keep=False)
        ]

        duplicated_ids = drop_different_altlabels[
            drop_different_altlabels.duplicated(subset=["id"], keep=False)
        ]

        # keep the entry with the later start date
        deduplicated = duplicated_ids.sort_values(
            by="partyStartDate", ascending=False
        ).drop_duplicates(subset=["id"], keep="first")

        # add the dedupe'd ones back on
        recombine = pd.concat([drop_duplicate_ids, deduplicated])
        logging.info(
            f"Matched {len(matches)} speeches by {speech_name_col} -> {wikidata_name_col}, "
            f"{len(recombine)} remaining after dropping invalid dates and duplicate matches"
        )

        return recombine

    def name_joining(
        self, speeches: pd.DataFrame, wikidata: pd.DataFrame
    ) -> pd.DataFrame:
        """Match clean speakernames to the name fields in the wikidata table."""

        joined_to_personLabel = self.join_speeches_on_name(
            speeches,
            wikidata,
            speech_name_col="clean_name",
            wikidata_name_col="personLabel",
        )

        # remove ones we've joined to avoid duplicates
        rest_of_speeches = speeches[~speeches.id.isin(joined_to_personLabel.id)]

        joined_to_altLabel = self.join_speeches_on_name(
            rest_of_speeches,
            wikidata,
            speech_name_col="clean_name",
            wikidata_name_col="altLabel",
        )

        stage3_joined = pd.concat([joined_to_personLabel, joined_to_altLabel])

        # don't need these columns anymore
        dropped_cols = stage3_joined.drop(
            columns=[
                "parliament_term",
                "parliament_term_id",
                "person",
                "altLabel",
                "partyStartDate",
                "partyEndDate",
                "mp_wikiID",
            ]
        )
        logging.info(f"Matched {len(dropped_cols)}/{len(speeches)} speeches by name")

        return dropped_cols

    @staticmethod
    def concat_joined_speeches(
        stage1: pd.DataFrame, stage2: pd.DataFrame, stage3: pd.DataFrame
    ) -> pd.DataFrame:
        combined = pd.concat([stage1, stage2, stage3])
        logging.info(f"Combined {len(combined)} joined speeches across all 3 stages")
        return combined

    def save_joined_speeches(self, joined_speeches: pd.DataFrame) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        years = joined_speeches.year.unique()
        for year in tqdm(years, desc="Saving joined speeches", disable=TQDM_DISABLE):
            this_year = joined_speeches[joined_speeches.year == year]
            file_path = self.output_dir / f"speeches_{year}.csv"
            this_year.to_csv(file_path, index=False)
        logging.info(
            f"Saved {len(joined_speeches)} speeches to {len(years)} files in {self.output_dir}"
        )
