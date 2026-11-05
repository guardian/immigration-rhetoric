import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
from immigration_rhetoric.common.utils import extract_date
from pandas import DataFrame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class SpeechesProcessor:
    """Processes raw XML files containing records of speeches made in parliamentary debates into structured data.

    Args:
        xml_path: Path to directory containing XML files
        output_path: Path to directory for output files
        start_date: Start date for filtering files
        end_date: End date for filtering files
    """

    def __init__(
        self,
        xml_path: str | Path,
        output_path: str | Path,
        start_date: datetime,
        end_date: datetime,
    ):
        self.output_path = Path(output_path)
        self.xml_path = Path(xml_path)

        self.start_date = start_date
        self.end_date = end_date

        # Configuration
        self.min_word_count = 10
        self.speaker_id_fields = [
            "person_id",
            "speakerid",
            "speakername",
            "membership_id",
        ]
        self.columns_to_drop = [
            "redirect",
            "url",
            "type",
            "oral-qnum",
            "error",
            "time",
            "speakeroffice",
            "colnum",
            "hansard_membership_id",
            "hansard_id",
            "nospeaker",
        ]

    def get_speeches_from_xml(self) -> DataFrame:
        """Process debates XML files into structured CSV data of individual speeches."""
        # Load and filter XML files
        raw_xml_df = self._load_and_filter_xml_files()

        years = sorted(raw_xml_df["file_name"].apply(extract_date).dt.year.unique())
        logging.info(f"Processing {len(years)} year(s): {years[0]}-{years[-1]}")

        speeches = []
        for i, year in enumerate(years, start=1):
            speeches_df = self._extract_and_clean_speeches(raw_xml_df, year=year)
            self._save_speeches(speeches=speeches_df, year=year)
            speeches.append(speeches_df)
            logging.info(f"[{i}/{len(years)}] {year}: kept {len(speeches_df)} speeches")

        all_speeches = pd.concat(speeches, ignore_index=True)
        logging.info(
            f"Finished processing {len(years)} year(s), {len(all_speeches)} speeches total"
        )
        return all_speeches

    def _load_and_filter_xml_files(self) -> DataFrame:
        """Load and filter XML files within the specified date range."""
        xml_files = list(self.xml_path.glob("*.xml"))

        processed_files = []
        for file_path in xml_files:
            # Skip files that don't have a date or are outside the date range
            file_date = extract_date(file_path.name)
            if not file_date or not (self.start_date <= file_date <= self.end_date):
                continue

            try:
                xml_content = self._read_xml_file(file_path)
                processed_files.append(
                    {"file_name": file_path.name, "xml_text": xml_content}
                )
            except Exception as e:
                logging.error(f"Error processing {file_path.name}: {e}")

        logging.info(
            f"Loaded {len(processed_files)}/{len(xml_files)} XML files within "
            f"{self.start_date.date()} to {self.end_date.date()}"
        )
        return pd.DataFrame(processed_files)

    @staticmethod
    def _read_xml_file(file_path: Path) -> str:
        """Read XML file with appropriate encoding handling."""
        try:
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return file_path.read_text(encoding="latin-1")

    def _extract_and_clean_speeches(
        self, raw_xml_df: DataFrame, year: int | None = None
    ) -> DataFrame:
        """Process a subset of speeches into a cleaned DataFrame.

        Args:
            raw_xml_df: DataFrame containing raw XML data
            year: Optional year for filtering data in year-based processing
        """
        if year:
            raw_xml_df = raw_xml_df[
                raw_xml_df["file_name"].apply(extract_date).dt.year == year
            ]

        speeches_df = self._extract_all_speeches(raw_xml_df)
        initial_count = len(speeches_df)
        speeches_df = self._remove_invalid_speeches(speeches_df)
        speeches_df = self._remove_short_speeches(speeches_df)
        speeches_df = self._remove_unused_columns(speeches_df)
        speeches_df = self._add_date_columns(speeches_df, "id")
        speeches_df = self._remove_duplicate_speeches(speeches_df)
        speeches_df = self._remove_unidentified_speeches(speeches_df)
        speeches_df = self._replace_unknown_identifiers(speeches_df)
        logging.debug(
            f"Year {year}: {initial_count} raw speeches -> {len(speeches_df)} after cleaning"
        )
        return speeches_df

    def _extract_all_speeches(self, raw_xml_df: DataFrame) -> DataFrame:
        """Extract structured speech data from XML content."""
        speech_fields = self._get_speech_fields(raw_xml_df.xml_text.tolist())
        logging.debug(f"speech_fields: {speech_fields}")

        # Group files by date and find the most recent file for each date
        raw_xml_df = raw_xml_df.copy()
        raw_xml_df.loc[:, "date_str"] = raw_xml_df["file_name"].apply(
            lambda x: extract_date(x).strftime("%Y-%m-%d")
        )

        # Get the latest revised file for each date
        # e.g. if we have 3 files: DATE, DATEa, DATEb, we should extract speeches from only DATEb
        latest_files = raw_xml_df.sort_values(
            "file_name", ascending=True
        ).drop_duplicates("date_str", keep="last")
        logging.debug(
            f"Processing {len(latest_files)} files (latest per date) out of {len(raw_xml_df)} total files"
        )

        speeches = []
        for _, row in latest_files.iterrows():
            file_speeches = self._extract_speeches_from_single_file(
                row["xml_text"], speech_fields
            )
            speeches.extend(file_speeches)

        return pd.DataFrame(speeches)

    @staticmethod
    def _get_speech_fields(xml_texts: List[str]) -> Set[str]:
        """Extract all possible speech attribute fields from XML files."""
        fields = set()
        for xml_text in xml_texts:
            root = ET.fromstring(xml_text)
            for speech in root.findall(".//speech"):
                fields.update(speech.attrib.keys())
        return fields

    @staticmethod
    def _extract_speeches_from_single_file(
        xml_text: str, field_names: Set[str]
    ) -> List[Dict]:
        """Extract all speeches from a single debate XML file."""
        root = ET.fromstring(xml_text)
        speeches = []

        for speech_elem in root.findall(".//speech"):
            speech_data = {field: speech_elem.get(field) for field in field_names}
            speech_data["text"] = " ".join(
                "".join(p.itertext()) for p in speech_elem.findall(".//p")
            )
            speeches.append(speech_data)

        return speeches

    def _remove_unidentified_speeches(self, df: DataFrame) -> DataFrame:
        """Remove speeches that lack speaker identification."""
        present_id_fields = df.columns.intersection(self.speaker_id_fields)
        filtered_df = df.dropna(how="all", subset=present_id_fields)

        logging.debug(
            f"Kept {len(filtered_df)} speeches with speaker identification "
            f"(removed {len(df) - len(filtered_df)})"
        )
        return filtered_df

    def _remove_invalid_speeches(self, df: DataFrame) -> DataFrame:
        """Remove speeches that should be excluded."""
        if "speakername" in df.columns:
            df = df[df["speakername"] != "Several hon. Members"]
        if "redirect" in df.columns:
            df = df[df["redirect"].isnull()]
        # Drop invalid rows
        logging.debug(
            f"Found {df.text.isna().sum()} speeches where the text is NaN, dropping"
        )
        df = df[df.text.notna()]
        return df

    def _remove_short_speeches(self, df: DataFrame) -> DataFrame:
        """Remove speeches with fewer than min_word_count words."""
        df["word_count"] = df["text"].apply(lambda x: len(x.split()))
        filtered_df = df[df["word_count"] >= self.min_word_count].drop(
            columns=["word_count"]
        )

        logging.debug(
            f"Kept {len(filtered_df)} speeches with at least {self.min_word_count} words "
            f"(removed {len(df) - len(filtered_df)})"
        )
        return filtered_df

    def _remove_unused_columns(self, df: DataFrame) -> DataFrame:
        """Remove unnecessary columns from the DataFrame."""
        df_with_dropped_cols = df.drop(columns=self.columns_to_drop, errors="ignore")
        return df_with_dropped_cols

    def _remove_duplicate_speeches(self, df: DataFrame) -> DataFrame:
        """Remove duplicate speeches based on the 'id' column."""
        logging.debug(
            f"Found {len(df[df.duplicated(subset=['id'])])} duplicate speech IDs, dropping"
        )
        df = df.drop_duplicates(subset=["id"])
        return df

    @staticmethod
    def _add_date_columns(df: DataFrame, date_column: str) -> DataFrame:
        """Add year, month, and day columns based on the date string."""
        df = df.copy()
        # pd.to_datetime keeps a proper datetime dtype even when df has zero rows
        df["date"] = pd.to_datetime(df[date_column].apply(extract_date))
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        return df

    def _replace_unknown_identifiers(self, df: DataFrame) -> DataFrame:
        """Replacing 'unknown' values with None"""
        df = df.copy()

        # Handle person_id field
        if "person_id" in df.columns:
            logging.debug(
                f"Found {len(df[df['person_id'] == 'unknown'])} where person_id is 'unknown', replacing with None"
            )
            df["person_id"] = df["person_id"].replace("unknown", None)

        # Handle speakerid/membership_id field
        if "speakerid" in df.columns:
            df.rename(columns={"speakerid": "membership_id"}, inplace=True)
            logging.debug(
                f"Found {len(df[df['membership_id'] == 'unknown'])} where membership_id is 'unknown', replacing with None"
            )
            df["membership_id"] = df["membership_id"].replace("unknown", None)

        return df

    def _save_speeches(self, speeches: DataFrame, year: int) -> None:
        """Save processed speeches to CSV.

        Args:
            speeches: DataFrame containing the speeches to save
            year: Optional specific year for per-year processing
        """

        # Ensure output directory exists
        self.output_path.mkdir(parents=True, exist_ok=True)

        filename = f"raw_speeches_{year}.csv"

        output_location = self.output_path / filename
        speeches.to_csv(output_location, index=False)
        logging.debug(f"Saved {len(speeches)} speeches to {output_location}")
