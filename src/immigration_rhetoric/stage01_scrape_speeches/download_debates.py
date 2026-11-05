import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from immigration_rhetoric.common.logging_utils import TqdmLoggingHandler
from immigration_rhetoric.common.utils import extract_date
from tqdm import tqdm

# Configure logging to show INFO level messages
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[TqdmLoggingHandler()],
)


class DebatesDownloader:
    """Downloads and preprocesses raw parliamentary debate data from TheyWorkForYou.

    This class handles the complete pipeline for downloading debate XML files:
    1. Fetching list of available XML files from TheyWorkForYou
    2. Filtering files by date range
    3. Downloading new or missing files
    4. Saving files to the output directory
    5. Handling incremental updates

    Attributes:
        xml_data_url: Base URL for the XML debate files
        output_path: Base directory for storing downloaded files
        xml_output_path: Directory specifically for XML debate files
    """

    def __init__(
        self,
        output_path: str | Path,
        start_date: datetime = datetime(1919, 2, 4),
        end_date: datetime = datetime.today(),
        download_all: bool = False,
    ) -> None:
        self.xml_data_url = "https://www.theyworkforyou.com/pwdata/scrapedxml/debates/"
        self.output_path = Path(output_path)
        self.start_date = start_date
        self.end_date = end_date
        self.download_all = download_all

    def download_debates(
        self,
    ) -> None:
        """Downloads parliamentary debate XML files within a date range."""
        # Ensure output directory exists
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Get available files and prepare download list
        available_files = self._fetch_available_xml_files()
        files_to_download = self._filter_download_list_by_date(
            available_files, self.start_date, self.end_date, self.download_all
        )

        # Download files
        if len(files_to_download) > 0:
            for file in tqdm(
                files_to_download["xml_file"].tolist(),
                desc=f"Downloading {len(files_to_download)} debate files",
            ):
                self._download_xml_file(
                    filename_to_download=file, output_location=self.output_path
                )

    def _fetch_available_xml_files(self) -> list:
        """Fetches list of available XML files from the data repository.

        Returns:
            List of XML filenames available for download
        """
        try:
            response = requests.get(self.xml_data_url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            xml_links = [
                link.get("href")
                for link in soup.find_all("a")
                if link.get("href", "").endswith(".xml")
            ]

            logging.info(f"Found {len(xml_links)} XML files available")
            return xml_links
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch XML file list: {str(e)}")
            raise

    def _filter_download_list_by_date(
        self,
        xml_files: list[str],
        start_date: datetime,
        end_date: datetime,
        download_all: bool,
    ) -> pd.DataFrame:
        """Prepares the final list of files to download based on date range and existing files.

        Returns:
            DataFrame containing filtered list of files to download with their dates
        """
        # Create date-filtered DataFrame
        files_df = pd.DataFrame({"xml_file": xml_files})
        files_df["date"] = files_df["xml_file"].apply(extract_date)

        date_filtered = files_df[
            (files_df["date"] >= start_date) & (files_df["date"] <= end_date)
        ]

        logging.info(
            f"Date range filtered: {date_filtered['date'].min()} to {date_filtered['date'].max()}"
        )

        if not download_all:
            existing_files = {file.name for file in self.output_path.glob("*.xml")}
            date_filtered = date_filtered[
                ~date_filtered["xml_file"].isin(existing_files)
            ]
            logging.info(f"Skipping {len(existing_files)} already downloaded files")

        return date_filtered

    def _download_xml_file(
        self, filename_to_download: str, output_location: str | Path
    ) -> None:
        """Downloads a single XML file from the base URL."""
        try:
            file_url = self.xml_data_url + filename_to_download
            file_path = Path(output_location) / filename_to_download

            response = requests.get(file_url, timeout=30)
            response.raise_for_status()

            file_path.write_bytes(response.content)
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download {filename_to_download}: {str(e)}")
            raise
