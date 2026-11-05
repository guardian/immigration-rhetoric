import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict

import pandas as pd
import requests
from immigration_rhetoric.common.logging_utils import TqdmLoggingHandler
from tqdm import tqdm

# When output is redirected to a file/log (not an interactive terminal), tqdm can't overwrite its own
# progress line, so every refresh becomes a separate line. Disable the bar in that case.
TQDM_DISABLE = not sys.stderr.isatty()

# Wikimedia's User-Agent policy asks for a descriptive name with a way to contact the maintainers.
WIKIDATA_USER_AGENT = (
    "immigration-rhetoric/1.0 (https://github.com/guardian/immigration-rhetoric; "
    "research code for UK parliamentary speeches) python-requests"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[TqdmLoggingHandler()],
)


class WikidataQuerier:
    """Handles all Wikidata-related queries and operations.

    This class provides methods for querying the Wikidata SPARQL endpoint to retrieve
    data about UK Members of Parliament and parliamentary terms. It includes:
    1. Methods to execute SPARQL queries against Wikidata
    2. Query generators for retrieving parliament term information
    3. Query generators for retrieving MP data within specific terms

    Attributes:
        None - This class uses only static methods
    """

    @staticmethod
    def query_wikidata(query: str, max_attempts: int = 5) -> Dict:
        """Execute a SPARQL query against Wikidata, retrying on transient errors."""

        retryable_status_codes = {429, 500, 502, 503, 504}

        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.get(
                    "https://query.wikidata.org/sparql",
                    headers={"User-Agent": WIKIDATA_USER_AGENT},
                    params={"query": query, "format": "json"},
                    timeout=60,
                )
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ) as e:
                if attempt == max_attempts:
                    raise
                wait_seconds = 2**attempt
                logging.warning(
                    f"Wikidata query attempt {attempt}/{max_attempts} failed ({e}); "
                    f"retrying in {wait_seconds}s"
                )
                time.sleep(wait_seconds)
                continue

            if response.status_code == 200:
                try:
                    return response.json()
                except requests.exceptions.JSONDecodeError as e:
                    # Wikidata can return a 200 whose body is cut off mid-stream.
                    if attempt == max_attempts:
                        raise
                    wait_seconds = 2**attempt
                    logging.warning(
                        f"Wikidata query attempt {attempt}/{max_attempts} returned an "
                        f"incomplete response ({e}); retrying in {wait_seconds}s"
                    )
                    time.sleep(wait_seconds)
                    continue

            if (
                response.status_code not in retryable_status_codes
                or attempt == max_attempts
            ):
                raise Exception(
                    f"Query failed with status code {response.status_code}: {response.text}"
                )

            wait_seconds = 2**attempt
            logging.warning(
                f"Wikidata query attempt {attempt}/{max_attempts} failed "
                f"(status {response.status_code}); retrying in {wait_seconds}s"
            )
            time.sleep(wait_seconds)

    @staticmethod
    def generate_all_parliament_terms_query() -> str:
        """
        Generate a SPARQL query to retrieve data about parliamentary terms.

        This query retrieves information related to parliamentary terms, specifically:
        - Term ID and label (name)
        - Date of inception (if available)
        - Date of dissolution (if available)

        The query filters items that are instances of "parliamentary term" (Q4164871) and are subclasses of
        the taxonomic rank "subclass of taxon" (Q16707842). It includes optional fields for inception and
        dissolution dates of the parliamentary term.

        Returns:
            str: A SPARQL query string to be executed for retrieving parliamentary term data.
        """

        INSTANCE_OF_TERM = "wd:Q4164871"
        SUBCLASS_OF_TAXON = "wd:Q16707842"

        query = f"""
            SELECT ?item ?itemLabel ?inception ?dissolved
            WHERE {{
            ?item wdt:P31 {INSTANCE_OF_TERM};
                    wdt:P279 {SUBCLASS_OF_TAXON}.
            OPTIONAL {{ ?item wdt:P571 ?inception. }}
            OPTIONAL {{ ?item wdt:P576 ?dissolved. }}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
            }}
        """

        return query

    @staticmethod
    def generate_single_parliament_term_query(term_wiki_id: str) -> str:
        """
        Generate a SPARQL query to retrieve MP data for a specific parliament term.

        This query retrieves information about Members of Parliament (MPs) who served during
        a specified parliamentary term, including:
        - MP ID and name
        - Alternative name (if available, in English)
        - TheyWorkForYou ID (if available)
        - Party affiliation and dates
        - Constituency information

        Args:
            term_wiki_id (str): Wikidata ID for the parliamentary term

        Returns:
            str: SPARQL query string for retrieving data for a single parliamentary term
        """
        query = f"""
            SELECT ?person ?personLabel ?altLabel ?theyWorkForYouID ?partyLabel ?partyStartDate ?partyEndDate ?constituencyLabel
            WHERE {{
            ?person wdt:P39 wd:{term_wiki_id}.
            ?person p:P39 ?partyStatement.
            ?partyStatement ps:P39 ?positionHeld.
            ?partyStatement ps:P39 wd:{term_wiki_id}.

            ?partyStatement pq:P4100 ?party.
            ?partyStatement pq:P768 ?constituency.

            OPTIONAL {{ ?person wdt:P2171 ?theyWorkForYouID. }}

            OPTIONAL {{
                ?person skos:altLabel ?altLabel.
                FILTER (LANG(?altLabel) = "en")
            }}

            OPTIONAL {{ ?partyStatement pq:P580 ?partyStartDate. }}
            OPTIONAL {{ ?partyStatement pq:P582 ?partyEndDate. }}

            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
        }}
        """
        return query


class SpeakerMetadataDownloader:
    """Collects and processes metadata about Members of Parliament from multiple sources.

    This class handles the complete pipeline for gathering MP metadata:
    1. Downloading TheyWorkForYou people data
    2. Retrieving parliament term information from Wikidata
    3. Querying MP data for each parliament term
    4. Processing and cleaning the collected metadata
    5. Saving to CSV format

    Attributes:
        metadata_dir: Directory for storing metadata files
        people_json_url: URL for TheyWorkForYou people data
        wikidata_querier: Helper class for Wikidata queries
        force_refresh: Whether to force refresh cached data
    """

    def __init__(self, output_path: str | Path, force_refresh: bool = False):
        self.people_json_url = "https://www.theyworkforyou.com/pwdata/people.json"
        self.wikidata_querier = WikidataQuerier()
        self.force_refresh = force_refresh

        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def collect_speaker_metadata(self) -> None:
        """Collect and save speaker metadata."""
        try:
            logging.info("Collecting speaker metadata")
            self.download_people_json()
            parliament_term_ids = self.get_uk_parliament_term_ids()
            self.get_mps_from_all_parliament_terms(parliament_term_ids)
        except Exception as e:
            logging.error(f"Error collecting speaker metadata: {str(e)}")
            raise

    def download_people_json(self, save_file_name: str = "people.json") -> None:
        """Download and save TheyWorkForYou people data if not already cached."""
        cache_location = self.output_path / save_file_name

        if cache_location.exists() and not self.force_refresh:
            logging.info(f"People data already cached at {cache_location}")
            return

        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                content = self._stream_download(self.people_json_url)
                json.dump(json.loads(content), cache_location.open("w"), indent=4)
                logging.info(
                    f"JSON file successfully downloaded and saved to {cache_location}"
                )
                return
            except (
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ) as e:
                if attempt == max_attempts:
                    logging.error(
                        f"Failed to download JSON file after {max_attempts} attempts: {e}"
                    )
                    raise
                wait_seconds = 2**attempt
                logging.warning(
                    f"Download attempt {attempt}/{max_attempts} failed ({e}); "
                    f"retrying in {wait_seconds}s"
                )
                time.sleep(wait_seconds)
            except requests.exceptions.RequestException as e:
                logging.error(f"Failed to download JSON file: {e}")
                raise

    @staticmethod
    def _stream_download(url: str, timeout: int = 60) -> bytes:
        """Download a URL's content in chunks, tolerating slow/unstable connections."""
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        chunks = []
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            chunks.append(chunk)

        return b"".join(chunks)

    def get_uk_parliament_term_ids(self) -> pd.DataFrame:
        """Retrieve and process parliament term data, using cache if available."""
        cache_location = self.output_path / "parliament_term_metadata.csv"

        if cache_location.exists() and not self.force_refresh:
            logging.info(f"Loading cached parliament term data from {cache_location}")
            return pd.read_csv(cache_location)

        logging.info("Scraping parliament term IDs from Wikidata")

        data = self.wikidata_querier.query_wikidata(
            self.wikidata_querier.generate_all_parliament_terms_query()
        )

        results = data.get("results", {}).get("bindings", [])
        parliament_data_list = [
            {
                "item": result.get("item", {}).get("value"),
                "itemLabel": result.get("itemLabel", {}).get("value"),
                "inception": result.get("inception", {}).get("value"),
                "dissolved": result.get("dissolved", {}).get("value"),
            }
            for result in results
        ]

        df = pd.DataFrame(parliament_data_list)
        df["term_wikiID"] = df["item"].apply(lambda x: x.split("/")[-1])
        df["term_numeric"] = df["itemLabel"].apply(
            lambda x: re.findall(r"\d+", x)[0] if re.findall(r"\d+", x) else None
        )

        df.to_csv(cache_location, index=False)
        logging.info(
            f"Found {len(df)} parliament terms ({df['inception'].min()} to {df['inception'].max()}), "
            f"saved to {cache_location}"
        )

        return df

    def get_mps_from_all_parliament_terms(self, parliament_terms: pd.DataFrame) -> None:
        """Collect and save MP data from all parliament terms if not already cached."""
        cache_location = self.output_path / "mps_wikidata.csv"

        if cache_location.exists() and not self.force_refresh:
            logging.info(f"Using cached MP data from {cache_location}")
            return

        mps_data = []

        for _, row in tqdm(
            parliament_terms.iterrows(),
            total=len(parliament_terms),
            desc="Collecting MPs from all parliament terms",
            disable=TQDM_DISABLE,
        ):
            mps_this_parliament = self._get_mps_from_single_parliament(
                row["term_numeric"], row["term_wikiID"]
            )
            if len(mps_this_parliament) > 0:
                mps_data.append(mps_this_parliament)

        if not mps_data:
            logging.warning("No MP data collected")
            return

        combined_data = pd.concat(mps_data)
        combined_data["mp_wikiID"] = combined_data["person"].apply(
            lambda x: x.split("/")[-1]
        )

        combined_data.to_csv(cache_location, index=False)
        logging.info(
            f"Found {len(combined_data)} MPs ({combined_data.mp_wikiID.nunique()} unique wikidata ids, "
            f"{combined_data.theyWorkForYouID.nunique()} unique theyworkforyou ids, "
            f"{combined_data.partyLabel.nunique()} parties, "
            f"{combined_data.partyLabel.isnull().sum()} missing a party), saved to {cache_location}"
        )

    def _get_mps_from_single_parliament(
        self, term_number: str, term_wiki_id: str
    ) -> pd.DataFrame:
        """
        Retrieve MP data for a specific parliamentary term from Wikidata.

        Args:
            term_number (str): The numeric identifier of the parliamentary term
            term_wiki_id (str): The Wikidata ID of the parliamentary term

        Returns:
            pd.DataFrame: DataFrame containing MP data including:
                - Personal identifiers
                - Party affiliations
                - Constituency information
                - Term dates
        """
        cache_location = self.output_path / f"mps_parliament_{term_number}.csv"

        if cache_location.exists() and not self.force_refresh:
            logging.debug(f"Using cached data for parliament term {term_number}")
            return pd.read_csv(cache_location)

        data = self.wikidata_querier.query_wikidata(
            self.wikidata_querier.generate_single_parliament_term_query(term_wiki_id)
        )

        results = []
        for item in data["results"]["bindings"]:
            row = {
                "parliament_term": term_number,
                "parliament_term_id": term_wiki_id,
            }
            row.update({var: item[var]["value"] for var in item})
            results.append(row)

        output = pd.DataFrame(results)
        output.to_csv(cache_location, index=False)

        return output
