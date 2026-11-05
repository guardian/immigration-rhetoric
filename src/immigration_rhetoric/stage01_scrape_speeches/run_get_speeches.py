import logging
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path

from immigration_rhetoric.config import DATA_DIR, DATE_TODAY
from immigration_rhetoric.stage01_scrape_speeches.download_debates import (
    DebatesDownloader,
)
from immigration_rhetoric.stage01_scrape_speeches.process_speeches import (
    SpeechesProcessor,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def run_get_speeches(
    output_dir: str, start_date: datetime, end_date: datetime, force_download: bool
) -> None:
    """
    Main function to collect debate data and process speeches.

    Args:
        args: Parsed command line arguments
    """
    xml_output_path = Path(output_dir) / "xml_files"
    speeches_output_path = Path(output_dir) / "raw_speeches"

    try:
        # Download debate XML files
        logging.info("Starting debate XML downloads")
        debate_downloader = DebatesDownloader(
            output_path=xml_output_path,
            start_date=start_date,
            end_date=end_date,
            download_all=force_download,
        )
        debate_downloader.download_debates()

        # Process debates
        logging.info("Starting debate processing")
        speeches_processor = SpeechesProcessor(
            xml_path=xml_output_path,
            output_path=speeches_output_path,
            start_date=start_date,
            end_date=end_date,
        )
        speeches_processor.get_speeches_from_xml()
        # all_speeches = speeches_processor.label_speeches_for_joining(all_speeches)

        logging.info("Speeches download and processing completed successfully")

    except Exception as e:
        logging.error(f"Speeches download and processing failed: {str(e)}")
        raise


def parse_args() -> Namespace:
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        help="Output directory for debate files",
        type=str,
        default=DATA_DIR,
    )
    parser.add_argument(
        "--start-date",
        dest="start_date",
        help="Least recent debates required (format: YYYY-MM-DD)",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        default="1919-02-04",
    )
    parser.add_argument(
        "--end-date",
        dest="end_date",
        help="Most recent debates required (format: YYYY-MM-DD)",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        default=DATE_TODAY,
    )
    parser.add_argument(
        "--force-download",
        dest="force_download",
        help="Ignore XML files already present in the output directory and redownload all",
        default=False,
        action="store_true",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_get_speeches(**vars(args))
