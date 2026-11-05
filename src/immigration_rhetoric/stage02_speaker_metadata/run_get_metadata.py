import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

from immigration_rhetoric.config import DATA_DIR
from immigration_rhetoric.stage02_speaker_metadata.download_speaker_metadata import (
    SpeakerMetadataDownloader,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def run_get_metadata(output_dir: str, force_download: bool) -> None:
    """
    Main function to download speaker metadata (TheyWorkForYou people data and Wikidata MP data).

    Args:
        output_dir: Directory under which speaker_metadata/ is created
        force_download: Ignore any cached files and redownload everything
    """
    output_path = Path(output_dir) / "speaker_metadata"

    try:
        speaker_metadata_downloader = SpeakerMetadataDownloader(
            output_path=output_path, force_refresh=force_download
        )
        speaker_metadata_downloader.collect_speaker_metadata()

    except Exception as e:
        logging.error(f"Speaker metadata download failed: {str(e)}")
        raise


def parse_args() -> Namespace:
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        help="Output directory for speaker metadata files",
        type=str,
        default=DATA_DIR,
    )
    parser.add_argument(
        "--force-download",
        dest="force_download",
        help="Ignore MP data files already present in the output directory and redownload all",
        default=False,
        action="store_true",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_get_metadata(**vars(args))
