import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
from immigration_rhetoric.common.logging_utils import TqdmLoggingHandler
from immigration_rhetoric.common.utils import extract_numbers_as_int
from immigration_rhetoric.config import DATA_DIR, THIS_YEAR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[TqdmLoggingHandler()],
)

COLUMN_NAMES = {
    "id": "speech_id",
    "text": "speech_text",
    "constituencyLabel": "constituency_label",
    "partyLabel": "party_label",
    "personLabel": "person_label",
    "date": "date",
}


def select_speech_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keeps the speech id, text, speaker metadata and date, with snake_case column names."""
    return df[list(COLUMN_NAMES)].rename(columns=COLUMN_NAMES)


def run_combine_speeches(
    input_dir: str, output_dir: str, start_year: int, end_year: int
) -> None:
    """Write each year's speeches with a reduced set of columns, plus one file of all years combined."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    yearly_dir = output_dir / "clean_speeches"
    yearly_dir.mkdir(parents=True, exist_ok=True)

    all_speeches = []
    for file_path in sorted(input_dir.glob("speeches_*.csv")):
        year = extract_numbers_as_int(file_path.name)
        if not start_year <= year <= end_year:
            continue
        df = select_speech_columns(pd.read_csv(file_path, low_memory=False))
        df.to_csv(yearly_dir / f"clean_speeches_{year}.csv", index=False)
        all_speeches.append(df)

    output_path = output_dir / "all_clean_speeches.csv"
    pd.concat(all_speeches, ignore_index=True).to_csv(output_path, index=False)
    logging.info(
        f"Wrote {len(all_speeches)} year(s) of speeches to {yearly_dir} and {output_path}"
    )


def parse_args() -> Namespace:
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-dir",
        dest="input_dir",
        help="Directory of speeches_{year}.csv files",
        type=str,
        default=str(DATA_DIR / "speeches"),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        help="Directory to write clean_speeches/ and all_clean_speeches.csv to",
        type=str,
        default=str(DATA_DIR / "analysis_datasets"),
    )
    parser.add_argument(
        "--start-year",
        dest="start_year",
        help="Least recent speeches required (format: YYYY)",
        type=int,
        default=1919,
    )
    parser.add_argument(
        "--end-year",
        dest="end_year",
        help="Most recent speeches required (format: YYYY)",
        type=int,
        default=THIS_YEAR,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_combine_speeches(**vars(args))
