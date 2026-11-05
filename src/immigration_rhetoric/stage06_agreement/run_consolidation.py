from argparse import ArgumentParser, Namespace
from typing import List

from immigration_rhetoric.config import ANNOTATIONS_DIR
from immigration_rhetoric.stage06_agreement.consolidate_annotations import (
    ConsolidateAnnotations,
)


def run_consolidation(
    input_dir: str, output_dir: str, files: List[str], label: str, date: str
) -> None:
    """Consolidate one or more annotators' raw annotation exports for a given `label` (e.g. 'human', 'llm')."""
    consolidator = ConsolidateAnnotations(input_dir=input_dir, output_dir=output_dir)
    consolidator.consolidate(files=files, label=label, date=date)


def parse_args() -> Namespace:
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-dir",
        dest="input_dir",
        help="Directory containing the raw annotation jsonl files",
        type=str,
        default=str(ANNOTATIONS_DIR / "raw_annotations"),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        help="Directory to write the consolidated/multi-annotator jsonl files to",
        type=str,
        default=str(ANNOTATIONS_DIR / "processed_annotations"),
    )
    parser.add_argument(
        "-f",
        "--files",
        dest="files",
        help="One or more raw annotation filenames (without the .jsonl extension) to combine",
        type=str,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "-l",
        "--label",
        dest="label",
        help="Label for this combination of files, e.g. 'human', 'llm', 'human_and_llm'",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-d",
        "--date",
        dest="date",
        help="Date to include in the output filenames, e.g. the date of the most recent annotation run",
        type=str,
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_consolidation(**vars(args))
