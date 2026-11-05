import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
from immigration_rhetoric.config import DATA_DIR, THIS_YEAR
from immigration_rhetoric.stage02_speaker_metadata.preprocess_speaker_metadata import (
    SpeakerMetadataPreprocessor,
)
from immigration_rhetoric.stage02_speaker_metadata.speeches_join_metadata import (
    SpeechMetadataJoiner,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def run_speech_speaker_joins(
    metadata_dir: str,
    raw_speech_dir: str,
    start_year: int,
    end_year: int,
    output_dir: str,
) -> None:
    """Join raw speeches to speaker metadata, matching on person ID, membership ID or name."""
    # some of the joining logic below intentionally does chained assignment
    pd.options.mode.chained_assignment = None

    metadata_preprocessing = SpeakerMetadataPreprocessor(input_dir=metadata_dir)

    # preprocessing for files from people.json
    _, memberships, _ = metadata_preprocessing.load_people_json()
    memberships = metadata_preprocessing.preprocess_memberships(memberships)

    # preprocessing for wikidata
    wikidata = metadata_preprocessing.load_wikidata()
    wikidata_clean = metadata_preprocessing.preprocess_wikidata(wikidata)

    # load and preprocess speeches
    speech_joiner = SpeechMetadataJoiner(
        raw_speeches_dir=raw_speech_dir, output_dir=output_dir
    )
    all_speeches = speech_joiner.load_speech_csv(min_year=start_year, max_year=end_year)
    preprocessed_speeches = speech_joiner.prepare_speeches_for_join(all_speeches)

    # split speeches into stages for joining
    stage1, stage2, stage3 = speech_joiner.split_speeches_into_join_stages(
        preprocessed_speeches
    )

    logging.info("Stage 1: joining on person_id")
    if len(stage1) > 0:
        stage1 = speech_joiner.prepare_stage1(stage1)
        stage1_joined = speech_joiner.join_speeches_on_person_id(stage1, wikidata_clean)
    else:
        stage1_joined = pd.DataFrame(columns=all_speeches.columns)

    logging.info("Stage 2: joining on membership_id")
    if len(stage2) > 0:
        stage2 = speech_joiner.prepare_stage2(stage2)
        stage2_joined = speech_joiner.join_speeches_on_membership_id(
            stage2, memberships, wikidata_clean
        )
    else:
        stage2_joined = pd.DataFrame(columns=all_speeches.columns)

    logging.info("Stage 3: joining on speaker name")
    stage3 = speech_joiner.prepare_stage3(stage3)
    stage3_joined = speech_joiner.name_joining(stage3, wikidata_clean)

    # output joined speeches
    all_joined_speeches = speech_joiner.concat_joined_speeches(
        stage1_joined, stage2_joined, stage3_joined
    )
    speech_joiner.save_joined_speeches(all_joined_speeches)


def parse_args() -> Namespace:
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "--metadata-dir",
        dest="metadata_dir",
        help="Directory where speaker metadata is saved",
        type=str,
        default=str(Path(DATA_DIR) / "speaker_metadata"),
    )
    parser.add_argument(
        "--raw-speech-dir",
        dest="raw_speech_dir",
        help="Directory where raw speech csvs are saved",
        type=str,
        default=str(Path(DATA_DIR) / "raw_speeches"),
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
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        help="Output directory for joined speech files",
        type=str,
        default=str(Path(DATA_DIR) / "speeches"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_speech_speaker_joins(**vars(args))
