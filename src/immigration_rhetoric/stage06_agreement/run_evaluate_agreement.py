import logging
from argparse import ArgumentParser, Namespace
from typing import Dict, List

from immigration_rhetoric.config import ANNOTATIONS_DIR
from immigration_rhetoric.stage06_agreement.evaluate_agreement import AnnotatorAgreement


def run_evaluate_agreement(
    input_dir: str, annotations_file: str, annotator_groups: List[str]
) -> Dict[str, float]:
    """Compute Krippendorff's alpha for each `name=id1,id2,...` group in `annotator_groups`."""
    agreement = AnnotatorAgreement(input_dir=input_dir)
    annotations = agreement.load_annotations(annotations_file)

    scores = {}
    for group in annotator_groups:
        name, ids = group.split("=", 1)
        annotator_list = ids.split(",")
        score = agreement.get_agreement_score(annotations, annotator_list)
        scores[name] = score
        logging.info(f"{name}, agreement: {score}")
    return scores


def parse_args() -> Namespace:
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-dir",
        dest="input_dir",
        help="Directory containing the 'multi' annotations file",
        type=str,
        default=str(ANNOTATIONS_DIR / "processed_annotations"),
    )
    parser.add_argument(
        "-f",
        "--annotations-file",
        dest="annotations_file",
        help="Filename of the 'multi' annotations jsonl to compute agreement over",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-g",
        "--annotator-group",
        dest="annotator_groups",
        help="A named group of annotator ids to score together, as 'name=id1,id2,id3'. Repeatable.",
        type=str,
        action="append",
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluate_agreement(**vars(args))
