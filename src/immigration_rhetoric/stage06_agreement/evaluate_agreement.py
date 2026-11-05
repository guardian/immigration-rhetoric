import logging
from pathlib import Path
from typing import List

import krippendorff
import numpy as np
import pandas as pd
from immigration_rhetoric.common.logging_utils import TqdmLoggingHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[TqdmLoggingHandler()],
)

DEFAULT_LABELS = [
    "POSITIVE_TOWARDS_IMMIGRATION",
    "NEGATIVE_TOWARDS_IMMIGRATION",
    "NEUTRAL_TOWARDS_IMMIGRATION",
    "NOT_ABOUT_IMMIGRATION",
]


class AnnotatorAgreement:
    """Computes Krippendorff's alpha (inter-annotator agreement) over a set of annotators' snippet labels."""

    def __init__(self, input_dir: str):
        self.input_dir = Path(input_dir)

    def load_annotations(self, filename: str) -> pd.DataFrame:
        """Load a 'multi' annotations file (e.g. produced by stage06_agreement's consolidate_annotations)."""
        filepath = self.input_dir / filename
        logging.info(f"Loading annotations from {filepath}")
        return pd.read_json(filepath, lines=True)

    @staticmethod
    def get_agreement_score(
        annotations_df: pd.DataFrame,
        unique_annotators: List[str],
        labels: List[str] = DEFAULT_LABELS,
    ) -> float:
        """Compute Krippendorff's alpha for `unique_annotators`' labels of every snippet in `annotations_df`."""
        df = annotations_df.copy()
        df["accept_str"] = df["accept"].apply(
            lambda a: a[0] if a and len(a) == 1 else ""
        )

        unique_snippets = df["text"].unique()
        encoded_labels = {label: i for i, label in enumerate(labels)}
        annotator_index = {
            annotator: i for i, annotator in enumerate(unique_annotators)
        }
        snippet_index = {snippet: i for i, snippet in enumerate(unique_snippets)}

        # krippendorff.alpha expects one row per coder, one column per unit, np.nan for not-rated units
        reliability_data = np.full(
            (len(unique_annotators), len(unique_snippets)), np.nan
        )
        for _, row in df.iterrows():
            if (
                row["accept_str"] in encoded_labels
                and row["_annotator_id"] in annotator_index
            ):
                reliability_data[
                    annotator_index[row["_annotator_id"]], snippet_index[row["text"]]
                ] = encoded_labels[row["accept_str"]]

        return krippendorff.alpha(
            reliability_data=reliability_data, level_of_measurement="nominal"
        )
