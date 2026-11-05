import logging
from pathlib import Path
from typing import List

import pandas as pd
from immigration_rhetoric.common.logging_utils import TqdmLoggingHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[TqdmLoggingHandler()],
)


class ConsolidateAnnotations:
    """Combines one or more annotators' raw Prodigy-style annotation exports into:

    - a "consolidated" file: one row per snippet, keeping only snippets a single annotator (or several
      agreeing annotators) labelled, suitable as training data for stage07_model_training.
    - a "multi" file: every row for snippets labelled by more than one annotator, suitable as input to
      stage06_agreement's evaluate_agreement module (inter-annotator agreement needs every annotator's
      individual label for each snippet, not a single consolidated one).
    """

    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

    def load_annotations(self, filename: str) -> pd.DataFrame:
        """Load a single annotator (or annotation run)'s raw `{filename}.jsonl` export."""
        filepath = self.input_dir / f"{filename}.jsonl"
        logging.info(f"Reading annotations from {filepath}")
        return pd.read_json(filepath, lines=True)

    def output_annotations(self, df: pd.DataFrame, filename: str) -> None:
        """Write `df` to `{output_dir}/{filename}.jsonl`, creating the directory if needed."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.output_dir / f"{filename}.jsonl"
        df.to_json(filepath, orient="records", lines=True)
        logging.info(f"Saved {len(df)} annotation(s) to {filepath}")

    @staticmethod
    def validate_annotations(df: pd.DataFrame) -> None:
        """Reject conflicting ratings for one snippet from the same annotator."""
        ratings = df.assign(_accept_key=df["accept"].apply(tuple))
        rating_counts = ratings.groupby(["text", "_annotator_id"], sort=False)[
            "_accept_key"
        ].nunique()
        conflicting_assignments = rating_counts[rating_counts > 1]

        if not conflicting_assignments.empty:
            duplicate_assignments = [
                {"text": text, "_annotator_id": annotator_id}
                for text, annotator_id in conflicting_assignments.index
            ]
            raise ValueError(
                "Multiple ratings found for the same snippet and annotator: "
                f"{duplicate_assignments}"
            )

    @staticmethod
    def deduplicate_annotations(df: pd.DataFrame) -> pd.DataFrame:
        """Remove repeated identical ratings from the same annotator."""
        ratings = df.assign(_accept_key=df["accept"].apply(tuple))
        return (
            ratings.drop_duplicates(
                subset=["text", "_annotator_id", "_accept_key"], keep="first"
            )
            .drop(columns=["_accept_key"])
            .reset_index(drop=True)
        )

    @staticmethod
    def consolidate_answers(df: pd.DataFrame) -> pd.DataFrame:
        """Combine duplicate rows where annotators agree into a single row, and drop rows where they disagree."""
        df = df.copy()
        df["accept_str"] = df["accept"].apply(
            lambda a: a[0] if a and len(a) == 1 else ""
        )

        consolidated = (
            df.drop_duplicates(subset=["text", "accept_str"], keep="first")
            .drop_duplicates(subset=["text"], keep=False)
            .drop(columns=["accept_str"])
            .reset_index(drop=True)
        )
        logging.info(f"Consolidated to {len(consolidated)} snippet(s)")
        return consolidated

    @staticmethod
    def get_duplicate_annotations(df: pd.DataFrame) -> pd.DataFrame:
        """Return every row for snippets that were labelled by more than one annotator."""
        duplicates = df[df["text"].duplicated(keep=False)].reset_index(drop=True)
        logging.info(
            f"Found {len(duplicates)} annotation(s) for multiply-labelled snippets"
        )
        return duplicates

    def consolidate(self, files: List[str], label: str, date: str) -> None:
        """Load one or more annotators' `files`, and write their consolidated + multi-annotator outputs."""
        df = pd.concat([self.load_annotations(f) for f in files], ignore_index=True)
        logging.info(f"Loaded {len(df)} annotation(s) for '{label}'")
        self.validate_annotations(df)
        df = self.deduplicate_annotations(df)

        self.output_annotations(
            self.consolidate_answers(df), f"{label}_{date}_consolidated"
        )
        self.output_annotations(
            self.get_duplicate_annotations(df), f"{label}_{date}_multi"
        )
