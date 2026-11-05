import json
import logging
from pathlib import Path
from typing import List

import pandas as pd
import spacy
from immigration_rhetoric.common.logging_utils import TqdmLoggingHandler
from spacy.tokens import Doc, DocBin
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[TqdmLoggingHandler()],
)

if not Doc.has_extension("snippet_id"):
    Doc.set_extension("snippet_id", default=None)
if not Doc.has_extension("date"):
    Doc.set_extension("date", default=None)

RELEVANCE_LABELS = ["ABOUT_IMMIGRATION", "NOT_ABOUT_IMMIGRATION"]
SENTIMENT_LABELS = [
    "POSITIVE_TOWARDS_IMMIGRATION",
    "NEUTRAL_TOWARDS_IMMIGRATION",
    "NEGATIVE_TOWARDS_IMMIGRATION",
]
DEFAULT_SEEDS = [17, 23, 31, 43, 59]


class TrainingDataBuilder:
    """Converts labelled snippets into spaCy DocBin (+ jsonl) files for training the relevance and sentiment
    models: a fixed holdout set, plus one train/eval split per random seed."""

    def __init__(
        self, output_dir: str, holdout_frac: float = 0.1, eval_frac: float = 0.2
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.holdout_frac = holdout_frac
        self.eval_frac = eval_frac
        self.nlp = spacy.blank("en")

    @staticmethod
    def load_snippets(input_file: str) -> pd.DataFrame:
        """Load consolidated annotations (jsonl with `text`, `accept` and `snippet_metadata`), deriving each
        snippet's relevance and sentiment labels from its accepted label(s)."""
        df = pd.read_json(input_file, lines=True, dtype=False, convert_dates=False)

        unlabelled = df["accept"].apply(len) == 0
        if unlabelled.any():
            logging.info(
                f"Dropping {unlabelled.sum()} snippet(s) with no accepted label"
            )
            df = df.loc[~unlabelled]

        df["snippet_id"] = df["snippet_metadata"].apply(
            lambda metadata: metadata["snippet_id"]
        )
        df["date"] = df["snippet_metadata"].apply(lambda metadata: metadata["date"])
        df["relevance_label"] = df["accept"].apply(
            lambda labels: (
                "NOT_ABOUT_IMMIGRATION"
                if "NOT_ABOUT_IMMIGRATION" in labels
                else "ABOUT_IMMIGRATION"
            )
        )
        # Multi-label snippets get no sentiment label, so only appear in the relevance files
        df["sentiment_label"] = df["accept"].apply(
            lambda labels: (
                labels[0]
                if len(labels) == 1 and labels[0] in SENTIMENT_LABELS
                else None
            )
        )
        return df

    def make_doc(
        self,
        text: str,
        snippet_id: str,
        date: str,
        true_label: str,
        label_set: List[str],
    ) -> Doc:
        """Build a Doc with one-hot `cats` over `label_set` and the snippet's id/date stored as extensions."""
        doc = self.nlp(text)
        doc._.snippet_id = str(snippet_id)
        doc._.date = str(date)
        doc.cats = {label: float(label == true_label) for label in label_set}
        return doc

    def save_docbin_and_jsonl(
        self,
        df: pd.DataFrame,
        label_col: str,
        label_set: List[str],
        out_path_prefix: Path,
    ) -> None:
        """Write `df` as both a .spacy DocBin and an equivalent human-readable .jsonl."""
        doc_bin = DocBin(store_user_data=True)
        jsonl_path = out_path_prefix.with_suffix(".jsonl")
        spacy_path = out_path_prefix.with_suffix(".spacy")

        with jsonl_path.open("w", encoding="utf8") as f:
            for _, row in tqdm(df.iterrows(), total=len(df)):
                doc = self.make_doc(
                    text=row["text"],
                    snippet_id=row["snippet_id"],
                    date=row["date"],
                    true_label=row[label_col],
                    label_set=label_set,
                )
                doc_bin.add(doc)
                record = {
                    "text": doc.text,
                    "cats": doc.cats,
                    "metadata": {"snippet_id": doc._.snippet_id, "date": doc._.date},
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        doc_bin.to_disk(spacy_path)
        logging.info(f"Saved: {spacy_path} and {jsonl_path}")

    def save_relevance_and_sentiment(self, df: pd.DataFrame, name: str) -> None:
        """Save relevance files for all of `df`, and sentiment files for its relevant, sentiment-labelled rows."""
        self.save_docbin_and_jsonl(
            df,
            "relevance_label",
            RELEVANCE_LABELS,
            self.output_dir / f"{name}_relevance",
        )

        sent_df = df[
            (df["relevance_label"] == "ABOUT_IMMIGRATION")
            & (df["sentiment_label"].notna())
        ]
        if sent_df.empty:
            logging.info(f"No sentiment-eligible samples in {name}")
            return
        self.save_docbin_and_jsonl(
            sent_df,
            "sentiment_label",
            SENTIMENT_LABELS,
            self.output_dir / f"{name}_sentiment",
        )

    def save_split_docbins(self, df: pd.DataFrame, name: str, seed: int) -> None:
        """Randomly split `df` into train/eval sets and save each."""
        eval_df = df.sample(frac=self.eval_frac, random_state=seed)
        train_df = df.drop(index=eval_df.index)
        for split, subset in [("eval", eval_df), ("train", train_df)]:
            logging.info(f"Processing {name} - {split}...")
            self.save_relevance_and_sentiment(subset, f"{name}_{split}")

    def build(self, input_file: str, seeds: List[int] = DEFAULT_SEEDS) -> None:
        """Create the holdout files, then a train/eval split for each seed from the remaining snippets."""
        df = self.load_snippets(input_file)

        df_holdout = df.sample(frac=self.holdout_frac, random_state=42)
        logging.info(f"Number of snippets in holdout set: {len(df_holdout)}")
        self.save_relevance_and_sentiment(df_holdout, "holdout")

        df_training = df.loc[~df["snippet_id"].isin(df_holdout["snippet_id"])]
        for seed in seeds:
            logging.info(f"Creating training data - seed {seed}...")
            self.save_split_docbins(df_training, name=f"rs{seed}", seed=seed)
