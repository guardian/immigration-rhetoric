import json
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import spacy
from immigration_rhetoric.common.logging_utils import TqdmLoggingHandler
from immigration_rhetoric.config import STYLES_DIR
from sklearn.metrics import confusion_matrix
from spacy.scorer import Scorer
from spacy.tokens import Doc, DocBin
from spacy.training import Example
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[TqdmLoggingHandler()],
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if not Doc.has_extension("date"):
    Doc.set_extension("date", default=None)
if not Doc.has_extension("snippet_id"):
    Doc.set_extension("snippet_id", default=None)

LABELS = [
    "POSITIVE_TOWARDS_IMMIGRATION",
    "NEUTRAL_TOWARDS_IMMIGRATION",
    "NEGATIVE_TOWARDS_IMMIGRATION",
    "NOT_ABOUT_IMMIGRATION",
]

LABELS_SHORT = ["POSITIVE", "NEUTRAL", "NEGATIVE", "NOT_ABOUT"]

GUARDIAN_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "guardian_blue",
    ["#ffffff", "#506991", "#052962", "#041f4a"],
    N=256,
)

PERIOD_RANGES = [
    (pd.to_datetime("1919-01-01"), pd.to_datetime("1979-12-31")),
    (pd.to_datetime("1980-01-01"), pd.to_datetime("2020-12-31")),
    (pd.to_datetime("2021-01-01"), pd.Timestamp.max),
]
PERIOD_LABELS = ["before_1980", "1980_to_2020", "2021_onwards"]


class ModelEvaluator:
    """Evaluates the combined relevance + sentiment models on a shared evaluation set and reports metrics."""

    def __init__(self):
        plt.style.use(
            [
                str(STYLES_DIR / "base.mplstyle"),
                str(STYLES_DIR / "immigration.mplstyle"),
            ]
        )

    def evaluate_models(
        self,
        relevance_model: Path,
        sentiment_model: Path,
        relevance_eval: Path,
        sentiment_eval: Path,
        output_path_metrics: Path,
        output_path_predictions: Path,
        plot: Optional[Path] = None,
        gpu_id: int = -1,
    ) -> None:
        """Scores the combined relevance+sentiment prediction against a shared gold evaluation set."""
        if gpu_id >= 0:
            spacy.require_gpu(gpu_id)

        logging.info("Loading models...")
        nlp_relevance = spacy.load(relevance_model)
        vocab = nlp_relevance.vocab
        nlp_sentiment = spacy.load(sentiment_model, vocab=vocab)

        logging.info("Loading and combining evaluation datasets...")
        gold_docs = []

        doc_bin_sentiment = DocBin().from_disk(sentiment_eval)
        for doc in doc_bin_sentiment.get_docs(vocab):
            doc.cats["NOT_ABOUT_IMMIGRATION"] = 0.0
            gold_docs.append(doc)

        doc_bin_relevance = DocBin().from_disk(relevance_eval)
        docs_relevance_not_about = [
            d
            for d in doc_bin_relevance.get_docs(vocab)
            if d.cats["NOT_ABOUT_IMMIGRATION"] > 0.5
        ]
        for doc in docs_relevance_not_about:
            doc.cats["POSITIVE_TOWARDS_IMMIGRATION"] = 0.0
            doc.cats["NEUTRAL_TOWARDS_IMMIGRATION"] = 0.0
            doc.cats["NEGATIVE_TOWARDS_IMMIGRATION"] = 0.0
            doc.cats.pop("ABOUT_IMMIGRATION", None)
        gold_docs.extend(docs_relevance_not_about)

        logging.info(f"Total documents to evaluate: {len(gold_docs)}")

        # nlp.pipe writes predictions into doc.cats in place, so keep an untouched copy of the gold labels.
        reference_docs = deepcopy(gold_docs)

        # Both models annotate the same Doc objects (their labels don't overlap), so each doc ends up with
        # both models' scores and the two lists below hold the same docs.
        logging.info("Running predictions with relevance model...")
        predicted_relevance_docs = [
            d for d in tqdm(nlp_relevance.pipe(gold_docs), total=len(gold_docs))
        ]

        logging.info("Running predictions with sentiment model...")
        predicted_sentiment_docs = [
            d for d in tqdm(nlp_sentiment.pipe(gold_docs), total=len(gold_docs))
        ]

        logging.info("Combining predictions for scoring...")
        predicted_docs = []
        examples = []
        rows = []

        for ref_doc, rel_pred, sent_pred in zip(
            reference_docs, predicted_relevance_docs, predicted_sentiment_docs
        ):
            is_about_pred = rel_pred.cats.get("NOT_ABOUT_IMMIGRATION", 0.0) <= 0.5

            combined_doc = rel_pred.copy()
            combined_doc.cats = {
                "POSITIVE_TOWARDS_IMMIGRATION": (
                    sent_pred.cats.get("POSITIVE_TOWARDS_IMMIGRATION", 0.0)
                    if is_about_pred
                    else 0.0
                ),
                "NEUTRAL_TOWARDS_IMMIGRATION": (
                    sent_pred.cats.get("NEUTRAL_TOWARDS_IMMIGRATION", 0.0)
                    if is_about_pred
                    else 0.0
                ),
                "NEGATIVE_TOWARDS_IMMIGRATION": (
                    sent_pred.cats.get("NEGATIVE_TOWARDS_IMMIGRATION", 0.0)
                    if is_about_pred
                    else 0.0
                ),
                "NOT_ABOUT_IMMIGRATION": (
                    rel_pred.cats.get("NOT_ABOUT_IMMIGRATION", 0.0)
                    if not is_about_pred
                    else 0.0
                ),
            }
            combined_doc._.date = ref_doc._.date
            combined_doc._.snippet_id = ref_doc._.snippet_id

            predicted_docs.append(combined_doc)
            examples.append(Example(combined_doc, ref_doc))

            true_combined_label = max(
                {k: v for k, v in ref_doc.cats.items() if k in LABELS},
                key=lambda label: ref_doc.cats[label],
                default="UNKNOWN",
            )
            predicted_label = max(combined_doc.cats, key=combined_doc.cats.get)

            rows.append(
                {
                    "date": combined_doc._.date,
                    "text": ref_doc.text,
                    "predicted_combined_label": predicted_label,
                    "predicted_combined_probs": combined_doc.cats,
                    "true_combined_label": true_combined_label,
                }
            )

        logging.info("Scoring predictions...")
        metrics = Scorer().score_cats(examples, "cats", labels=LABELS)
        with open(output_path_metrics, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)
        logging.info(f"Evaluation metrics saved to: {output_path_metrics}")

        logging.info("Writing predictions to JSONL...")
        pd.DataFrame(rows).to_json(
            output_path_predictions, orient="records", lines=True
        )
        logging.info(f"Predictions saved to: {output_path_predictions}")

        if plot:
            logging.info("Generating confusion matrix plot...")
            true_labels = [max(doc.cats, key=doc.cats.get) for doc in reference_docs]
            predicted_labels = [
                max(doc.cats, key=doc.cats.get) for doc in predicted_docs
            ]
            cm = confusion_matrix(true_labels, predicted_labels, labels=LABELS)
            self.plot_confusion_matrix_with_style(cm, LABELS_SHORT, save_path=plot)
            self.plot_confusion_matrix_per_period(reference_docs, predicted_docs, plot)

    def plot_confusion_matrix_per_period(
        self, docs, predicted_docs, plot: Path
    ) -> None:
        """Generates a separate confusion matrix plot for each of `PERIOD_RANGES` (e.g. pre/post-1980)."""
        for (start_date, end_date), label in zip(PERIOD_RANGES, PERIOD_LABELS):
            period_docs = [
                doc
                for doc in docs
                if start_date <= pd.to_datetime(doc._.date) <= end_date
            ]
            period_predicted_docs = [
                doc
                for doc in predicted_docs
                if start_date <= pd.to_datetime(doc._.date) <= end_date
            ]

            if not period_docs:
                logging.info(f"No documents found for period {label}, skipping.")
                continue

            logging.info(f"Generating confusion matrix for period {label}...")
            true_labels = [max(doc.cats, key=doc.cats.get) for doc in period_docs]
            predicted_labels = [
                max(doc.cats, key=doc.cats.get) for doc in period_predicted_docs
            ]
            cm = confusion_matrix(true_labels, predicted_labels, labels=LABELS)
            self.plot_confusion_matrix_with_style(
                cm, LABELS_SHORT, save_path=str(plot).split(".png")[0] + f"_{label}.png"
            )

    def plot_confusion_matrix_with_style(self, cm, labels, save_path=None):
        """Plot a confusion matrix heatmap using the Guardian colour scheme."""
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=GUARDIAN_CMAP,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=300)
            logging.info(f"Confusion matrix saved to: {save_path}")

        return fig, ax
