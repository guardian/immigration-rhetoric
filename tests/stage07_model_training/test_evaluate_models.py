import json
from unittest.mock import patch

import matplotlib
import pandas as pd
import pytest
import spacy
from immigration_rhetoric.stage07_model_training.evaluate_models import ModelEvaluator
from spacy.tokens import DocBin

matplotlib.use("Agg")

MODULE = "immigration_rhetoric.stage07_model_training.evaluate_models"


class FakeModel:
    """Stands in for a trained textcat pipeline, predicting fixed cats per snippet_id."""

    def __init__(self, vocab, predictions):
        self.vocab = vocab
        self.predictions = predictions

    def pipe(self, docs):
        for doc in docs:
            predicted = doc.copy()
            predicted.cats = dict(self.predictions[doc._.snippet_id])
            yield predicted


def _write_docbin(path, nlp, items):
    db = DocBin(store_user_data=True)
    for snippet_id, date, cats in items:
        doc = nlp(f"Text for {snippet_id}")
        doc._.snippet_id = snippet_id
        doc._.date = date
        doc.cats = cats
        db.add(doc)
    db.to_disk(path)


@pytest.fixture
def eval_files(tmp_path):
    nlp = spacy.blank("en")
    sentiment_eval = tmp_path / "eval_sentiment.spacy"
    relevance_eval = tmp_path / "eval_relevance.spacy"
    _write_docbin(
        sentiment_eval,
        nlp,
        [
            (
                "pos",
                "1975-01-01",
                {
                    "POSITIVE_TOWARDS_IMMIGRATION": 1.0,
                    "NEUTRAL_TOWARDS_IMMIGRATION": 0.0,
                    "NEGATIVE_TOWARDS_IMMIGRATION": 0.0,
                },
            ),
            (
                "neg",
                "2001-01-01",
                {
                    "POSITIVE_TOWARDS_IMMIGRATION": 0.0,
                    "NEUTRAL_TOWARDS_IMMIGRATION": 0.0,
                    "NEGATIVE_TOWARDS_IMMIGRATION": 1.0,
                },
            ),
        ],
    )
    _write_docbin(
        relevance_eval,
        nlp,
        [
            (
                "pos",
                "1975-01-01",
                {"ABOUT_IMMIGRATION": 1.0, "NOT_ABOUT_IMMIGRATION": 0.0},
            ),
            (
                "neg",
                "2001-01-01",
                {"ABOUT_IMMIGRATION": 1.0, "NOT_ABOUT_IMMIGRATION": 0.0},
            ),
            (
                "off",
                "2022-01-01",
                {"ABOUT_IMMIGRATION": 0.0, "NOT_ABOUT_IMMIGRATION": 1.0},
            ),
        ],
    )
    return nlp.vocab, relevance_eval, sentiment_eval


def _fake_models(vocab, relevance_preds, sentiment_preds):
    return [FakeModel(vocab, relevance_preds), FakeModel(vocab, sentiment_preds)]


PERFECT_RELEVANCE = {
    "pos": {"ABOUT_IMMIGRATION": 0.9, "NOT_ABOUT_IMMIGRATION": 0.1},
    "neg": {"ABOUT_IMMIGRATION": 0.9, "NOT_ABOUT_IMMIGRATION": 0.1},
    "off": {"ABOUT_IMMIGRATION": 0.1, "NOT_ABOUT_IMMIGRATION": 0.9},
}
PERFECT_SENTIMENT = {
    "pos": {
        "POSITIVE_TOWARDS_IMMIGRATION": 0.8,
        "NEUTRAL_TOWARDS_IMMIGRATION": 0.1,
        "NEGATIVE_TOWARDS_IMMIGRATION": 0.1,
    },
    "neg": {
        "POSITIVE_TOWARDS_IMMIGRATION": 0.1,
        "NEUTRAL_TOWARDS_IMMIGRATION": 0.1,
        "NEGATIVE_TOWARDS_IMMIGRATION": 0.8,
    },
    "off": {
        "POSITIVE_TOWARDS_IMMIGRATION": 0.8,
        "NEUTRAL_TOWARDS_IMMIGRATION": 0.1,
        "NEGATIVE_TOWARDS_IMMIGRATION": 0.1,
    },
}


def _run(tmp_path, eval_files, relevance_preds, sentiment_preds, plot=None):
    vocab, relevance_eval, sentiment_eval = eval_files
    metrics_path = tmp_path / "metrics.json"
    predictions_path = tmp_path / "predictions.jsonl"
    with patch(
        f"{MODULE}.spacy.load",
        side_effect=_fake_models(vocab, relevance_preds, sentiment_preds),
    ):
        ModelEvaluator().evaluate_models(
            relevance_model=tmp_path / "relevance",
            sentiment_model=tmp_path / "sentiment",
            relevance_eval=relevance_eval,
            sentiment_eval=sentiment_eval,
            output_path_metrics=metrics_path,
            output_path_predictions=predictions_path,
            plot=plot,
        )
    return json.loads(metrics_path.read_text()), pd.read_json(
        predictions_path, lines=True
    )


class TestEvaluateModels:
    def test_combines_sentiment_docs_with_not_about_relevance_docs(
        self, tmp_path, eval_files
    ):
        _, predictions = _run(
            tmp_path, eval_files, PERFECT_RELEVANCE, PERFECT_SENTIMENT
        )

        assert predictions["true_combined_label"].tolist() == [
            "POSITIVE_TOWARDS_IMMIGRATION",
            "NEGATIVE_TOWARDS_IMMIGRATION",
            "NOT_ABOUT_IMMIGRATION",
        ]

    def test_perfect_predictions_score_perfectly(self, tmp_path, eval_files):
        metrics, predictions = _run(
            tmp_path, eval_files, PERFECT_RELEVANCE, PERFECT_SENTIMENT
        )

        assert (
            predictions["predicted_combined_label"]
            == predictions["true_combined_label"]
        ).all()
        assert metrics["cats_micro_f"] == pytest.approx(1.0)

    def test_sentiment_ignored_when_predicted_not_about(self, tmp_path, eval_files):
        relevance = dict(
            PERFECT_RELEVANCE,
            pos={"ABOUT_IMMIGRATION": 0.2, "NOT_ABOUT_IMMIGRATION": 0.8},
        )

        _, predictions = _run(tmp_path, eval_files, relevance, PERFECT_SENTIMENT)

        pos_row = predictions.iloc[0]
        assert pos_row["predicted_combined_label"] == "NOT_ABOUT_IMMIGRATION"
        assert (
            pos_row["predicted_combined_probs"]["POSITIVE_TOWARDS_IMMIGRATION"] == 0.0
        )

    def test_writes_overall_and_per_period_plots(self, tmp_path, eval_files):
        plot = tmp_path / "confusion_matrix.png"

        _run(tmp_path, eval_files, PERFECT_RELEVANCE, PERFECT_SENTIMENT, plot=plot)

        assert plot.exists()
        for period in ["before_1980", "1980_to_2020", "2021_onwards"]:
            assert (tmp_path / f"confusion_matrix_{period}.png").exists()


class TestPlotConfusionMatrix:
    def test_saves_png(self, tmp_path):
        save_path = tmp_path / "cm.png"

        ModelEvaluator().plot_confusion_matrix_with_style(
            [[1, 0], [0, 1]], ["A", "B"], save_path=save_path
        )

        assert save_path.exists()
