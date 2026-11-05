import json
from types import SimpleNamespace

import pandas as pd
from immigration_rhetoric.stage08_inference.classify_snippets import SnippetClassifier

RELEVANCE_CATS = {
    "about": {"ABOUT_IMMIGRATION": 0.9, "NOT_ABOUT_IMMIGRATION": 0.1},
    "not about": {"ABOUT_IMMIGRATION": 0.2, "NOT_ABOUT_IMMIGRATION": 0.8},
}

SENTIMENT_CATS = {
    "about": {
        "POSITIVE_TOWARDS_IMMIGRATION": 0.1,
        "NEUTRAL_TOWARDS_IMMIGRATION": 0.2,
        "NEGATIVE_TOWARDS_IMMIGRATION": 0.7,
    },
}


class FakeNlp:
    """Stands in for a spaCy pipeline, returning fixed cats for each text and recording what it was given."""

    def __init__(self, cats: dict):
        self.cats = cats
        self.calls = []

    def pipe(self, texts, n_process=1):
        texts = list(texts)
        self.calls.append((texts, n_process))
        return (SimpleNamespace(cats=self.cats[text]) for text in texts)


def make_classifier(n_process=1):
    return SnippetClassifier(
        FakeNlp(RELEVANCE_CATS), FakeNlp(SENTIMENT_CATS), n_process
    )


class TestPredict:
    def test_returns_top_label_and_json_scores(self):
        classifier = make_classifier()
        labels, scores = classifier.predict(
            classifier.relevance_nlp, ["about", "not about"], "test"
        )

        assert labels == ["ABOUT_IMMIGRATION", "NOT_ABOUT_IMMIGRATION"]
        assert [json.loads(s) for s in scores] == [
            RELEVANCE_CATS["about"],
            RELEVANCE_CATS["not about"],
        ]

    def test_ties_go_to_first_label(self):
        nlp = FakeNlp({"tie": {"A": 0.5, "B": 0.5}})
        labels, _ = make_classifier().predict(nlp, ["tie"], "test")

        assert labels == ["A"]

    def test_passes_n_process_to_pipe(self):
        classifier = make_classifier(n_process=3)
        classifier.predict(classifier.relevance_nlp, ["about"], "test")

        assert classifier.relevance_nlp.calls[0][1] == 3


class TestClassify:
    def test_adds_relevance_sentiment_and_final_label(self):
        df = pd.DataFrame(
            {
                "snippet_id": ["s1", "s2", "s3"],
                "snippet": ["about", "not about", "about"],
            }
        )
        result = make_classifier().classify(df)

        assert result["relevance_label"].to_list() == [
            "ABOUT_IMMIGRATION",
            "NOT_ABOUT_IMMIGRATION",
            "ABOUT_IMMIGRATION",
        ]
        assert result["label"].to_list() == [
            "NEGATIVE_TOWARDS_IMMIGRATION",
            "NOT_ABOUT_IMMIGRATION",
            "NEGATIVE_TOWARDS_IMMIGRATION",
        ]
        assert result.loc[0, "sentiment_label"] == "NEGATIVE_TOWARDS_IMMIGRATION"
        assert (
            json.loads(result.loc[2, "sentiment_label_scores"])
            == SENTIMENT_CATS["about"]
        )
        assert pd.isna(result.loc[1, "sentiment_label"])
        assert pd.isna(result.loc[1, "sentiment_label_scores"])
        assert result["snippet_id"].to_list() == ["s1", "s2", "s3"]

    def test_only_relevant_snippets_go_to_sentiment_model(self):
        classifier = make_classifier()
        classifier.classify(pd.DataFrame({"snippet": ["not about", "about"]}))

        assert classifier.sentiment_nlp.calls == [(["about"], 1)]

    def test_no_relevant_snippets_still_has_all_columns(self):
        result = make_classifier().classify(pd.DataFrame({"snippet": ["not about"]}))

        assert result["sentiment_label"].isna().all()
        assert result["sentiment_label_scores"].isna().all()
        assert result["label"].to_list() == ["NOT_ABOUT_IMMIGRATION"]

    def test_custom_snippet_column(self):
        result = make_classifier().classify(
            pd.DataFrame({"text": ["about"]}), snippet_col="text"
        )

        assert result["label"].to_list() == ["NEGATIVE_TOWARDS_IMMIGRATION"]

    def test_does_not_modify_input(self):
        df = pd.DataFrame({"snippet": ["about"]})
        make_classifier().classify(df)

        assert df.columns.to_list() == ["snippet"]
