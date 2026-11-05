import json

import pandas as pd
import pytest
from immigration_rhetoric.stage08_inference.combine_inference_outputs import (
    InferenceOutputCombiner,
)

POS = "POSITIVE_TOWARDS_IMMIGRATION"
NEU = "NEUTRAL_TOWARDS_IMMIGRATION"
NEG = "NEGATIVE_TOWARDS_IMMIGRATION"
NOT = "NOT_ABOUT_IMMIGRATION"


def labelled_snippets(rows):
    """Rows of (speech_id, snippet_id, date, label), with the other columns run_inference writes."""
    return pd.DataFrame(
        [
            {
                "id": speech_id,
                "text": f"text of {speech_id}",
                "constituencyLabel": "Somewhere",
                "partyLabel": "A Party",
                "personLabel": "An MP",
                "date": date,
                "snippet_id": snippet_id,
                "snippet": f"snippet {snippet_id}",
                "relevance_label": NOT if label == NOT else "ABOUT_IMMIGRATION",
                "relevance_label_scores": json.dumps({"NOT_ABOUT_IMMIGRATION": 0.5}),
                "sentiment_label": None if label == NOT else label,
                "sentiment_label_scores": None if label == NOT else "{}",
                "label": label,
            }
            for speech_id, snippet_id, date, label in rows
        ]
    )


def write(df, path):
    df.to_json(path, orient="records", date_format="iso", lines=True)


class TestSentimentScore:
    @pytest.mark.parametrize(
        "labels, expected",
        [
            ([POS, NEG, NEU, NOT], 0.0),
            ([POS, POS, NEU, NOT], 2 / 3),
            ([NEG, NOT], -1.0),
            ([NOT, NOT], 0),
        ],
    )
    def test_positive_minus_negative_over_about(self, labels, expected):
        assert InferenceOutputCombiner.sentiment_score(
            pd.Series(labels)
        ) == pytest.approx(expected)


class TestLoadLabelledSnippets:
    def test_formats_date_and_renames_columns(self, tmp_path):
        path = tmp_path / "labelled_snippets_1950.jsonl"
        write(labelled_snippets([("sp1", "sp1_1", "1950-03-06", NOT)]), path)

        df = InferenceOutputCombiner.load_labelled_snippets(path)

        assert df.loc[0, "date"] == "1950-03-06"
        assert {"speech_id", "speech_text", "constituency_label", "party_label"} <= set(
            df.columns
        )
        assert {"person_label", "snippet_text"} <= set(df.columns)

    def test_strips_time_from_datetime_formatted_date(self, tmp_path):
        path = tmp_path / "labelled_snippets_1950.jsonl"
        write(
            labelled_snippets([("sp1", "sp1_1", "1950-03-06T00:00:00.000", NOT)]), path
        )

        df = InferenceOutputCombiner.load_labelled_snippets(path)

        assert df.loc[0, "date"] == "1950-03-06"

    def test_drops_snippets_without_a_speaker(self, tmp_path):
        path = tmp_path / "labelled_snippets_2020.jsonl"
        df = labelled_snippets(
            [("sp1", "sp1_1", "2020-01-01", NOT), ("sp2", "sp2_1", "2020-01-01", NOT)]
        )
        df.loc[1, "personLabel"] = None
        write(df, path)

        result = InferenceOutputCombiner.load_labelled_snippets(path)

        assert result["snippet_id"].to_list() == ["sp1_1"]


class TestSummariseSpeeches:
    def test_one_row_per_speech_with_counts(self, tmp_path):
        path = tmp_path / "labelled_snippets_2020.jsonl"
        rows = [
            ("sp1", "sp1_1", "2020-01-01", POS),
            ("sp1", "sp1_2", "2020-01-01", NOT),
            ("sp1", "sp1_3", "2020-01-01", NEG),
            ("sp1", "sp1_4", "2020-01-01", POS),
            ("sp2", "sp2_1", "2020-01-02", NOT),
        ]
        write(labelled_snippets(rows), path)
        df = InferenceOutputCombiner.load_labelled_snippets(path)

        speeches = InferenceOutputCombiner.summarise_speeches(df).set_index("speech_id")

        assert speeches.loc["sp1", "snippet_ids"] == [
            "sp1_1",
            "sp1_2",
            "sp1_3",
            "sp1_4",
        ]
        assert speeches.loc[
            "sp1", "sentiment_score_positive_minus_negative"
        ] == pytest.approx(1 / 3)
        assert speeches.loc["sp1", "n_snippets_positive"] == 2
        assert speeches.loc["sp1", "n_snippets_neutral"] == 0
        assert speeches.loc["sp1", "n_snippets_total"] == 4
        assert speeches.loc["sp1", "n_snippets_about"] == 3
        assert speeches.loc["sp2", "n_snippets_about"] == 0
        assert speeches.loc["sp2", "sentiment_score_positive_minus_negative"] == 0

    def test_keeps_speeches_with_a_speaker_but_no_party(self, tmp_path):
        path = tmp_path / "labelled_snippets_2020.jsonl"
        df = labelled_snippets([("sp1", "sp1_1", "2020-01-01", NOT)])
        df["partyLabel"] = None
        write(df, path)

        speeches = InferenceOutputCombiner.summarise_speeches(
            InferenceOutputCombiner.load_labelled_snippets(path)
        )

        assert speeches["speech_id"].to_list() == ["sp1"]
        assert speeches.loc[0, "n_snippets_total"] == 1


class TestCombine:
    def test_combines_years_with_expected_columns(self, tmp_path):
        write(
            labelled_snippets([("a", "a_1", "2019-05-01", POS)]),
            tmp_path / "labelled_snippets_2019.jsonl",
        )
        write(
            labelled_snippets(
                [("b", "b_1", "2020-05-01", NOT), ("b", "b_2", "2020-05-01", NEU)]
            ),
            tmp_path / "labelled_snippets_2020.jsonl",
        )

        speeches, snippets = InferenceOutputCombiner().combine(tmp_path)

        assert speeches.columns.to_list() == [
            "speech_id",
            "speech_text",
            "constituency_label",
            "party_label",
            "person_label",
            "date",
            "snippet_ids",
            "sentiment_score_positive_minus_negative",
            "n_snippets_negative",
            "n_snippets_neutral",
            "n_snippets_not_about",
            "n_snippets_positive",
            "n_snippets_total",
            "n_snippets_about",
        ]
        assert speeches["speech_id"].to_list() == ["a", "b"]
        assert speeches["n_snippets_not_about"].to_list() == [0, 1]
        assert snippets.columns.to_list() == [
            "snippet_id",
            "snippet_text",
            "speech_id",
            "date",
            "party_label",
            "person_label",
            "relevance_label",
            "relevance_label_scores",
            "sentiment_label",
            "sentiment_label_scores",
            "label",
        ]
        assert snippets["snippet_id"].to_list() == ["a_1", "b_1", "b_2"]
