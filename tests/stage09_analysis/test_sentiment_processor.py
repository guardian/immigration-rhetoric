import pandas as pd
import pytest
from immigration_rhetoric.stage09_analysis.sentiment_calculator import (
    SentimentCalculator,
)
from immigration_rhetoric.stage09_analysis.sentiment_processor import (
    SentimentDataProcessor,
)
from tests.stage09_analysis.conftest import (
    NEG,
    NEU,
    NOT,
    POS,
    TERMS,
    identity_corrections,
)


@pytest.fixture
def negatives_are_neutral():
    """Corrections under which every predicted negative snippet is truly neutral."""
    corrections = identity_corrections()
    corrections[NEG] = {POS: 0.0, NEU: 1.0, NEG: 0.0, NOT: 0.0}
    return SentimentDataProcessor(
        calculator=SentimentCalculator(corrections=corrections)
    )


class TestPrepareDataframe:
    def test_adds_year(self):
        df = pd.DataFrame(
            {"date": ["2001-03-15"], "label": [POS], "party_label": ["Labour Party"]}
        )

        result = SentimentDataProcessor()._prepare_dataframe(df)

        assert result["year"].iloc[0] == pd.Timestamp("2001-01-01")

    def test_maps_party_names_and_masks_parties_outside_their_windows(self):
        df = pd.DataFrame(
            {
                "date": [
                    "1990-01-01",
                    "2000-01-01",
                    "1990-01-01",
                    "2000-01-01",
                    "2020-01-01",
                    "1990-01-01",
                ],
                "label": [POS] * 6,
                "party_label": [
                    "Scottish National Party",
                    "Scottish National Party",
                    "UK Independence Party",
                    "UK Independence Party",
                    "UK Independence Party",
                    "Labour Co-operative",
                ],
            }
        )

        result = SentimentDataProcessor()._prepare_dataframe(df)

        assert result["party_label"].isna().tolist() == [
            True,
            False,
            True,
            False,
            True,
            False,
        ]
        assert result["party_label"].dropna().tolist() == [
            "SNP",
            "UKIP",
            "Labour Party",
        ]

    def test_empty_dataframe_raises(self):
        with pytest.raises(ValueError, match="empty"):
            SentimentDataProcessor()._prepare_dataframe(pd.DataFrame())


class TestSentimentByGroup:
    def test_time_groups_are_corrected(self, snippets, negatives_are_neutral):
        result = negatives_are_neutral.get_party_sentiment_over_time(snippets)

        # Each year: 1 positive, 3 neutral (after correction), 0 negative
        assert result["Overall"]["sentiment"].tolist() == pytest.approx([25.0] * 4)

    def test_uncorrected_on_request(self, snippets, negatives_are_neutral):
        result = negatives_are_neutral.get_party_sentiment_over_time(
            snippets, include_uncorrected=True
        )

        assert result["Overall"]["sentiment"].tolist() == pytest.approx([-25.0] * 4)

    def test_trigger_groups_are_not_corrected(self, snippets, negatives_are_neutral):
        result = negatives_are_neutral.sentiment_per_trigger(
            snippets, trigger_terms=TERMS, min=1
        )

        assert result["sentiment"].tolist() == pytest.approx([-25.0] * 3)

    def test_party_data_only_for_parties_with_snippets(self, snippets, corrections):
        processor = SentimentDataProcessor(
            calculator=SentimentCalculator(corrections=corrections)
        )

        result = processor.get_party_sentiment_over_time(
            snippets, parties_of_interest=["Labour Party", "Green Party"]
        )

        assert set(result) == {"Overall", "Labour Party"}
        assert len(result["Labour Party"]) == 4


class TestSentimentPerTrigger:
    def test_keeps_terms_in_at_least_min_snippets(self):
        df = pd.DataFrame(
            {
                "label": [POS, NEG, NEG, POS],
                "snippet_text": ["Migrants", "migrant", "MIGRANT here", "a visa"],
            }
        )

        result = SentimentDataProcessor().sentiment_per_trigger(
            df, trigger_terms=["migrant", "visa"], min=2
        )

        assert result["trigger"].tolist() == ["migrant"]
        assert result["sentiment"].iloc[0] == pytest.approx(-100)


class TestPrevalence:
    def test_counts_speeches_about_immigration_out_of_all_speeches(self):
        trigger_speeches = pd.DataFrame(
            {
                "speech_id": ["a", "b", "c"],
                "date": ["2000-01-01", "2000-06-01", "2001-01-01"],
                "party_label": ["Labour Party", "Labour Party", "Conservative Party"],
                "n_snippets_about": [1, 0, 2],
            }
        )
        all_speeches = pd.DataFrame(
            {
                "speech_id": ["a", "b", "c", "d", "e", "f"],
                "date": ["2000-01-01", "2000-06-01", "2001-01-01"] * 2,
                "party_label": ["Labour Party", "Labour Party", "Conservative Party"]
                * 2,
            }
        )

        result = SentimentDataProcessor().get_prevalence_over_time(
            trigger_speeches, all_speeches, parties_of_interest=["Labour Party"]
        )

        overall = result["Overall"]
        assert overall["about_count"].tolist() == [1, 1]
        assert overall["total_count"].tolist() == [4, 2]
        # Labour has no speeches in 2001, so gets zeros
        assert result["Labour Party"]["total_count"].tolist() == [4, 0]


class TestTermSensitivity:
    @pytest.fixture
    def df(self):
        return pd.DataFrame(
            {
                "speech_id": ["s1", "s1", "s2", "s3"],
                "snippet_id": ["a", "b", "c", "d"],
                "snippet_text": ["migrant", "asylum", "refugee", "migrant refugee"],
            }
        )

    def test_one_run_per_group_of_dropped_terms(self, df):
        results = SentimentDataProcessor().calculate_term_sensitivity(
            df, all_terms=TERMS, drop_percentage=10
        )

        assert list(results) == ["baseline", "run_1", "run_2", "run_3"]
        assert len(results["baseline"]) == 4
        # A speech is dropped only if none of its snippets has a remaining term, so only dropping
        # "refugee" (s2's only term) drops a speech
        kept = sorted(
            sorted(set(results[run]["speech_id"]))
            for run in ["run_1", "run_2", "run_3"]
        )
        assert kept == [["s1", "s2", "s3"], ["s1", "s2", "s3"], ["s1", "s3"]]

    def test_caches_term_presence_in_cache_dir(self, df, tmp_path):
        processor = SentimentDataProcessor(cache_dir=tmp_path / "cache")

        processor.calculate_term_sensitivity(df, all_terms=TERMS)

        assert len(list((tmp_path / "cache").glob("presence_matrix_*.pkl"))) == 1

    def test_cache_is_per_set_of_snippets(self, df, tmp_path):
        processor = SentimentDataProcessor(cache_dir=tmp_path / "cache")
        other = df.assign(
            speech_id=["t1", "t2", "t3", "t4"], snippet_id=["e", "f", "g", "h"]
        )

        processor.calculate_term_sensitivity(df, all_terms=TERMS)
        results = processor.calculate_term_sensitivity(other, all_terms=TERMS)

        # Each run drops one term, so one single-snippet speech
        assert len(results["run_1"]) == 3
