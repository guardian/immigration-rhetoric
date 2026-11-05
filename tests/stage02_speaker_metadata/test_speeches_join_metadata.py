import pandas as pd
import pytest
from immigration_rhetoric.stage02_speaker_metadata.speeches_join_metadata import (
    SpeechMetadataJoiner,
)


@pytest.fixture
def joiner(tmp_path):
    return SpeechMetadataJoiner(
        raw_speeches_dir=tmp_path / "raw_speeches", output_dir=tmp_path / "speeches"
    )


@pytest.fixture
def wikidata():
    return pd.DataFrame(
        {
            "person": ["http://wikidata/Q1", "http://wikidata/Q2"],
            "personLabel": ["Jane Smith", "John Doe"],
            "altLabel": [None, None],
            "theyWorkForYouID": [10001, 10002],
            "partyLabel": ["Party A", "Party B"],
            "partyStartDate": pd.to_datetime(["2010-01-01", "2010-01-01"]),
            "partyEndDate": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "constituencyLabel": ["Constituency A", "Constituency B"],
            "parliament_term": ["55", "55"],
            "parliament_term_id": ["Q55", "Q55"],
            "mp_wikiID": ["Q1", "Q2"],
        }
    )


class TestSplitSpeechesIntoJoinStages:
    def test_splits_by_available_identifiers(self, joiner):
        speeches = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "person_id": [10001, None, None],
                "membership_id": [None, 55, None],
                "speakername": ["Jane Smith", "John Doe", "A Member"],
            }
        )

        stage1, stage2, stage3 = joiner.split_speeches_into_join_stages(speeches)

        assert list(stage1.id) == [1]
        assert list(stage2.id) == [2]
        assert list(stage3.id) == [3]

    def test_handles_missing_person_id_column_entirely(self, joiner):
        speeches = pd.DataFrame(
            {
                "id": [1, 2],
                "membership_id": [55, None],
                "speakername": ["Jane Smith", "A Member"],
            }
        )

        stage1, stage2, stage3 = joiner.split_speeches_into_join_stages(speeches)

        assert len(stage1) == 0
        assert list(stage2.id) == [1]
        assert list(stage3.id) == [2]


class TestJoinSpeechesOnPersonId:
    def test_matches_person_id_to_wikidata(self, joiner, wikidata):
        speeches = pd.DataFrame(
            {
                "id": [1, 2],
                "person_id": [10001, 99999],
                "date": pd.to_datetime(["2015-01-01", "2015-01-01"]),
                "year": [2015, 2015],
            }
        )

        result = joiner.join_speeches_on_person_id(speeches, wikidata)

        assert len(result) == 1
        assert result.iloc[0]["id"] == 1
        assert result.iloc[0]["partyLabel"] == "Party A"

    def test_drops_matches_outside_party_date_range(self, joiner, wikidata):
        speeches = pd.DataFrame(
            {
                "id": [1],
                "person_id": [10001],
                "date": pd.to_datetime(["2025-01-01"]),
                "year": [2025],
            }
        )

        result = joiner.join_speeches_on_person_id(speeches, wikidata)

        assert len(result) == 0


class TestJoinSpeechesOnMembershipId:
    """Regression tests for defensive handling when person_id is entirely absent from the input speeches."""

    @pytest.fixture
    def memberships(self):
        return pd.DataFrame({"id": [55, 56], "person_id": [10001, 10002]})

    def test_joins_when_speeches_have_no_person_id_column(
        self, joiner, memberships, wikidata
    ):
        speeches = pd.DataFrame(
            {
                "id": [1],
                "membership_id": [55],
                "date": pd.to_datetime(["2015-01-01"]),
                "year": [2015],
            }
        )

        result = joiner.join_speeches_on_membership_id(speeches, memberships, wikidata)

        assert len(result) == 1
        assert result.iloc[0]["partyLabel"] == "Party A"

    def test_joins_when_speeches_have_a_stale_person_id_column(
        self, joiner, memberships, wikidata
    ):
        speeches = pd.DataFrame(
            {
                "id": [1],
                "membership_id": [55],
                "person_id": [None],
                "date": pd.to_datetime(["2015-01-01"]),
                "year": [2015],
            }
        )

        result = joiner.join_speeches_on_membership_id(speeches, memberships, wikidata)

        assert len(result) == 1
        assert result.iloc[0]["partyLabel"] == "Party A"


class TestCleanName:
    def test_removes_titles_and_brackets(self, joiner):
        assert (
            joiner.clean_name("Mr. Patrick Cormack (Staffordshire)")
            == "Patrick Cormack"
        )

    def test_extracts_name_from_speaker_brackets(self, joiner):
        assert joiner.clean_name("Mr Speaker (Jane Smith)") == "Jane Smith"


class TestSaveJoinedSpeeches:
    def test_writes_one_csv_per_year(self, joiner, tmp_path):
        joined_speeches = pd.DataFrame(
            {"id": [1, 2], "year": [2015, 2016], "text": ["a", "b"]}
        )

        joiner.save_joined_speeches(joined_speeches)

        assert (tmp_path / "speeches" / "speeches_2015.csv").exists()
        assert (tmp_path / "speeches" / "speeches_2016.csv").exists()
