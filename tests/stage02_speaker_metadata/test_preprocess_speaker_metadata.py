import json

import pandas as pd
import pytest
from immigration_rhetoric.stage02_speaker_metadata.preprocess_speaker_metadata import (
    SpeakerMetadataPreprocessor,
)


@pytest.fixture
def preprocessor(tmp_path):
    return SpeakerMetadataPreprocessor(input_dir=tmp_path)


class TestSpeakerMetadataPreprocessor:
    def test_init(self, preprocessor, tmp_path):
        assert preprocessor.input_dir == tmp_path

    def test_load_people_json(self, preprocessor, tmp_path):
        data = {
            "persons": [{"id": "uk.org.publicwhip/person/1"}],
            "memberships": [{"id": "uk.org.publicwhip/member/1"}],
            "posts": [{"id": "uk.org.publicwhip/post/1"}],
        }
        (tmp_path / "people.json").write_text(json.dumps(data))

        people, memberships, posts = preprocessor.load_people_json()

        assert len(people) == 1
        assert len(memberships) == 1
        assert len(posts) == 1

    def test_fill_redirected_ids(self, preprocessor):
        data = {
            "id": ["1", "2", "3"],
            "person_id": ["101", None, "103"],
            "redirect": [None, "3", None],
            "other_names": [["John"], None, ["Jane"]],
        }
        df = pd.DataFrame(data)

        result = preprocessor.fill_redirected_ids(df)

        assert len(result) == 3
        redirected_row = result[result["id"] == "2"].iloc[0]
        assert redirected_row["person_id"] == "103"
        assert redirected_row["other_names"] == ["Jane"]

    def test_load_wikidata(self, preprocessor, tmp_path):
        wikidata = pd.DataFrame({"theyWorkForYouID": ["1234"]})
        wikidata.to_csv(tmp_path / "mps_wikidata.csv", index=False)

        result = preprocessor.load_wikidata()

        assert list(result["theyWorkForYouID"]) == [1234]

    def test_preprocess_wikidata(self, preprocessor):
        wikidata = pd.DataFrame(
            {
                "theyWorkForYouID": ["1234", "5678", None],
                "partyStartDate": [
                    "2020-01-01T00:00:00Z",
                    "2020-06-15T00:00:00Z",
                    "2020-01-01T00:00:00Z",
                ],
                "partyEndDate": ["2021-01-01T00:00:00Z", None, "2021-01-01T00:00:00Z"],
            }
        )

        result = preprocessor.preprocess_wikidata(wikidata)

        assert len(result) == 2  # Should drop row with None theyWorkForYouID
        assert all(isinstance(id_, int) for id_ in result["theyWorkForYouID"])
        assert all(pd.notnull(result["partyEndDate"]))  # All end dates should be filled

    def test_preprocess_memberships(self, preprocessor):
        memberships = pd.DataFrame(
            {
                "id": [
                    "uk.org.publicwhip/member/1",
                    "uk.org.publicwhip/member/2",
                ],
                "person_id": [
                    "uk.org.publicwhip/person/10001",
                    "uk.org.publicwhip/person/10002",
                ],
                "redirect": [None, None],
            }
        )

        result = preprocessor.preprocess_memberships(memberships)

        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"id", "person_id"}
        assert list(result["id"]) == [1, 2]
        assert list(result["person_id"]) == [10001, 10002]
