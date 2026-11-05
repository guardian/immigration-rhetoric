import json
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import requests
from immigration_rhetoric.stage02_speaker_metadata.download_speaker_metadata import (
    SpeakerMetadataDownloader,
    WikidataQuerier,
)


@pytest.fixture
def mock_response():
    """Mock requests.Response object"""
    mock = Mock()
    mock.status_code = 200
    mock.json.return_value = {"results": {"bindings": []}}
    return mock


@pytest.fixture
def temp_metadata_path(tmp_path):
    """Create a temporary directory for metadata"""
    metadata_path = tmp_path / "speaker_metadata"
    metadata_path.mkdir()
    return metadata_path


class TestWikidataQuerier:
    def test_query_wikidata_success(self, mock_response):
        with patch("requests.get", return_value=mock_response):
            result = WikidataQuerier.query_wikidata("test query")
            assert result == {"results": {"bindings": []}}

    def test_query_wikidata_failure(self):
        mock_failed_response = Mock()
        mock_failed_response.status_code = 404
        mock_failed_response.text = "Not Found"

        with patch("requests.get", return_value=mock_failed_response):
            with pytest.raises(Exception) as exc_info:
                WikidataQuerier.query_wikidata("test query")
            assert "Query failed with status code 404" in str(exc_info.value)

    def test_query_wikidata_retries_incomplete_response(self, mock_response):
        truncated = Mock()
        truncated.status_code = 200
        truncated.json.side_effect = requests.exceptions.JSONDecodeError(
            "Unterminated string", '{"results": "abc', 15
        )

        with patch("requests.get", side_effect=[truncated, mock_response]) as get:
            with patch("time.sleep"):
                result = WikidataQuerier.query_wikidata("test query")

        assert result == {"results": {"bindings": []}}
        assert get.call_count == 2

    def test_query_wikidata_raises_if_response_stays_incomplete(self):
        truncated = Mock()
        truncated.status_code = 200
        truncated.json.side_effect = requests.exceptions.JSONDecodeError(
            "Unterminated string", '{"results": "abc', 15
        )

        with patch("requests.get", return_value=truncated) as get:
            with patch("time.sleep"):
                with pytest.raises(requests.exceptions.JSONDecodeError):
                    WikidataQuerier.query_wikidata("test query", max_attempts=3)

        assert get.call_count == 3

    def test_generate_all_parliament_terms_query(self):
        query = WikidataQuerier.generate_all_parliament_terms_query()
        assert isinstance(query, str)
        assert "SELECT" in query
        assert "wd:Q4164871" in query  # Check for INSTANCE_OF_TERM
        assert "wd:Q16707842" in query  # Check for SUBCLASS_OF_TAXON

    def test_generate_single_parliament_term_query(self):
        term_id = "Q123"
        query = WikidataQuerier.generate_single_parliament_term_query(term_id)
        assert isinstance(query, str)
        assert "SELECT" in query
        assert term_id in query


class TestSpeakerMetadataDownloader:
    @pytest.fixture
    def collector(self, temp_metadata_path):
        return SpeakerMetadataDownloader(temp_metadata_path)

    def test_init(self, collector, temp_metadata_path):
        assert collector.output_path == temp_metadata_path
        assert not collector.force_refresh

    def test_download_people_json_new(self, collector, mock_response):
        mock_response.json.return_value = {"test": "data"}
        mock_response.iter_content.return_value = [
            json.dumps({"test": "data"}).encode()
        ]

        with patch("requests.get", return_value=mock_response):
            collector.download_people_json()

        people_file = collector.output_path / "people.json"
        assert people_file.exists()
        assert json.loads(people_file.read_text()) == {"test": "data"}

    def test_download_people_json_cached(self, collector):
        # Create cached file
        people_file = collector.output_path / "people.json"
        people_file.write_text('{"cached": "data"}')

        # Should use cached file
        collector.download_people_json()

        assert json.loads(people_file.read_text()) == {"cached": "data"}

    def test_get_uk_parliament_term_ids(self, collector, mock_response):
        mock_response.json.return_value = {
            "results": {
                "bindings": [
                    {
                        "item": {"value": "http://test/Q123"},
                        "itemLabel": {"value": "57th Parliament"},
                        "inception": {"value": "2019-12-12"},
                        "dissolved": {"value": "2024-01-01"},
                    }
                ]
            }
        }

        with patch.object(
            WikidataQuerier, "query_wikidata", return_value=mock_response.json()
        ):
            result = collector.get_uk_parliament_term_ids()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]["term_wikiID"] == "Q123"
        assert result.iloc[0]["term_numeric"] == "57"
