from datetime import datetime
from unittest.mock import Mock

import pandas as pd
import pytest
from immigration_rhetoric.stage01_scrape_speeches.process_speeches import (
    SpeechesProcessor,
)


@pytest.fixture
def mock_args():
    args = Mock()
    args.start_date = datetime(2020, 1, 1)
    args.end_date = datetime(2020, 12, 31)
    return args


@pytest.fixture
def processor(tmp_path, mock_args):
    return SpeechesProcessor(
        xml_path=tmp_path / "xml_files",
        output_path=tmp_path / "raw_speeches",
        start_date=mock_args.start_date,
        end_date=mock_args.end_date,
    )


@pytest.fixture
def sample_xml():
    return """
    <publicwhip>
        <speech id="2020-01-01.1" speakername="John Smith" person_id="1234">
            <p>Test speech content</p>
        </speech>
        <speech id="2020-01-01.2" speakername="Jane Doe" person_id="5678">
            <p>Another speech</p>
        </speech>
    </publicwhip>
    """


class TestSpeechesProcessor:
    def test_init(self, processor, tmp_path, mock_args):
        assert processor.output_path == tmp_path / "raw_speeches"
        assert processor.xml_path == tmp_path / "xml_files"
        assert processor.start_date == mock_args.start_date
        assert processor.end_date == mock_args.end_date

    def test_get_speech_fields(self, processor, sample_xml):
        fields = processor._get_speech_fields([sample_xml])
        assert "id" in fields
        assert "speakername" in fields
        assert "person_id" in fields

    def test_extract_speeches_from_single_file(self, processor, sample_xml):
        fields = {"id", "speakername", "person_id"}
        speeches = processor._extract_speeches_from_single_file(sample_xml, fields)

        assert len(speeches) == 2
        assert speeches[0]["speakername"] == "John Smith"
        assert speeches[0]["person_id"] == "1234"
        assert "Test speech content" in speeches[0]["text"]

    def test_remove_unidentified_speeches(self, processor):
        data = {
            "person_id": ["1234", None, None],
            "speakerid": [None, "5678", None],
            "speakername": ["John Smith", "Jane Doe", None],
            "text": ["Speech 1", "Speech 2", "Speech 3"],
        }
        df = pd.DataFrame(data)

        filtered_df = processor._remove_unidentified_speeches(df)

        assert (
            len(filtered_df) == 2
        )  # Only rows with some form of identification remain
        assert "1234" in filtered_df["person_id"].values
        assert "5678" in filtered_df["speakerid"].values

    def test_remove_invalid_speeches(self, processor):
        data = {
            "speakername": ["John Smith", "Several hon. Members", "Jane Doe"],
            "redirect": [None, None, "something"],
            "text": ["Speech 1", "Speech 2", "Speech 3"],
        }
        df = pd.DataFrame(data)

        filtered_df = processor._remove_invalid_speeches(df)

        assert len(filtered_df) == 1  # Only one valid row should remain
        assert filtered_df.iloc[0]["speakername"] == "John Smith"
        assert filtered_df.iloc[0]["text"] == "Speech 1"

    def test_remove_unused_columns(self, processor):
        data = {
            "person_id": ["1234"],
            "text": ["Speech 1"],
            "redirect": ["something"],
            "url": ["http://example.com"],
            "colnum": ["123"],
        }
        df = pd.DataFrame(data)

        cleaned_df = processor._remove_unused_columns(df)

        assert len(cleaned_df) == 1
        assert "person_id" in cleaned_df.columns
        assert "text" in cleaned_df.columns
        assert "redirect" not in cleaned_df.columns
        assert "url" not in cleaned_df.columns
        assert "colnum" not in cleaned_df.columns

    def test_add_date_columns(self, processor):
        data = {"id": ["2020-01-01.1", "2020-06-15.2"]}
        df = pd.DataFrame(data)

        result = processor._add_date_columns(df, "id")

        assert "year" in result.columns
        assert "month" in result.columns
        assert "day" in result.columns
        assert result.iloc[0]["year"] == 2020
        assert result.iloc[0]["month"] == 1
        assert result.iloc[0]["day"] == 1

    def test_remove_duplicate_speeches(self, processor):
        data = {
            "id": ["speech1", "speech1", "speech2"],
            "text": ["First speech", "First speech duplicate", "Second speech"],
        }
        df = pd.DataFrame(data)

        result = processor._remove_duplicate_speeches(df)

        assert len(result) == 2
        assert list(result["id"].values) == ["speech1", "speech2"]

    def test_replace_unknown_identifiers(self, processor):
        data = {
            "person_id": ["1234", "unknown", "5678"],
            "speakerid": ["unknown", "9012", None],
            "text": ["Speech 1", "Speech 2", "Speech 3"],
        }
        df = pd.DataFrame(data)

        result = processor._replace_unknown_identifiers(df)

        assert result["person_id"].tolist()[0] == "1234"
        assert pd.isna(result["person_id"].tolist()[1])
        assert result["person_id"].tolist()[2] == "5678"
        assert "membership_id" in result.columns
        assert "speakerid" not in result.columns
        assert pd.isna(result["membership_id"].tolist()[0])
        assert result["membership_id"].tolist()[1] == "9012"
        assert pd.isna(result["membership_id"].tolist()[2])

    @pytest.mark.parametrize(
        "xml_content,expected_count",
        [
            (
                """
            <publicwhip>
                <speech id="2020-01-01.1">
                    <p>First paragraph</p>
                    <p>Second paragraph</p>
                </speech>
            </publicwhip>
            """,
                1,
            ),
            (
                """
            <publicwhip>
                <speech id="2020-01-01.1">
                    <p>Speech one</p>
                </speech>
                <speech id="2020-01-01.2">
                    <p>Speech two</p>
                </speech>
            </publicwhip>
            """,
                2,
            ),
        ],
    )
    def test_extract_speeches_with_multiple_paragraphs(
        self, processor, xml_content, expected_count
    ):
        fields = {"id"}
        speeches = processor._extract_speeches_from_single_file(xml_content, fields)

        assert len(speeches) == expected_count
        if expected_count == 1:
            assert "First paragraph Second paragraph" in speeches[0]["text"]

    def test_load_and_filter_xml_files(self, processor):
        # Create test XML files
        xml_dir = processor.xml_path
        xml_dir.mkdir()

        # Create files with different dates
        valid_file = xml_dir / "2020-01-01.xml"
        valid_file.write_text("<publicwhip><speech></speech></publicwhip>")

        invalid_date_file_lower = xml_dir / "2019-01-01.xml"
        invalid_date_file_lower.write_text("<publicwhip><speech></speech></publicwhip>")
        invalid_date_file_upper = xml_dir / "2021-01-01.xml"
        invalid_date_file_upper.write_text("<publicwhip><speech></speech></publicwhip>")

        result = processor._load_and_filter_xml_files()

        assert len(result) == 1
        assert "2020-01-01.xml" in result["file_name"].values

    def test_extract_and_clean_speeches(self, processor):
        raw_data = {
            "file_name": ["2020-01-01.xml"],
            "xml_text": [
                """
                <publicwhip>
                    <speech id="2020-01-01.1" person_id="1234">
                        <p>This is a perfectly valid speech with plenty of words in it</p>
                    </speech>
                    <speech id="2020-01-01.2" speakername="Several hon. Members">
                        <p>Invalid speech</p>
                    </speech>
                </publicwhip>
            """
            ],
        }
        raw_df = pd.DataFrame(raw_data)

        result = processor._extract_and_clean_speeches(raw_df, year=2020)

        assert len(result) == 1
        assert result.iloc[0]["person_id"] == "1234"
        assert "perfectly valid speech" in result.iloc[0]["text"]

    def test_read_xml_file_encoding_handling(self, processor, tmp_path):
        test_file = tmp_path / "test.xml"

        # Test UTF-8
        utf8_content = "<speech>Test é</speech>"
        test_file.write_text(utf8_content, encoding="utf-8")
        result = processor._read_xml_file(test_file)
        assert "Test é" in result
