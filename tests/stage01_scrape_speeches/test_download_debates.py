from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import requests
from immigration_rhetoric.stage01_scrape_speeches.download_debates import (
    DebatesDownloader,
)


@pytest.fixture
def mock_args():
    args = Mock()
    args.start_date = datetime(2020, 1, 1)
    args.end_date = datetime(2020, 12, 31)
    return args


@pytest.fixture
def downloader(tmp_path):
    return DebatesDownloader(
        output_path=tmp_path / "xml_files",
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2020, 12, 31),
    )


@pytest.fixture
def mock_response():
    mock = Mock()
    mock.status_code = 200
    return mock


class TestDebatesDownloader:
    def test_init(self, downloader, tmp_path):
        assert downloader.output_path == tmp_path / "xml_files"

    def test_download_xml_file(self, downloader, mock_response):
        # Create the output directory
        downloader.output_path.mkdir(parents=True, exist_ok=True)

        filename = "test_debate.xml"
        mock_response.content = b"<xml>test content</xml>"

        with patch("requests.get", return_value=mock_response):
            downloader._download_xml_file(filename, downloader.output_path)

        file_path = downloader.output_path / filename
        assert file_path.exists()
        assert file_path.read_bytes() == b"<xml>test content</xml>"

    def test_fetch_available_xml_files(self, downloader, mock_response):
        mock_html = """
        <html>
            <a href="debate1.xml">debate1.xml</a>
            <a href="debate2.xml">debate2.xml</a>
            <a href="other.txt">other.txt</a>
        </html>
        """
        mock_response.text = mock_html

        with patch("requests.get", return_value=mock_response):
            result = downloader._fetch_available_xml_files()

        assert len(result) == 2
        assert "debate1.xml" in result
        assert "debate2.xml" in result
        assert "other.txt" not in result

    def test_filter_download_list_by_date(self, downloader, mock_args):
        xml_files = ["2020-01-01.xml", "2020-06-15.xml", "2021-01-01.xml"]

        result = downloader._filter_download_list_by_date(
            xml_files, mock_args.start_date, mock_args.end_date, download_all=True
        )

        assert len(result) == 2
        assert "2021-01-01.xml" not in result["xml_file"].values

    def test_filter_download_list_by_date_skip_existing(self, downloader, mock_args):
        # Create existing file
        downloader.output_path.mkdir(parents=True)
        (downloader.output_path / "2020-01-01.xml").touch()

        xml_files = ["2020-01-01.xml", "2020-06-15.xml"]
        result = downloader._filter_download_list_by_date(
            xml_files, mock_args.start_date, mock_args.end_date, download_all=False
        )

        assert len(result) == 1
        assert "2020-01-01.xml" not in result["xml_file"].values

    def test_download_debates_empty_date_range(self, downloader):
        with patch(
            "immigration_rhetoric.stage01_scrape_speeches.download_debates.DebatesDownloader._fetch_available_xml_files"
        ) as mock_fetch, patch(
            "immigration_rhetoric.stage01_scrape_speeches.download_debates.DebatesDownloader._filter_download_list_by_date"
        ) as mock_filter:
            # Mock the fetch to return some files
            mock_fetch.return_value = ["2020-01-01.xml", "2020-02-01.xml"]
            # Mock the filter to return an empty dataframe
            mock_filter.return_value = pd.DataFrame(columns=["xml_file"])

            downloader.download_debates()

            # Should not attempt to download any files
            assert not list(downloader.output_path.glob("*.xml"))
            # Verify our mocks were called
            mock_fetch.assert_called_once()
            mock_filter.assert_called_once()

    @patch("requests.get")
    def test_download_xml_file_timeout(self, mock_get, downloader):
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")

        with pytest.raises(requests.exceptions.RequestException) as exc_info:
            downloader._download_xml_file("test.xml", downloader.output_path)
        assert "Connection timed out" in str(exc_info.value)

    @patch("requests.get")
    def test_download_xml_file_http_error(self, mock_get, downloader):
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "404 Client Error"
        )
        mock_get.return_value = mock_response

        with pytest.raises(requests.exceptions.RequestException) as exc_info:
            downloader._download_xml_file("test.xml", downloader.output_path)
        assert "404 Client Error" in str(exc_info.value)

    @patch("requests.get")
    def test_fetch_available_files_connection_error(self, mock_get, downloader):
        mock_get.side_effect = requests.exceptions.ConnectionError("Failed to connect")

        with pytest.raises(requests.exceptions.RequestException) as exc_info:
            downloader._fetch_available_xml_files()
        assert "Failed to connect" in str(exc_info.value)

    def test_filter_download_list_empty_input(self, downloader):
        result = downloader._filter_download_list_by_date(
            [], datetime(2020, 1, 1), datetime(2020, 12, 31), download_all=True
        )
        assert len(result) == 0

    def test_filter_download_list_invalid_dates(self, downloader):
        xml_files = ["invalid_date.xml", "2020-01-01.xml"]
        result = downloader._filter_download_list_by_date(
            xml_files, datetime(2020, 1, 1), datetime(2020, 12, 31), download_all=True
        )
        # Should only include valid dated file
        assert len(result) == 1
        assert "2020-01-01.xml" in result["xml_file"].values

    def test_download_debates_with_existing_files(self, downloader, mock_response):
        # Create existing file
        downloader.output_path.mkdir(parents=True)
        existing_file = downloader.output_path / "2020-01-01.xml"
        existing_file.write_text("<xml>existing content</xml>")

        with patch("requests.get", return_value=mock_response) as mock_get:
            mock_response.text = (
                "<html><a href='2020-01-01.xml'>2020-01-01.xml</a></html>"
            )

            # Test with download_all=False (should skip existing)
            downloader.download_debates()
            assert mock_get.call_count == 1  # Only called for fetching file list
            assert existing_file.read_text() == "<xml>existing content</xml>"  # To check content is unchanged with download_all=False

            # Test with download_all=True (should redownload)
            mock_get.reset_mock()
            mock_response.content = b"<xml>new content</xml>"
            downloader.download_all = True  # Set download_all flag
            downloader.download_debates()
            assert mock_get.call_count == 2  # Called for list and file download
            assert existing_file.read_bytes() == b"<xml>new content</xml>" # To check. content is replaced when download_all=True

    def test_download_debates_malformed_html(self, downloader):
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.text = "Invalid HTML"
            mock_get.return_value = mock_response

            # Should handle malformed HTML gracefully
            result = downloader._fetch_available_xml_files()
            assert isinstance(result, list)
            assert len(result) == 0
