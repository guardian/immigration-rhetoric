from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from immigration_rhetoric.stage01_scrape_speeches.run_get_speeches import (
    parse_args,
    run_get_speeches,
)


@pytest.fixture
def mock_args(tmp_path):
    args = Mock()
    args.output_dir = str(tmp_path)
    args.start_date = datetime(2020, 1, 1)
    args.end_date = datetime(2020, 12, 31)
    args.force_download = False
    return args


class TestRunGetSpeeches:
    def test_parse_args_defaults(self):
        with patch("sys.argv", ["script.py"]):
            args = parse_args()
            assert not args.force_download

    def test_parse_args_with_values(self):
        test_date = "2023-01-15"
        with patch(
            "sys.argv",
            [
                "script.py",
                "--output-dir",
                "/test/path",
                "--start-date",
                test_date,
                "--end-date",
                test_date,
                "--force-download",
            ],
        ):
            args = parse_args()
            assert args.output_dir == "/test/path"
            assert args.start_date == datetime(2023, 1, 15)
            assert args.end_date == datetime(2023, 1, 15)
            assert args.force_download is True

    def test_parse_args_invalid_date(self):
        with patch("sys.argv", ["script.py", "--start-date", "invalid-date"]):
            with pytest.raises(SystemExit):
                parse_args()

    @patch(
        "immigration_rhetoric.stage01_scrape_speeches.run_get_speeches.DebatesDownloader"
    )
    @patch(
        "immigration_rhetoric.stage01_scrape_speeches.run_get_speeches.SpeechesProcessor"
    )
    def test_run_get_speeches(
        self,
        mock_speeches_processor,
        mock_debates_downloader,
        mock_args,
    ):
        run_get_speeches(
            mock_args.output_dir,
            mock_args.start_date,
            mock_args.end_date,
            mock_args.force_download,
        )

        # Verify each component was called correctly
        xml_output_path = Path(mock_args.output_dir) / "xml_files"
        speeches_output_path = Path(mock_args.output_dir) / "raw_speeches"

        mock_debates_downloader.assert_called_once_with(
            output_path=xml_output_path,
            start_date=mock_args.start_date,
            end_date=mock_args.end_date,
            download_all=mock_args.force_download,
        )
        mock_debates_downloader.return_value.download_debates.assert_called_once()

        mock_speeches_processor.assert_called_once_with(
            xml_path=xml_output_path,
            output_path=speeches_output_path,
            start_date=mock_args.start_date,
            end_date=mock_args.end_date,
        )

        mock_speeches_processor.return_value.get_speeches_from_xml.assert_called_once()

    def test_run_get_speeches_error_handling(self, mock_args):
        with patch(
            "immigration_rhetoric.stage01_scrape_speeches.run_get_speeches.DebatesDownloader.download_debates",
            side_effect=Exception("Test error"),
        ):
            with pytest.raises(Exception) as exc_info:
                run_get_speeches(
                    mock_args.output_dir,
                    mock_args.start_date,
                    mock_args.end_date,
                    mock_args.force_download,
                )
            assert "Test error" in str(exc_info.value)
