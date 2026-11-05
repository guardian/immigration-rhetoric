from unittest.mock import Mock, patch

import pytest
from immigration_rhetoric.stage02_speaker_metadata.run_get_metadata import (
    parse_args,
    run_get_metadata,
)


@pytest.fixture
def mock_args(tmp_path):
    args = Mock()
    args.output_dir = str(tmp_path)
    args.force_download = False
    return args


class TestRunGetMetadata:
    def test_parse_args_defaults(self):
        with patch("sys.argv", ["script.py"]):
            args = parse_args()
            assert not args.force_download

    def test_parse_args_with_values(self):
        with patch(
            "sys.argv",
            [
                "script.py",
                "--output-dir",
                "/test/path",
                "--force-download",
            ],
        ):
            args = parse_args()
            assert args.output_dir == "/test/path"
            assert args.force_download is True

    @patch(
        "immigration_rhetoric.stage02_speaker_metadata.run_get_metadata.SpeakerMetadataDownloader"
    )
    def test_run_get_metadata(self, mock_metadata_downloader, mock_args):
        run_get_metadata(mock_args.output_dir, mock_args.force_download)

        mock_metadata_downloader.return_value.collect_speaker_metadata.assert_called_once()

    def test_run_get_metadata_error_handling(self, mock_args):
        with patch(
            "immigration_rhetoric.stage02_speaker_metadata.run_get_metadata."
            "SpeakerMetadataDownloader.collect_speaker_metadata",
            side_effect=Exception("Test error"),
        ):
            with pytest.raises(Exception) as exc_info:
                run_get_metadata(mock_args.output_dir, mock_args.force_download)
            assert "Test error" in str(exc_info.value)
