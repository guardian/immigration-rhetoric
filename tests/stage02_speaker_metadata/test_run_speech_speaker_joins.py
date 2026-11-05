from unittest.mock import Mock, patch

from immigration_rhetoric.stage02_speaker_metadata.run_speech_speaker_joins import (
    parse_args,
    run_speech_speaker_joins,
)


class TestParseArgs:
    def test_parse_args_defaults(self):
        with patch("sys.argv", ["script.py"]):
            args = parse_args()
            assert args.start_year == 1919

    def test_parse_args_with_values(self):
        with patch(
            "sys.argv",
            [
                "script.py",
                "--metadata-dir",
                "/meta",
                "--raw-speech-dir",
                "/raw",
                "--start-year",
                "1986",
                "--end-year",
                "2016",
                "--output-dir",
                "/out",
            ],
        ):
            args = parse_args()
            assert args.metadata_dir == "/meta"
            assert args.raw_speech_dir == "/raw"
            assert args.start_year == 1986
            assert args.end_year == 2016
            assert args.output_dir == "/out"


class TestRunSpeechSpeakerJoins:
    @patch(
        "immigration_rhetoric.stage02_speaker_metadata.run_speech_speaker_joins.SpeechMetadataJoiner"
    )
    @patch(
        "immigration_rhetoric.stage02_speaker_metadata.run_speech_speaker_joins."
        "SpeakerMetadataPreprocessor"
    )
    def test_calls_each_join_stage(self, mock_preprocessor_cls, mock_joiner_cls):
        mock_preprocessor = mock_preprocessor_cls.return_value
        mock_preprocessor.load_people_json.return_value = (Mock(), Mock(), Mock())

        mock_joiner = mock_joiner_cls.return_value
        empty_calls_df = Mock()
        empty_calls_df.__len__ = Mock(return_value=0)
        mock_joiner.split_speeches_into_join_stages.return_value = (
            empty_calls_df,
            empty_calls_df,
            empty_calls_df,
        )
        mock_joiner.load_speech_csv.return_value.columns = []

        run_speech_speaker_joins(
            metadata_dir="/meta",
            raw_speech_dir="/raw",
            start_year=1986,
            end_year=2016,
            output_dir="/out",
        )

        mock_preprocessor.preprocess_memberships.assert_called_once()
        mock_preprocessor.preprocess_wikidata.assert_called_once()
        mock_joiner.name_joining.assert_called_once()
        mock_joiner.concat_joined_speeches.assert_called_once()
        mock_joiner.save_joined_speeches.assert_called_once()
