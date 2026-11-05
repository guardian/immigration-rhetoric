from unittest.mock import patch

import pandas as pd
from immigration_rhetoric.stage04_prepare_annotation_data.run_snippet_sampling import (
    parse_args,
    run_snippet_sampling,
)


class TestParseArgs:
    def test_parse_args_defaults(self):
        with patch("sys.argv", ["script.py"]):
            args = parse_args()
            assert args.start_year == 1919
            assert args.sample_size == 5000
            assert args.sample_by_year_month is False
            assert args.exclude_sample_file is None

    def test_parse_args_with_values(self):
        with patch(
            "sys.argv",
            [
                "script.py",
                "--input-dir",
                "/in",
                "--output-dir",
                "/out",
                "--start-year",
                "1986",
                "--end-year",
                "2016",
                "--number-to-sample",
                "100",
                "--sample-by-year-month",
                "--exclude-sample-file",
                "/previous.csv",
            ],
        ):
            args = parse_args()
            assert args.input_dir == "/in"
            assert args.output_dir == "/out"
            assert args.start_year == 1986
            assert args.end_year == 2016
            assert args.sample_size == 100
            assert args.sample_by_year_month is True
            assert args.exclude_sample_file == "/previous.csv"


class TestRunSnippetSampling:
    @patch(
        "immigration_rhetoric.stage04_prepare_annotation_data.run_snippet_sampling.SnippetSampler"
    )
    def test_samples_by_year_by_default(self, mock_sampler_cls):
        mock_sampler = mock_sampler_cls.return_value
        mock_sampler.load_snippets.return_value = pd.DataFrame(
            {"year": [1986], "month": [1]}
        )

        run_snippet_sampling(
            input_dir="/in",
            output_dir="/out",
            start_year=1986,
            end_year=1986,
            sample_size=10,
            sample_by_year_month=False,
            exclude_sample_file=None,
        )

        mock_sampler.sample_by_column_value.assert_called_once()
        assert mock_sampler.sample_by_column_value.call_args.args[1] == "year"
        mock_sampler.exclude_already_sampled.assert_not_called()
        mock_sampler.output_sample.assert_called_once()

    @patch(
        "immigration_rhetoric.stage04_prepare_annotation_data.run_snippet_sampling.SnippetSampler"
    )
    def test_samples_by_year_month_when_requested(self, mock_sampler_cls):
        mock_sampler = mock_sampler_cls.return_value
        mock_sampler.load_snippets.return_value = pd.DataFrame(
            {"year": [1986], "month": [1]}
        )

        run_snippet_sampling(
            input_dir="/in",
            output_dir="/out",
            start_year=1986,
            end_year=1986,
            sample_size=10,
            sample_by_year_month=True,
            exclude_sample_file=None,
        )

        assert mock_sampler.sample_by_column_value.call_args.args[1] == "year-month"

    @patch("pandas.read_csv")
    @patch(
        "immigration_rhetoric.stage04_prepare_annotation_data.run_snippet_sampling.SnippetSampler"
    )
    def test_excludes_previous_sample_when_file_given(
        self, mock_sampler_cls, mock_read_csv
    ):
        mock_sampler = mock_sampler_cls.return_value
        mock_sampler.load_snippets.return_value = pd.DataFrame(
            {"year": [1986], "month": [1]}
        )
        mock_read_csv.return_value = pd.DataFrame({"snippet": ["a"]})

        run_snippet_sampling(
            input_dir="/in",
            output_dir="/out",
            start_year=1986,
            end_year=1986,
            sample_size=10,
            sample_by_year_month=False,
            exclude_sample_file="/previous.csv",
        )

        mock_sampler.exclude_already_sampled.assert_called_once()
