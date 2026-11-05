from unittest.mock import patch

import pandas as pd
from immigration_rhetoric.stage03_snippet_extraction.run_extract_snippets import (
    _init_worker,
    load_speeches_by_year,
    parse_args,
    run_extract_snippets,
)


class TestParseArgs:
    def test_parse_args_defaults(self):
        with patch("sys.argv", ["script.py"]):
            args = parse_args()
            assert args.start_year == 1919
            assert args.min_word_count == 0
            assert args.speeches_per_year is None

    def test_parse_args_with_values(self):
        with patch(
            "sys.argv",
            [
                "script.py",
                "--output-dir",
                "/out",
                "--trigger-file",
                "/triggers.csv",
                "--start-year",
                "1986",
                "--end-year",
                "2016",
                "--min-word-count",
                "10",
                "--num-processes",
                "7",
            ],
        ):
            args = parse_args()
            assert args.output_dir == "/out"
            assert args.trigger_file == "/triggers.csv"
            assert args.start_year == 1986
            assert args.end_year == 2016
            assert args.min_word_count == 10
            assert args.num_processes == 7


class TestLoadSpeechesByYear:
    def test_loads_only_files_within_year_range(self, tmp_path):
        pd.DataFrame({"id": [1], "text": ["a"]}).to_csv(
            tmp_path / "speeches_1985.csv", index=False
        )
        pd.DataFrame({"id": [2], "text": ["b"]}).to_csv(
            tmp_path / "speeches_1986.csv", index=False
        )
        pd.DataFrame({"id": [3], "text": ["c"]}).to_csv(
            tmp_path / "speeches_1987.csv", index=False
        )

        dataframes = load_speeches_by_year(tmp_path, start_year=1986, end_year=1986)

        assert len(dataframes) == 1
        assert dataframes[0][0] == 1986


class TestRunExtractSnippets:
    @patch("immigration_rhetoric.stage03_snippet_extraction.run_extract_snippets.spacy")
    @patch("immigration_rhetoric.stage03_snippet_extraction.run_extract_snippets.Pool")
    @patch(
        "immigration_rhetoric.stage03_snippet_extraction.run_extract_snippets.SnippetExtractor"
    )
    @patch(
        "immigration_rhetoric.stage03_snippet_extraction.run_extract_snippets.load_speeches_by_year"
    )
    def test_runs_pool_over_loaded_years(
        self, mock_load_speeches, mock_extractor_cls, mock_pool_cls, mock_spacy
    ):
        mock_load_speeches.return_value = [
            (1986, pd.DataFrame({"id": [1], "text": ["asylum"]}))
        ]
        mock_pool = mock_pool_cls.return_value.__enter__.return_value
        mock_pool.imap.return_value = iter([(1, 1)])

        with patch("pandas.read_csv", return_value=pd.DataFrame({"term": ["asylum"]})):
            run_extract_snippets(
                output_dir="/out",
                trigger_file="/triggers.csv",
                start_year=1986,
                end_year=1986,
                min_word_count=0,
                speeches_per_year=None,
                num_processes=2,
            )

        mock_spacy.load.assert_called_once()
        mock_extractor_cls.assert_called_once()
        mock_pool.imap.assert_called_once()
        assert mock_pool_cls.call_args.kwargs["initializer"] is _init_worker
        assert mock_pool_cls.call_args.kwargs["initargs"] == (
            mock_spacy.load.return_value,
        )
        assert mock_pool_cls.call_args.kwargs["initializer"] is _init_worker
