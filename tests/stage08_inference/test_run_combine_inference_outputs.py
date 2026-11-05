from unittest.mock import patch

import pandas as pd
from immigration_rhetoric.stage08_inference.run_combine_inference_outputs import (
    parse_args,
    run_combine_inference_outputs,
)

MODULE = "immigration_rhetoric.stage08_inference.run_combine_inference_outputs"


class TestParseArgs:
    def test_defaults(self):
        with patch(
            "sys.argv", ["script.py", "-v", "v1.0.0", "--max-speech-date", "20251218"]
        ):
            args = parse_args()

        assert args.input_dir.endswith("inference")
        assert args.min_speech_date == "19190101"
        assert args.max_speech_date == "20251218"


class TestRunCombineInferenceOutputs:
    @patch(f"{MODULE}.InferenceOutputCombiner")
    def test_writes_both_files_with_version_and_dates(
        self, mock_combiner_cls, tmp_path
    ):
        speeches = pd.DataFrame({"speech_id": ["a"], "date": ["1950-03-06"]})
        snippets = pd.DataFrame({"snippet_id": ["a_1", "a_2"]})
        mock_combiner_cls.return_value.combine.return_value = (speeches, snippets)

        run_combine_inference_outputs(
            input_dir=str(tmp_path / "in"),
            output_dir=str(tmp_path / "out"),
            model_version="v1.0.0",
            min_speech_date="19190101",
            max_speech_date="20251218",
        )

        mock_combiner_cls.return_value.combine.assert_called_once_with(tmp_path / "in")
        written_speeches = pd.read_json(
            tmp_path / "out" / "speeches_v1.0.0_19190101_20251218.jsonl",
            lines=True,
            convert_dates=False,
        )
        written_snippets = pd.read_json(
            tmp_path / "out" / "labelled_snippets_v1.0.0_19190101_20251218.jsonl",
            lines=True,
        )
        assert written_speeches["date"].to_list() == ["1950-03-06"]
        assert written_snippets["snippet_id"].to_list() == ["a_1", "a_2"]
