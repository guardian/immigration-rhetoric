from unittest.mock import patch

from immigration_rhetoric.stage07_model_training.run_build_training_data import (
    parse_args,
    run_build_training_data,
)


class TestParseArgs:
    def test_defaults(self):
        with patch("sys.argv", ["script.py", "-i", "in.jsonl", "-o", "corpus"]):
            args = parse_args()

        assert args.seeds == [17, 23, 31, 43, 59]
        assert args.holdout_frac == 0.1
        assert args.eval_frac == 0.2

    def test_custom_seeds(self):
        with patch(
            "sys.argv",
            ["script.py", "-i", "in.jsonl", "-o", "corpus", "--seeds", "1", "2"],
        ):
            args = parse_args()

        assert args.seeds == [1, 2]


class TestRunBuildTrainingData:
    @patch(
        "immigration_rhetoric.stage07_model_training.run_build_training_data.TrainingDataBuilder"
    )
    def test_delegates_to_builder(self, mock_builder_cls):
        run_build_training_data(
            input_file="in.jsonl",
            output_dir="corpus",
            seeds=[1],
            holdout_frac=0.2,
            eval_frac=0.3,
        )

        mock_builder_cls.assert_called_once_with(
            output_dir="corpus", holdout_frac=0.2, eval_frac=0.3
        )
        mock_builder_cls.return_value.build.assert_called_once_with(
            input_file="in.jsonl", seeds=[1]
        )
