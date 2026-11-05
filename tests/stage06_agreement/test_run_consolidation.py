from unittest.mock import patch

from immigration_rhetoric.stage06_agreement.run_consolidation import (
    parse_args,
    run_consolidation,
)


class TestParseArgs:
    def test_parse_args_required_and_defaults(self):
        with patch(
            "sys.argv",
            [
                "script.py",
                "--files",
                "annotator_a",
                "annotator_b",
                "--label",
                "humans",
                "--date",
                "20241006",
            ],
        ):
            args = parse_args()
            assert args.files == ["annotator_a", "annotator_b"]
            assert args.label == "humans"
            assert args.date == "20241006"


class TestRunConsolidation:
    @patch(
        "immigration_rhetoric.stage06_agreement.run_consolidation.ConsolidateAnnotations"
    )
    def test_delegates_to_consolidator(self, mock_consolidator_cls):
        mock_consolidator = mock_consolidator_cls.return_value

        run_consolidation(
            input_dir="/in",
            output_dir="/out",
            files=["annotator_a", "annotator_b"],
            label="humans",
            date="20241006",
        )

        mock_consolidator_cls.assert_called_once_with(
            input_dir="/in", output_dir="/out"
        )
        mock_consolidator.consolidate.assert_called_once_with(
            files=["annotator_a", "annotator_b"], label="humans", date="20241006"
        )
