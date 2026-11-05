from unittest.mock import patch

from immigration_rhetoric.stage06_agreement.run_evaluate_agreement import (
    parse_args,
    run_evaluate_agreement,
)


class TestParseArgs:
    def test_parse_args_supports_repeated_groups(self):
        with patch(
            "sys.argv",
            [
                "script.py",
                "--annotations-file",
                "human_and_llm_20241006_multi.jsonl",
                "--annotator-group",
                "humans=A,B,C",
                "--annotator-group",
                "humans-and-llm=A,B,C,claude",
            ],
        ):
            args = parse_args()
            assert args.annotations_file == "human_and_llm_20241006_multi.jsonl"
            assert args.annotator_groups == [
                "humans=A,B,C",
                "humans-and-llm=A,B,C,claude",
            ]


class TestRunEvaluateAgreement:
    @patch(
        "immigration_rhetoric.stage06_agreement.run_evaluate_agreement.AnnotatorAgreement"
    )
    def test_scores_each_group_and_returns_results(self, mock_agreement_cls):
        mock_agreement = mock_agreement_cls.return_value
        mock_agreement.load_annotations.return_value = "annotations-df"
        mock_agreement.get_agreement_score.side_effect = [0.9, 0.7]

        scores = run_evaluate_agreement(
            input_dir="/in",
            annotations_file="human_and_llm_20241006_multi.jsonl",
            annotator_groups=["humans=A,B", "humans-and-llm=A,B,claude"],
        )

        assert scores == {"humans": 0.9, "humans-and-llm": 0.7}
        mock_agreement.get_agreement_score.assert_any_call("annotations-df", ["A", "B"])
        mock_agreement.get_agreement_score.assert_any_call(
            "annotations-df", ["A", "B", "claude"]
        )
