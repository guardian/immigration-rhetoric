from unittest.mock import MagicMock, patch

from immigration_rhetoric.stage05_llm_annotation.run_bedrock_annotations import (
    parse_args,
    run_bedrock_annotations,
)


class TestParseArgs:
    def test_parse_args_required_and_defaults(self):
        with patch(
            "sys.argv",
            [
                "script.py",
                "--model-name",
                "claude_haiku",
                "--input-file",
                "/in.jsonl",
                "--chunk-size",
                "10",
            ],
        ):
            args = parse_args()
            assert args.model_name == "claude_haiku"
            assert args.input_file == "/in.jsonl"
            assert args.chunk_size == 10
            assert args.temperature == 0.0
            assert args.max_tokens == 300
            assert args.output_file is None
            assert args.max_processes == 10

    def test_parse_args_with_values(self):
        with patch(
            "sys.argv",
            [
                "script.py",
                "--model-name",
                "claude_haiku",
                "--input-file",
                "/in.jsonl",
                "--chunk-size",
                "5",
                "--temperature",
                "0.7",
                "--max-tokens",
                "500",
                "--output-file",
                "/out.jsonl",
                "--max-processes",
                "3",
            ],
        ):
            args = parse_args()
            assert args.temperature == 0.7
            assert args.max_tokens == 500
            assert args.output_file == "/out.jsonl"
            assert args.max_processes == 3


class TestRunBedrockAnnotations:
    @patch("immigration_rhetoric.stage05_llm_annotation.run_bedrock_annotations.boto3")
    @patch(
        "immigration_rhetoric.stage05_llm_annotation.run_bedrock_annotations.BedrockAnnotations"
    )
    def test_builds_client_in_the_models_region_and_runs(
        self, mock_annotations_cls, mock_boto3
    ):
        mock_annotations = mock_annotations_cls.return_value

        async def fake_annotate_snippets(**kwargs):
            return []

        mock_annotations.annotate_snippets = MagicMock(
            side_effect=fake_annotate_snippets
        )

        run_bedrock_annotations(
            model_name="claude_opus",
            temperature=0.0,
            max_tokens=300,
            input_file="/in.jsonl",
            chunk_size=10,
            output_file=None,
            max_processes=7,
        )

        mock_boto3.client.assert_called_once_with(
            service_name="bedrock-runtime", region_name="us-west-2"
        )
        mock_annotations_cls.assert_called_once_with(
            bedrock_runtime=mock_boto3.client.return_value
        )

    @patch("immigration_rhetoric.stage05_llm_annotation.run_bedrock_annotations.boto3")
    @patch(
        "immigration_rhetoric.stage05_llm_annotation.run_bedrock_annotations.BedrockAnnotations"
    )
    def test_writes_combined_output_file_when_given(
        self, mock_annotations_cls, mock_boto3, tmp_path
    ):
        import pandas as pd

        mock_annotations = mock_annotations_cls.return_value

        async def fake_annotate_snippets(**kwargs):
            return [pd.DataFrame({"text": ["a"]})]

        mock_annotations.annotate_snippets = MagicMock(
            side_effect=fake_annotate_snippets
        )

        output_file = tmp_path / "out.jsonl"
        run_bedrock_annotations(
            model_name="claude_haiku",
            temperature=0.0,
            max_tokens=300,
            input_file="/in.jsonl",
            chunk_size=10,
            output_file=str(output_file),
            max_processes=7,
        )

        assert output_file.exists()
