from unittest.mock import MagicMock, patch

from immigration_rhetoric.stage05_llm_annotation.run_openai_annotations import (
    parse_args,
    run_openai_annotations,
)


class TestParseArgs:
    def test_parse_args_required_and_defaults(self):
        with patch(
            "sys.argv",
            [
                "script.py",
                "--model-name",
                "gpt4_1",
                "--input-file",
                "/in.jsonl",
                "--chunk-size",
                "10",
            ],
        ):
            args = parse_args()
            assert args.model_name == "gpt4_1"
            assert args.input_file == "/in.jsonl"
            assert args.chunk_size == 10
            assert args.temperature == 0.0
            assert args.max_tokens == 300
            assert args.output_file is None
            assert args.max_processes == 10

    def test_parse_args_only_defines_output_file_once(self):
        # Regression test: defining --output-file twice raises argparse.ArgumentError (conflicting option
        # strings) as soon as parse_args() is called.
        with patch(
            "sys.argv",
            [
                "script.py",
                "--model-name",
                "gpt4_1",
                "--input-file",
                "/in.jsonl",
                "--chunk-size",
                "10",
                "--output-file",
                "/out.jsonl",
            ],
        ):
            args = parse_args()
            assert args.output_file == "/out.jsonl"


class TestRunOpenAIAnnotations:
    @patch(
        "immigration_rhetoric.stage05_llm_annotation.run_openai_annotations.OpenAIAnnotations"
    )
    def test_runs_annotation(self, mock_annotations_cls):
        mock_annotations = mock_annotations_cls.return_value

        async def fake_annotate_snippets(**kwargs):
            return []

        mock_annotations.annotate_snippets = MagicMock(
            side_effect=fake_annotate_snippets
        )

        run_openai_annotations(
            model_name="gpt4_1",
            temperature=0.0,
            max_tokens=300,
            input_file="/in.jsonl",
            chunk_size=10,
            output_file=None,
            max_processes=7,
        )

        mock_annotations_cls.assert_called_once_with()

    @patch(
        "immigration_rhetoric.stage05_llm_annotation.run_openai_annotations.OpenAIAnnotations"
    )
    def test_writes_combined_output_file_when_given(
        self, mock_annotations_cls, tmp_path
    ):
        import pandas as pd

        mock_annotations = mock_annotations_cls.return_value

        async def fake_annotate_snippets(**kwargs):
            return [pd.DataFrame({"text": ["a"]})]

        mock_annotations.annotate_snippets = MagicMock(
            side_effect=fake_annotate_snippets
        )

        output_file = tmp_path / "out.jsonl"
        run_openai_annotations(
            model_name="gpt4_1",
            temperature=0.0,
            max_tokens=300,
            input_file="/in.jsonl",
            chunk_size=10,
            output_file=str(output_file),
            max_processes=7,
        )

        assert output_file.exists()
