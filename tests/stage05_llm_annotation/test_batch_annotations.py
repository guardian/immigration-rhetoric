import asyncio
import json

import pandas as pd
import pytest
from immigration_rhetoric.stage05_llm_annotation.batch_annotations import (
    BatchAnnotations,
)


class FakeAnnotations(BatchAnnotations):
    """A fake backend that returns a canned step-2 JSON response, for testing the base class in isolation."""

    models = {"fake_model": {"modelId": "fake-model-id"}}

    def generate_message(self, prompt, temperature, max_tokens, model_id):
        if "label" in prompt:
            return '{"A": "very high", "B": "low", "C": "low", "D": "low"}'
        return "a step-1 explanation"


class TestSplitIntoBatches:
    def test_single_batch_when_chunk_size_is_negative(self):
        df = pd.DataFrame({"text": ["a", "b", "c"]})
        batches = BatchAnnotations.split_into_batches(df, chunk_size=-1)
        assert len(batches) == 1
        assert len(batches[0]) == 3

    def test_splits_into_chunks(self):
        df = pd.DataFrame({"text": ["a", "b", "c", "d", "e"]})
        batches = BatchAnnotations.split_into_batches(df, chunk_size=2)
        assert [len(b) for b in batches] == [2, 2, 1]


class TestAnnotateSnippetsBatch:
    def test_adds_label_columns_and_prints_json(self, capsys):
        df = pd.DataFrame({"text": ["an immigration snippet"], "meta": [{}]})

        result = FakeAnnotations().annotate_snippets_batch(
            df, model_name="fake_model", temperature=0.0, max_tokens=100
        )

        assert result["fake_model_step1_responses"].iloc[0] == "a step-1 explanation"
        assert result["fake_model_labels"].iloc[0] == ["POSITIVE_TOWARDS_IMMIGRATION"]

        printed = json.loads(capsys.readouterr().out)
        assert printed["fake_model_labels"] == ["POSITIVE_TOWARDS_IMMIGRATION"]

    def test_records_error_for_failing_row_and_continues(self):
        class FailingAnnotations(FakeAnnotations):
            def generate_message(self, prompt, temperature, max_tokens, model_id):
                raise RuntimeError("API error")

        df = pd.DataFrame({"text": ["a snippet"]})
        result = FailingAnnotations().annotate_snippets_batch(
            df, model_name="fake_model", temperature=0.0, max_tokens=100
        )

        assert result["fake_model_labels"].iloc[0] == "Error"

    def test_only_keeps_available_columns(self):
        df = pd.DataFrame({"text": ["a snippet"], "irrelevant_column": ["x"]})
        result = FakeAnnotations().annotate_snippets_batch(
            df, model_name="fake_model", temperature=0.0, max_tokens=100
        )
        assert "irrelevant_column" not in result.columns
        assert "text" in result.columns


class TestAnnotateSnippets:
    def test_annotates_snippets_from_jsonl(self, tmp_path):
        input_file = tmp_path / "snippets.jsonl"
        pd.DataFrame({"text": ["one", "two"]}).to_json(
            input_file, orient="records", lines=True
        )

        results = asyncio.run(
            FakeAnnotations().annotate_snippets(
                executor=None,
                model_name="fake_model",
                temperature=0.0,
                max_tokens=100,
                input_file=str(input_file),
                chunk_size=-1,
            )
        )

        combined = pd.concat(results, axis=0)
        assert len(combined) == 2
        assert set(combined["fake_model_labels"].iloc[0]) == {
            "POSITIVE_TOWARDS_IMMIGRATION"
        }

    def test_raises_on_unsupported_file_type(self, tmp_path):
        input_file = tmp_path / "snippets.txt"
        input_file.write_text("not a real input")

        with pytest.raises(ValueError):
            asyncio.run(
                FakeAnnotations().annotate_snippets(
                    executor=None,
                    model_name="fake_model",
                    temperature=0.0,
                    max_tokens=100,
                    input_file=str(input_file),
                    chunk_size=-1,
                )
            )
