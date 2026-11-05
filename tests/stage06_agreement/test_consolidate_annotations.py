import pandas as pd
import pytest
from immigration_rhetoric.stage06_agreement.consolidate_annotations import (
    ConsolidateAnnotations,
)


class TestConsolidateAnswers:
    def test_keeps_a_single_row_when_annotators_agree(self):
        df = pd.DataFrame(
            {
                "text": ["snippet a", "snippet a"],
                "accept": [
                    ["POSITIVE_TOWARDS_IMMIGRATION"],
                    ["POSITIVE_TOWARDS_IMMIGRATION"],
                ],
                "_annotator_id": ["A", "B"],
            }
        )
        result = ConsolidateAnnotations.consolidate_answers(df)
        assert len(result) == 1
        assert result["text"].iloc[0] == "snippet a"

    def test_drops_rows_when_annotators_disagree(self):
        df = pd.DataFrame(
            {
                "text": ["snippet b", "snippet b"],
                "accept": [
                    ["POSITIVE_TOWARDS_IMMIGRATION"],
                    ["NEGATIVE_TOWARDS_IMMIGRATION"],
                ],
                "_annotator_id": ["A", "B"],
            }
        )
        result = ConsolidateAnnotations.consolidate_answers(df)
        assert len(result) == 0

    def test_keeps_single_annotator_snippets(self):
        df = pd.DataFrame(
            {
                "text": ["snippet c"],
                "accept": [["NEUTRAL_TOWARDS_IMMIGRATION"]],
                "_annotator_id": ["A"],
            }
        )
        result = ConsolidateAnnotations.consolidate_answers(df)
        assert len(result) == 1

    def test_does_not_leave_accept_str_column(self):
        df = pd.DataFrame(
            {
                "text": ["snippet a"],
                "accept": [["POSITIVE_TOWARDS_IMMIGRATION"]],
                "_annotator_id": ["A"],
            }
        )
        result = ConsolidateAnnotations.consolidate_answers(df)
        assert "accept_str" not in result.columns


class TestValidateAnnotations:
    def test_rejects_conflicting_ratings_from_the_same_annotator(self):
        df = pd.DataFrame(
            {
                "text": ["snippet a", "snippet a"],
                "accept": [
                    ["POSITIVE_TOWARDS_IMMIGRATION"],
                    ["NEGATIVE_TOWARDS_IMMIGRATION"],
                ],
                "_annotator_id": ["A", "A"],
            }
        )

        with pytest.raises(
            ValueError,
            match="Multiple ratings found for the same snippet and annotator",
        ):
            ConsolidateAnnotations.validate_annotations(df)


class TestDeduplicateAnnotations:
    def test_removes_identical_ratings_from_the_same_annotator(self):
        df = pd.DataFrame(
            {
                "text": ["snippet a", "snippet a"],
                "accept": [
                    ["POSITIVE_TOWARDS_IMMIGRATION"],
                    ["POSITIVE_TOWARDS_IMMIGRATION"],
                ],
                "_annotator_id": ["A", "A"],
            }
        )

        result = ConsolidateAnnotations.deduplicate_annotations(df)

        assert len(result) == 1
        assert result["accept"].iloc[0] == ["POSITIVE_TOWARDS_IMMIGRATION"]


class TestGetDuplicateAnnotations:
    def test_returns_every_row_for_multiply_annotated_snippets(self):
        df = pd.DataFrame(
            {
                "text": ["snippet a", "snippet a", "snippet b"],
                "accept": [
                    ["POSITIVE_TOWARDS_IMMIGRATION"],
                    ["NEGATIVE_TOWARDS_IMMIGRATION"],
                    ["NEUTRAL_TOWARDS_IMMIGRATION"],
                ],
                "_annotator_id": ["A", "B", "A"],
            }
        )
        result = ConsolidateAnnotations.get_duplicate_annotations(df)
        assert len(result) == 2
        assert set(result["text"]) == {"snippet a"}


class TestConsolidate:
    def test_loads_multiple_files_and_writes_both_outputs(self, tmp_path):
        input_dir = tmp_path / "raw"
        input_dir.mkdir()
        output_dir = tmp_path / "processed"

        pd.DataFrame(
            {
                "text": ["snippet a"],
                "accept": [["POSITIVE_TOWARDS_IMMIGRATION"]],
                "_annotator_id": ["A"],
            }
        ).to_json(input_dir / "annotator_a.jsonl", orient="records", lines=True)
        pd.DataFrame(
            {
                "text": ["snippet a"],
                "accept": [["POSITIVE_TOWARDS_IMMIGRATION"]],
                "_annotator_id": ["B"],
            }
        ).to_json(input_dir / "annotator_b.jsonl", orient="records", lines=True)

        consolidator = ConsolidateAnnotations(
            input_dir=input_dir, output_dir=output_dir
        )
        consolidator.consolidate(
            files=["annotator_a", "annotator_b"], label="humans", date="20241006"
        )

        consolidated = pd.read_json(
            output_dir / "humans_20241006_consolidated.jsonl", lines=True
        )
        multi = pd.read_json(output_dir / "humans_20241006_multi.jsonl", lines=True)

        assert len(consolidated) == 1
        assert len(multi) == 2
