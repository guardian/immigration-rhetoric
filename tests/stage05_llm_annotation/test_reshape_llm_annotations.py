import pandas as pd
from immigration_rhetoric.stage05_llm_annotation.reshape_llm_annotations import (
    reshape_llm_annotations,
)


class TestReshapeLlmAnnotations:
    def test_converts_labels_to_prodigy_style_rows(self):
        df = pd.DataFrame(
            {
                "text": ["snippet one", "snippet two"],
                "meta": [{"qualifier": "immigration"}, {}],
                "snippet_metadata": [{"snippet_id": "a"}, {"snippet_id": "b"}],
                "fake_labels": [
                    ["POSITIVE_TOWARDS_IMMIGRATION"],
                    ["NEUTRAL_TOWARDS_IMMIGRATION", "NEGATIVE_TOWARDS_IMMIGRATION"],
                ],
                "fake_step1_responses": ["x", "y"],
            }
        )

        result = reshape_llm_annotations(df, "fake")

        assert result.columns.to_list() == [
            "text",
            "meta",
            "snippet_metadata",
            "accept",
            "answer",
            "_annotator_id",
        ]
        assert result["accept"].to_list() == df["fake_labels"].to_list()
        assert result["answer"].unique().tolist() == ["accept"]
        assert result["_annotator_id"].unique().tolist() == ["fake"]

    def test_drops_rows_without_parseable_labels(self):
        df = pd.DataFrame(
            {
                "text": ["good", "unparseable reply", "errored"],
                "fake_labels": [["NOT_ABOUT_IMMIGRATION"], "not json at all", "Error"],
            }
        )

        result = reshape_llm_annotations(df, "fake")

        assert result["text"].to_list() == ["good"]
