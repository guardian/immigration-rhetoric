import pandas as pd
import pytest
from immigration_rhetoric.stage06_agreement.evaluate_agreement import AnnotatorAgreement


def _annotations_df(labels_by_annotator: dict) -> pd.DataFrame:
    """Build a multi-annotator annotations dataframe from {annotator_id: [label_per_snippet]}."""
    rows = []
    for annotator_id, labels in labels_by_annotator.items():
        for i, label in enumerate(labels):
            rows.append(
                {
                    "text": f"snippet {i}",
                    "accept": [label],
                    "_annotator_id": annotator_id,
                }
            )
    return pd.DataFrame(rows)


class TestGetAgreementScore:
    def test_perfect_agreement_gives_alpha_of_one(self):
        df = _annotations_df(
            {
                "A": [
                    "POSITIVE_TOWARDS_IMMIGRATION",
                    "NEGATIVE_TOWARDS_IMMIGRATION",
                    "NEUTRAL_TOWARDS_IMMIGRATION",
                ],
                "B": [
                    "POSITIVE_TOWARDS_IMMIGRATION",
                    "NEGATIVE_TOWARDS_IMMIGRATION",
                    "NEUTRAL_TOWARDS_IMMIGRATION",
                ],
            }
        )
        score = AnnotatorAgreement.get_agreement_score(df, ["A", "B"])
        assert score == pytest.approx(1.0)

    def test_disagreement_lowers_alpha(self):
        df = _annotations_df(
            {
                "A": [
                    "POSITIVE_TOWARDS_IMMIGRATION",
                    "NEGATIVE_TOWARDS_IMMIGRATION",
                    "NEUTRAL_TOWARDS_IMMIGRATION",
                ],
                "B": [
                    "POSITIVE_TOWARDS_IMMIGRATION",
                    "POSITIVE_TOWARDS_IMMIGRATION",
                    "NEUTRAL_TOWARDS_IMMIGRATION",
                ],
            }
        )
        score = AnnotatorAgreement.get_agreement_score(df, ["A", "B"])
        assert score < 1.0

    def test_only_considers_requested_annotators(self):
        df = _annotations_df(
            {
                "A": [
                    "POSITIVE_TOWARDS_IMMIGRATION",
                    "NEGATIVE_TOWARDS_IMMIGRATION",
                    "NEUTRAL_TOWARDS_IMMIGRATION",
                ],
                "B": [
                    "POSITIVE_TOWARDS_IMMIGRATION",
                    "NEGATIVE_TOWARDS_IMMIGRATION",
                    "NEUTRAL_TOWARDS_IMMIGRATION",
                ],
                "C": [
                    "NOT_ABOUT_IMMIGRATION",
                    "NOT_ABOUT_IMMIGRATION",
                    "NOT_ABOUT_IMMIGRATION",
                ],
            }
        )
        score = AnnotatorAgreement.get_agreement_score(df, ["A", "B"])
        assert score == pytest.approx(1.0)


class TestLoadAnnotations:
    def test_loads_jsonl_file(self, tmp_path):
        pd.DataFrame(
            {
                "text": ["snippet a"],
                "accept": [["POSITIVE_TOWARDS_IMMIGRATION"]],
                "_annotator_id": ["A"],
            }
        ).to_json(tmp_path / "multi.jsonl", orient="records", lines=True)

        agreement = AnnotatorAgreement(input_dir=tmp_path)
        df = agreement.load_annotations("multi.jsonl")

        assert len(df) == 1
        assert df["text"].iloc[0] == "snippet a"
