import pytest
from immigration_rhetoric.stage05_llm_annotation.prompts import (
    clean_step2_response,
    convert_alphabetic_labels,
    highest_score_labels,
    prompt_step_1,
    prompt_step_2,
)


class TestPrompts:
    def test_prompt_step_1_includes_snippet(self):
        prompt = prompt_step_1("A snippet about borders.")
        assert "A snippet about borders." in prompt

    def test_prompt_step_2_includes_step_1_response(self):
        prompt = prompt_step_2("This expresses fear of immigrants.")
        assert "This expresses fear of immigrants." in prompt


class TestCleanStep2Response:
    def test_strips_markdown_and_parses_json(self):
        response = (
            '```json\n{"A": "low", "B": "high", "C": "low", "D": "very low"}\n```'
        )
        assert clean_step2_response(response) == {
            "A": "low",
            "B": "high",
            "C": "low",
            "D": "very low",
        }

    def test_parses_plain_json(self):
        response = (
            '{"A": "moderate", "B": "moderate", "C": "moderate", "D": "moderate"}'
        )
        assert clean_step2_response(response) == {
            "A": "moderate",
            "B": "moderate",
            "C": "moderate",
            "D": "moderate",
        }


class TestHighestScoreLabels:
    def test_returns_single_highest_label(self):
        labels = {"A": "low", "B": "very high", "C": "moderate", "D": "low"}
        assert highest_score_labels(labels) == ["B"]

    def test_returns_all_tied_highest_labels(self):
        labels = {"A": "high", "B": "high", "C": "low", "D": "low"}
        assert highest_score_labels(labels) == ["A", "B"]

    def test_raises_on_unrecognised_probability(self):
        with pytest.raises(ValueError):
            highest_score_labels({"A": "not a real probability"})


class TestConvertAlphabeticLabels:
    def test_maps_each_label(self):
        assert convert_alphabetic_labels(["A", "B", "C", "D"]) == [
            "POSITIVE_TOWARDS_IMMIGRATION",
            "NEGATIVE_TOWARDS_IMMIGRATION",
            "NEUTRAL_TOWARDS_IMMIGRATION",
            "NOT_ABOUT_IMMIGRATION",
        ]

    def test_empty_list(self):
        assert convert_alphabetic_labels([]) == []
