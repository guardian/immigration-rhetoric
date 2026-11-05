import pandas as pd
import pytest
from immigration_rhetoric.stage03_snippet_extraction.clean_and_extract_snippets import (
    SnippetExtractor,
)


@pytest.fixture
def extractor(tmp_path):
    return SnippetExtractor(
        output_dir=tmp_path, trigger_terms=["asylum", "refugee", "refugees"]
    )


class FakeSentence:
    def __init__(self, text):
        self.text = text


class FakeDoc:
    def __init__(self, sentences):
        self.sents = [FakeSentence(s) for s in sentences]


class TestCleanText:
    def test_transliterates_accented_characters(self, extractor):
        assert extractor.clean_text("café") == "cafe"

    def test_standardises_hyphens(self, extractor):
        assert "‐" not in extractor.clean_text("a‐b")

    def test_removes_periods_from_acronyms(self, extractor):
        assert extractor.clean_text("N.A.T.O. is here.").strip() == "NATO is here."

    def test_expands_hon_and_strips_audience_reactions(self, extractor):
        result = extractor.clean_text(
            "The hon. Member said [hon. members: shame] this."
        )
        assert "honourable Member said" in result
        assert "[" not in result

    def test_fixes_punctuation_spacing_without_touching_numbers(self, extractor):
        assert (
            extractor.clean_text("Hello!This costs 3.14 pounds.").strip()
            == "Hello! This costs 3.14 pounds."
        )


class TestTriggerInText:
    def test_matches_trigger_term(self, extractor):
        assert extractor.trigger_in_text("this is about asylum seekers") is True

    def test_no_match_returns_false(self, extractor):
        assert extractor.trigger_in_text("this is about something else") is False


class TestFilterCleanSpeechData:
    def test_drops_missing_text(self, extractor):
        df = pd.DataFrame({"id": [1, 2], "text": ["about asylum policy", None]})

        result = extractor.filter_clean_speech_data(df, "1986")

        assert list(result["id"]) == [1]

    def test_filters_by_min_word_count(self, tmp_path):
        extractor = SnippetExtractor(
            output_dir=tmp_path, trigger_terms=["asylum"], min_word_count=5
        )
        df = pd.DataFrame(
            {"id": [1, 2], "text": ["asylum asylum asylum asylum asylum", "asylum"]}
        )

        result = extractor.filter_clean_speech_data(df, "1986")

        assert list(result["id"]) == [1]

    def test_drops_speeches_without_a_trigger_term(self, extractor):
        df = pd.DataFrame(
            {"id": [1, 2], "text": ["about asylum policy", "about something else"]}
        )

        result = extractor.filter_clean_speech_data(df, "1986")

        assert list(result["id"]) == [1]

    def test_samples_down_to_speeches_per_year(self, tmp_path):
        extractor = SnippetExtractor(
            output_dir=tmp_path, trigger_terms=["asylum"], speeches_per_year=1
        )
        df = pd.DataFrame(
            {"id": [1, 2], "text": ["about asylum policy", "more asylum talk"]}
        )

        result = extractor.filter_clean_speech_data(df, "1986")

        assert len(result) == 1


class TestExtractSnippets:
    def test_extracts_snippet_around_trigger_sentence(self, extractor):
        sentences = ["First sentence.", "Talk about asylum.", "Third sentence."]

        snippets = extractor.extract_snippets(
            sentences, sentences_around_trigger=1, allowed_overlap=0
        )

        assert len(snippets) == 1
        assert snippets[0]["snippet"] == " ".join(sentences)
        assert snippets[0]["qualifier"] == "asylum"
        assert snippets[0]["trigger_list"] == ["asylum"]
        assert snippets[0]["num_triggers"] == 1

    def test_no_trigger_produces_no_snippets(self, extractor):
        sentences = ["First sentence.", "Second sentence."]

        snippets = extractor.extract_snippets(
            sentences, sentences_around_trigger=1, allowed_overlap=0
        )

        assert snippets == []

    def test_does_not_reuse_sentences_across_snippets(self, extractor):
        sentences = [
            "About asylum here.",
            "About asylum again.",
            "About asylum once more.",
        ]

        snippets = extractor.extract_snippets(
            sentences, sentences_around_trigger=1, allowed_overlap=0
        )

        # each snippet's sentence range shouldn't overlap with the previous one
        assert len(snippets) == 2


class TestSnippetExtract:
    def test_writes_snippets_csv(self, extractor, tmp_path):
        df = pd.DataFrame(
            {
                "id": ["s1"],
                "date": ["1986-01-13"],
                "text": ["Talk about asylum here."],
            }
        )

        class FakeNlp:
            @staticmethod
            def pipe(docs):
                return [FakeDoc(["Talk about asylum here."]) for _ in docs]

        extractor.snippet_extract(df, "1986", FakeNlp())

        output_path = (
            extractor.clean_snippets_dir / "snippets_hansard_data_clean_1986.csv"
        )
        assert output_path.exists()

        result = pd.read_csv(output_path)
        assert len(result) == 1
        assert result.iloc[0]["snippet_id"] == "s1_1"
        assert "asylum" in result.iloc[0]["snippet"]

    def test_returns_speeches_and_snippets_counts(self, extractor):
        df = pd.DataFrame(
            {
                "id": ["s1"],
                "date": ["1986-01-13"],
                "text": ["Talk about asylum here."],
            }
        )

        class FakeNlp:
            @staticmethod
            def pipe(docs):
                return [FakeDoc(["Talk about asylum here."]) for _ in docs]

        result = extractor.snippet_extract(df, "1986", FakeNlp())

        assert result == (1, 1)

    def test_writes_empty_csv_when_no_speeches_remain(self, extractor):
        df = pd.DataFrame(
            {"id": ["s1"], "date": ["1986-01-13"], "text": ["nothing relevant"]}
        )

        class FakeNlp:
            @staticmethod
            def pipe(docs):
                return [FakeDoc(["nothing relevant"]) for _ in docs]

        extractor.snippet_extract(df, "1986", FakeNlp())

        output_path = (
            extractor.clean_snippets_dir / "snippets_hansard_data_clean_1986.csv"
        )
        result = pd.read_csv(output_path)
        assert result.empty
        assert list(result.columns) == [
            "id",
            "date",
            "speaker_id",
            "snippet_id",
            "snippet",
            "text",
        ]
