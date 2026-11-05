import json

import pandas as pd
import pytest
import spacy
from immigration_rhetoric.stage07_model_training.build_training_data import (
    RELEVANCE_LABELS,
    SENTIMENT_LABELS,
    TrainingDataBuilder,
)
from spacy.tokens import DocBin


def _annotation_records(n: int = 20) -> list[dict]:
    return [
        {
            "text": f"Snippet number {i} about something.",
            "accept": [
                (
                    "NEGATIVE_TOWARDS_IMMIGRATION"
                    if i % 2 == 0
                    else "NOT_ABOUT_IMMIGRATION"
                )
            ],
            "_annotator_id": "A",
            "snippet_metadata": {
                "snippet_id": f"s{i}",
                "date": f"2001-03-{(i % 28) + 1:02d}",
            },
        }
        for i in range(n)
    ]


def _write_jsonl(records: list[dict], path) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return str(path)


def _snippets_df(n: int = 20) -> pd.DataFrame:
    rows = []
    for i in range(n):
        about = i % 2 == 0
        rows.append(
            {
                "snippet_id": f"s{i}",
                "date": "2001-03-01",
                "text": f"Snippet number {i} about something.",
                "relevance_label": (
                    "ABOUT_IMMIGRATION" if about else "NOT_ABOUT_IMMIGRATION"
                ),
                "sentiment_label": "NEGATIVE_TOWARDS_IMMIGRATION" if about else None,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def builder(tmp_path):
    return TrainingDataBuilder(output_dir=str(tmp_path / "corpus"))


def _read_docs(path):
    return list(DocBin().from_disk(path).get_docs(spacy.blank("en").vocab))


class TestLoadSnippets:
    def test_takes_snippet_id_and_date_from_metadata(self, tmp_path):
        path = _write_jsonl(_annotation_records(2), tmp_path / "consolidated.jsonl")

        df = TrainingDataBuilder.load_snippets(path)

        assert df["snippet_id"].tolist() == ["s0", "s1"]
        assert df["date"].tolist() == ["2001-03-01", "2001-03-02"]

    def test_derives_relevance_and_sentiment_from_accept(self, tmp_path):
        records = _annotation_records(3)
        records[0]["accept"] = ["NEUTRAL_TOWARDS_IMMIGRATION"]
        records[1]["accept"] = ["NOT_ABOUT_IMMIGRATION"]
        records[2]["accept"] = [
            "POSITIVE_TOWARDS_IMMIGRATION",
            "NEUTRAL_TOWARDS_IMMIGRATION",
        ]
        path = _write_jsonl(records, tmp_path / "consolidated.jsonl")

        df = TrainingDataBuilder.load_snippets(path)

        assert df["relevance_label"].tolist() == [
            "ABOUT_IMMIGRATION",
            "NOT_ABOUT_IMMIGRATION",
            "ABOUT_IMMIGRATION",
        ]
        assert df["sentiment_label"].iloc[0] == "NEUTRAL_TOWARDS_IMMIGRATION"
        assert df["sentiment_label"].iloc[1:].isna().all()

    def test_drops_snippets_with_no_accepted_label(self, tmp_path):
        records = _annotation_records(3)
        records[1]["accept"] = []
        path = _write_jsonl(records, tmp_path / "consolidated.jsonl")

        df = TrainingDataBuilder.load_snippets(path)

        assert df["snippet_id"].tolist() == ["s0", "s2"]


class TestMakeDoc:
    def test_one_hot_cats_and_metadata(self, builder):
        doc = builder.make_doc(
            text="Some text",
            snippet_id="s1",
            date="2001-03-01",
            true_label="NEUTRAL_TOWARDS_IMMIGRATION",
            label_set=SENTIMENT_LABELS,
        )

        assert doc.cats == {
            "POSITIVE_TOWARDS_IMMIGRATION": 0.0,
            "NEUTRAL_TOWARDS_IMMIGRATION": 1.0,
            "NEGATIVE_TOWARDS_IMMIGRATION": 0.0,
        }
        assert doc._.snippet_id == "s1"
        assert doc._.date == "2001-03-01"


class TestSaveDocbinAndJsonl:
    def test_writes_matching_spacy_and_jsonl(self, builder):
        df = _snippets_df(4)
        prefix = builder.output_dir / "test_relevance"

        builder.save_docbin_and_jsonl(df, "relevance_label", RELEVANCE_LABELS, prefix)

        docs = _read_docs(prefix.with_suffix(".spacy"))
        records = [
            json.loads(line)
            for line in prefix.with_suffix(".jsonl").read_text().splitlines()
        ]
        assert len(docs) == len(records) == 4
        assert docs[0]._.snippet_id == "s0"
        assert records[1]["cats"] == {
            "ABOUT_IMMIGRATION": 0.0,
            "NOT_ABOUT_IMMIGRATION": 1.0,
        }
        assert records[1]["metadata"] == {"snippet_id": "s1", "date": "2001-03-01"}


class TestSaveRelevanceAndSentiment:
    def test_sentiment_only_includes_relevant_labelled_rows(self, builder):
        df = _snippets_df(6)

        builder.save_relevance_and_sentiment(df, "x")

        assert len(_read_docs(builder.output_dir / "x_relevance.spacy")) == 6
        assert len(_read_docs(builder.output_dir / "x_sentiment.spacy")) == 3

    def test_skips_sentiment_when_no_eligible_rows(self, builder):
        df = _snippets_df(6)
        df = df[df["relevance_label"] == "NOT_ABOUT_IMMIGRATION"]

        builder.save_relevance_and_sentiment(df, "x")

        assert (builder.output_dir / "x_relevance.spacy").exists()
        assert not (builder.output_dir / "x_sentiment.spacy").exists()


class TestBuild:
    def test_holdout_excluded_from_every_split(self, builder, tmp_path):
        path = _write_jsonl(_annotation_records(40), tmp_path / "consolidated.jsonl")

        builder.build(path, seeds=[17, 23])

        holdout_ids = {
            d._.snippet_id
            for d in _read_docs(builder.output_dir / "holdout_relevance.spacy")
        }
        assert len(holdout_ids) == 4
        for seed in [17, 23]:
            train = _read_docs(builder.output_dir / f"rs{seed}_train_relevance.spacy")
            eval_ = _read_docs(builder.output_dir / f"rs{seed}_eval_relevance.spacy")
            split_ids = {d._.snippet_id for d in train + eval_}
            assert len(train) + len(eval_) == 36
            assert not holdout_ids & split_ids
            assert (builder.output_dir / f"rs{seed}_train_sentiment.spacy").exists()

    def test_different_seeds_give_different_eval_sets(self, builder, tmp_path):
        path = _write_jsonl(_annotation_records(40), tmp_path / "consolidated.jsonl")

        builder.build(path, seeds=[17, 23])

        eval_17 = {
            d._.snippet_id
            for d in _read_docs(builder.output_dir / "rs17_eval_relevance.spacy")
        }
        eval_23 = {
            d._.snippet_id
            for d in _read_docs(builder.output_dir / "rs23_eval_relevance.spacy")
        }
        assert eval_17 != eval_23
