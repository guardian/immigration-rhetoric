from unittest.mock import patch

import pandas as pd
from immigration_rhetoric.stage08_inference.run_inference import (
    parse_args,
    run_inference,
)
from tests.stage08_inference.test_classify_snippets import (
    RELEVANCE_CATS,
    SENTIMENT_CATS,
    FakeNlp,
)

MODULE = "immigration_rhetoric.stage08_inference.run_inference"


def write_snippets(input_dir, year, snippets, texts=None):
    pd.DataFrame(
        {
            "snippet_id": [f"{year}_{i}" for i in range(len(snippets))],
            "date": [f"{year}-01-01"] * len(snippets),
            "text": texts or ["speech"] * len(snippets),
            "snippet": snippets,
        }
    ).to_csv(input_dir / f"snippets_hansard_data_clean_{year}.csv", index=False)


def fake_load(path):
    return FakeNlp(RELEVANCE_CATS if path == "rel" else SENTIMENT_CATS)


def run(input_dir, output_dir, **kwargs):
    defaults = dict(
        relevance_model="rel",
        sentiment_model="sent",
        start_year=1919,
        end_year=2025,
        processes=1,
        snippet_col="snippet",
        gpu_id=-1,
    )
    run_inference(str(input_dir), str(output_dir), **{**defaults, **kwargs})


class TestParseArgs:
    def test_defaults(self):
        with patch("sys.argv", ["script.py", "-r", "rel", "-s", "sent"]):
            args = parse_args()

        assert args.input_dir.endswith("clean_snippets")
        assert args.start_year == 1919
        assert args.processes == 1
        assert args.snippet_col == "snippet"
        assert args.gpu_id == -1


class TestRunInference:
    @patch(f"{MODULE}.spacy.load", side_effect=fake_load)
    def test_writes_one_labelled_file_per_year_in_range(self, _, tmp_path):
        output_dir = tmp_path / "out"
        write_snippets(tmp_path, 2024, ["about", "not about"])
        write_snippets(tmp_path, 2025, ["about"])
        write_snippets(tmp_path, 2026, ["about"])

        run(tmp_path, output_dir, end_year=2025)

        assert sorted(p.name for p in output_dir.iterdir()) == [
            "labelled_snippets_2024.jsonl",
            "labelled_snippets_2025.jsonl",
        ]
        df = pd.read_json(output_dir / "labelled_snippets_2024.jsonl", lines=True)
        assert df["label"].to_list() == [
            "NEGATIVE_TOWARDS_IMMIGRATION",
            "NOT_ABOUT_IMMIGRATION",
        ]
        assert {
            "text",
            "date",
            "relevance_label_scores",
            "sentiment_label_scores",
        } <= set(df.columns)

    @patch(f"{MODULE}.spacy.load", side_effect=fake_load)
    def test_drops_snippets_without_speech_text(self, _, tmp_path):
        write_snippets(tmp_path, 2025, ["about", "not about"], texts=["speech", None])

        run(tmp_path, tmp_path / "out")

        df = pd.read_json(
            tmp_path / "out" / "labelled_snippets_2025.jsonl", lines=True, dtype=False
        )
        assert df["snippet_id"].to_list() == ["2025_0"]

    @patch(f"{MODULE}.spacy.require_gpu")
    @patch(f"{MODULE}.spacy.load", side_effect=fake_load)
    def test_gpu_only_required_when_requested(self, _, mock_require_gpu, tmp_path):
        run(tmp_path, tmp_path / "out")
        mock_require_gpu.assert_not_called()

        run(tmp_path, tmp_path / "out", gpu_id=0)
        mock_require_gpu.assert_called_once_with(0)
