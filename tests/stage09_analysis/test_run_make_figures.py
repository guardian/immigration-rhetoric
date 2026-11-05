from pathlib import Path
from unittest.mock import patch

import pandas as pd
from immigration_rhetoric.stage09_analysis.run_make_figures import (
    parse_args,
    run_make_figures,
)
from tests.stage09_analysis.conftest import TERMS

MODULE = "immigration_rhetoric.stage09_analysis.run_make_figures"


class TestParseArgs:
    def test_defaults(self):
        with patch("sys.argv", ["script.py", "-l", "l", "-s", "s"]):
            args = parse_args()

        assert args.all_speeches.endswith("all_clean_speeches.csv")
        assert args.trigger_file.endswith("trigger_terms.csv")
        assert args.output_dir.endswith("figures")
        assert args.cache_dir.endswith("term_presence_cache")


class TestRunMakeFigures:
    @patch(f"{MODULE}.TernaryPlotter")
    @patch(f"{MODULE}.FramesPlotter")
    @patch(f"{MODULE}.SentimentPlotter")
    def test_makes_every_figure(
        self, sentiment_plotter, frames_plotter, ternary_plotter, snippets, tmp_path
    ):
        snippets_file = tmp_path / "snippets.jsonl"
        snippets.to_json(snippets_file, orient="records", lines=True)
        speeches_file = tmp_path / "speeches.jsonl"
        pd.DataFrame({"speech_id": ["a"]}).to_json(
            speeches_file, orient="records", lines=True
        )
        all_speeches_file = tmp_path / "all_speeches.csv"
        pd.DataFrame(
            columns=["speech_id", "party_label", "person_label", "date"]
        ).to_csv(all_speeches_file, index=False)
        trigger_file = tmp_path / "terms.csv"
        pd.DataFrame({"term": TERMS}).to_csv(trigger_file, index=False)
        output_dir = tmp_path / "figures"

        run_make_figures(
            labelled_snippets=str(snippets_file),
            speeches=str(speeches_file),
            all_speeches=str(all_speeches_file),
            trigger_file=str(trigger_file),
            output_dir=str(output_dir),
            cache_dir=str(tmp_path / "cache"),
        )

        sentiment_plotter.assert_called_once_with(cache_dir=tmp_path / "cache")
        plotter = sentiment_plotter.return_value
        calls = (
            plotter.plot_sentiment_over_time.call_args_list
            + plotter.plot_sentiment_per_term.call_args_list
            + plotter.plot_prevalence_over_time.call_args_list
            + frames_plotter.return_value.plot_frames.call_args_list
            + ternary_plotter.return_value.plot_term_sentiment.call_args_list
        )
        save_paths = {Path(c.kwargs["save_path"]) for c in calls}
        assert len(calls) == len(save_paths) == 13
        assert all(p.parent == output_dir for p in save_paths)
        jackknife_terms = [
            c.kwargs["jackknife_trigger_terms"]
            for c in plotter.plot_sentiment_over_time.call_args_list
            if c.kwargs.get("show_ribbons")
        ]
        assert jackknife_terms == [TERMS, TERMS]
