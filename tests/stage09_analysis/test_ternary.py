import pandas as pd
import pytest
from immigration_rhetoric.stage09_analysis.plot_config import TernaryStyleConfig
from immigration_rhetoric.stage09_analysis.ternary_plotter import TernaryPlotter
from immigration_rhetoric.stage09_analysis.ternary_processor import (
    TernaryDataProcessor,
)
from tests.stage09_analysis.conftest import NEG, NEU, NOT, POS

TEXTS = {
    POS: ["The welcoming of refugees", "welcoming workers", "valued refugees"],
    NEU: ["border statistics", "statistics on workers", "border refugees"],
    NEG: ["illegal crossings", "illegal workers", "crossings and crime"],
    NOT: ["ignored tax debate"],
}


@pytest.fixture
def df():
    rows = []
    for year in [2000, 2001]:
        for label, texts in TEXTS.items():
            for i, text in enumerate(texts):
                rows.append(
                    {
                        "speech_id": f"{year}_{label}_{i}",
                        "date": pd.Timestamp(f"{year}-06-01"),
                        "label": label,
                        "snippet_text": text,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def plotter():
    style = TernaryStyleConfig()
    style.FIGSIZE = (4, 3)
    style.DPI = 20
    return TernaryPlotter(style_config=style)


class TestTernaryDataProcessor:
    def test_filter_keeps_whole_speeches_mentioning_word(self):
        df = pd.DataFrame(
            {
                "speech_id": ["a", "a", "b", "c"],
                "snippet_text": ["On ASYLUM.", "Other", "asylums", None],
            }
        )

        filtered = TernaryDataProcessor.filter_to_speeches_mentioning(df, "asylum")

        assert filtered["snippet_text"].to_list() == ["On ASYLUM.", "Other"]

    @pytest.mark.parametrize(
        "n_docs, expected",
        [(100, (10, 0.95)), (5, (4, 0.95)), (2, (2, 1.0))],
    )
    def test_adjust_df_thresholds(self, n_docs, expected):
        assert TernaryDataProcessor._adjust_df_thresholds(n_docs, 10, 0.95) == expected

    def test_compute_scores(self, df):
        positive, neutral, negative, doc_freqs = TernaryDataProcessor().compute_scores(
            df, min_df=1
        )

        # Terms are stemmed, and stop words, short terms and irrelevant snippets are dropped
        assert set(positive) == set(neutral) == set(negative)
        assert {"welcom", "refuge", "worker", "illeg"} <= set(positive)
        assert not {"the", "tax", "ignor"} & set(positive)
        assert positive["welcom"] > 0 and negative["welcom"] == 0
        assert negative["illeg"] > 0 and neutral["illeg"] == 0
        assert doc_freqs["worker"] == 6

    def test_compute_scores_needs_sentiment_snippets(self, df):
        with pytest.raises(ValueError, match="No positive"):
            TernaryDataProcessor().compute_scores(df[df["label"] == NOT], min_df=1)


class TestTernaryPlotter:
    def test_saves_plot(self, plotter, df, tmp_path):
        save_path = tmp_path / "ternary.png"

        plotter.plot_term_sentiment(
            df, min_df=1, save_path=save_path, min_total_score=0
        )

        assert save_path.exists()

    def test_saves_plot_per_period_with_data(self, plotter, df, tmp_path):
        plotter.plot_term_sentiment(
            df,
            min_df=1,
            save_path=tmp_path / "ternary.png",
            periods=[(1990, 1999), (2000, 2000), (2001, 2010)],
            min_total_score=0,
        )

        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "ternary_2000-2000.png",
            "ternary_2001-2010.png",
        ]

    def test_skips_plot_without_terms_above_min_total_score(
        self, plotter, df, tmp_path
    ):
        save_path = tmp_path / "ternary.png"

        plotter.plot_term_sentiment(df, min_df=1, save_path=save_path)

        assert not save_path.exists()
