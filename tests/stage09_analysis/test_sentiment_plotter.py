import pandas as pd
import pytest
from immigration_rhetoric.stage09_analysis.plot_config import PlotConfig
from immigration_rhetoric.stage09_analysis.sentiment_plotter import SentimentPlotter
from tests.stage09_analysis.conftest import NEG, NEU, PARTIES, POS, TERMS

SMALL = PlotConfig(figsize=(4, 3), dpi=20)


@pytest.fixture
def plotter(corrections, tmp_path):
    return SentimentPlotter(corrections=corrections, cache_dir=tmp_path / "cache")


@pytest.fixture
def speeches(snippets):
    speeches = snippets.groupby("speech_id", as_index=False).agg(
        date=("date", "first"), party_label=("party_label", "first")
    )
    speeches["n_snippets_about"] = 1
    speeches["person_label"] = "An MP"
    return speeches


class TestPlotSentimentOverTime:
    def test_saves_plot_with_ribbons_and_overlays(self, plotter, snippets, tmp_path):
        save_path = tmp_path / "plot.png"

        plotter.plot_sentiment_over_time(
            df=snippets,
            title="Sentiment",
            parties_of_interest=PARTIES,
            events={"2001-01-01": "An event"},
            shaded_block={"start": "2000-01-01", "end": "2001-01-01"},
            show_ribbons=True,
            jackknife_trigger_terms=TERMS,
            config=SMALL,
            save_path=save_path,
        )

        assert save_path.exists()

    def test_saves_plot_with_uncorrected(self, plotter, snippets, tmp_path):
        save_path = tmp_path / "plot.png"

        plotter.plot_sentiment_over_time(
            df=snippets,
            title="Sentiment",
            parties_of_interest=PARTIES,
            include_uncorrected=True,
            config=SMALL,
            save_path=save_path,
        )

        assert save_path.exists()

    def test_saves_pdf_without_title(self, plotter, snippets, tmp_path):
        save_path = tmp_path / "plot.pdf"

        plotter.plot_sentiment_over_time(
            df=snippets,
            title=None,
            parties_of_interest=PARTIES,
            config=SMALL,
            save_path=save_path,
        )

        assert save_path.read_bytes().startswith(b"%PDF")

    def test_jackknife_needs_trigger_terms(self, plotter, snippets):
        with pytest.raises(ValueError, match="jackknife_trigger_terms"):
            plotter.plot_sentiment_over_time(
                df=snippets, title="Sentiment", show_ribbons=True
            )


class TestOtherPlots:
    def test_sentiment_per_term(self, plotter, tmp_path):
        save_path = tmp_path / "plot.png"
        df = pd.DataFrame(
            {
                "label": [POS, POS, NEG, NEU],
                "snippet_text": ["migrant", "migrant refugee", "asylum", "refugee"],
            }
        )

        plotter.plot_sentiment_per_term(
            df=df,
            title="Per term",
            x_label="Net sentiment",
            trigger_terms=TERMS,
            min_occurrences=1,
            figsize=(4, 3),
            config=SMALL,
            save_path=save_path,
        )

        assert save_path.exists()

    def test_prevalence(self, plotter, speeches, tmp_path):
        save_path = tmp_path / "plot.png"
        all_speeches = pd.concat([speeches, speeches.assign(speech_id="other")])

        plotter.plot_prevalence_over_time(
            df_trigger_speeches=speeches,
            df_all_speeches=all_speeches,
            title="Prevalence",
            parties_of_interest=PARTIES,
            percentage=True,
            config=SMALL,
            save_path=save_path,
        )

        assert save_path.exists()
