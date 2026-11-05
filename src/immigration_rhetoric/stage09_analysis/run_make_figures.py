import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from immigration_rhetoric.common.logging_utils import TqdmLoggingHandler
from immigration_rhetoric.config import DATA_DIR, ROOT_DIR, STYLES_DIR
from immigration_rhetoric.stage09_analysis.frames_plotter import FramesPlotter
from immigration_rhetoric.stage09_analysis.plot_config import (
    AnalysisConfig,
    DataConfig,
    PlotConfig,
)
from immigration_rhetoric.stage09_analysis.sentiment_plotter import SentimentPlotter
from immigration_rhetoric.stage09_analysis.ternary_plotter import TernaryPlotter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[TqdmLoggingHandler()],
)


def run_make_figures(
    labelled_snippets: str,
    speeches: str,
    all_speeches: str,
    trigger_file: str,
    output_dir: str,
    cache_dir: str,
) -> None:
    """Make every figure from the labelled snippets and speeches."""
    plt.style.use([STYLES_DIR / "base.mplstyle", STYLES_DIR / "immigration.mplstyle"])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_json(labelled_snippets, lines=True)

    # Load trigger terms (for sensitivity analysis plots)
    trigger_terms = pd.read_csv(trigger_file).term.to_list()

    plotter = SentimentPlotter(cache_dir=Path(cache_dir))
    paper_config = AnalysisConfig.create_paper_plots()

    # MAIN PLOT FOR THE PAPER -------------------------------------------------

    paper_parties = [
        "Labour Party",
        "Conservative Party",
        "Reform UK",
    ]

    paper_events = {
        "1948-01-01": "Windrush",
        "1962-01-01": "Commonwealth Immigration Act",
        "1981-01-01": "British Nationality Act",
        "1992-01-01": "Maastricht Treaty",
        "2004-05-01": "First Eastern EU enlargement",
    }

    paper_shaded_block = {
        "start": "1923-01-01",
        "end": "1947-12-31",
        "color": "#808080",
        "alpha": 0.22,
    }

    # Plot used in main paper
    plotter.plot_sentiment_over_time(
        df=df,
        title="Immigration Sentiment (Net %)",
        parties_of_interest=paper_parties,
        config=paper_config.plot,
        timeseries="year",
        events=paper_events,
        save_path=output_dir / "paper_sentiment_over_time_events_jackknife_ribbons.png",
        shaded_block=paper_shaded_block,
        show_ribbons=True,
        jackknife_trigger_terms=trigger_terms,
        jackknife_drop_percentage=1.5,
        ribbon_alpha=0.25,
    )

    # High-resolution version without a title
    plotter.plot_sentiment_over_time(
        df=df,
        title=None,
        parties_of_interest=paper_parties,
        config=PlotConfig(figsize=(20, 8), dpi=300, xlim=("1923", "2025")),
        timeseries="year",
        events=paper_events,
        save_path=output_dir
        / "paper_sentiment_over_time_events_jackknife_ribbons_high_res_no_title.pdf",
        shaded_block=paper_shaded_block,
        show_ribbons=True,
        jackknife_trigger_terms=trigger_terms,
        jackknife_drop_percentage=1.5,
        ribbon_alpha=0.25,
    )

    # PLOTS IN SUPPLEMENTARY INFORMATION --------------------------------------

    # Sentiment plots

    plotter.plot_sentiment_over_time(
        df=df,
        title="Immigration Sentiment (Net %), corrected and uncorrected",
        parties_of_interest=False,
        config=paper_config.plot,
        timeseries="year",
        include_uncorrected=True,
        save_path=output_dir / "paper_sentiment_over_time_uncorrected.png",
    )

    plotter.plot_sentiment_over_time(
        df=df,
        title="Immigration Sentiment (Net %), corrected and uncorrected by party",
        parties_of_interest=paper_parties,
        events=False,
        config=paper_config.plot,
        timeseries="year",
        include_uncorrected=True,
        save_path=output_dir / "paper_sentiment_over_time_uncorrected_parties.png",
    )

    plotter.plot_sentiment_per_term(
        df=df,
        title="Immigration Sentiment Per Term",
        x_label="Net sentiment of snippets containing each term (%)",
        trigger_terms=trigger_terms,
        text_col="snippet_text",
        min_occurrences=50,
        figsize=(20, 15),
        config=paper_config.plot,
        save_path=output_dir / "paper_terms_sentiment_abs.png",
    )

    # Load speech data for prevalence plots
    df_trigger_speeches = pd.read_json(speeches, lines=True)

    df_all_speeches = pd.read_csv(
        all_speeches,
        usecols=["speech_id", "party_label", "date"],
    )

    # Prevalence plots

    plotter.plot_prevalence_over_time(
        df_trigger_speeches=df_trigger_speeches,
        df_all_speeches=df_all_speeches,
        title="Proportion of all speeches about immigration (%)",
        config=paper_config.plot,
        timeseries="year",
        percentage=True,
        save_path=output_dir / "paper_prevalence_over_time_percent.png",
    )

    plotter.plot_prevalence_over_time(
        df_trigger_speeches=df_trigger_speeches,
        df_all_speeches=df_all_speeches,
        title="Frequency of immigration-related speeches by party",
        parties_of_interest=paper_config.data.get_default_parties(),
        config=paper_config.plot,
        timeseries="year",
        save_path=output_dir / "paper_prevalence_over_time_by_party.png",
    )

    plotter.plot_prevalence_over_time(
        df_trigger_speeches=df_trigger_speeches,
        df_all_speeches=df_all_speeches,
        title="Proportion of all speeches about immigration by party (%)",
        parties_of_interest=paper_config.data.get_default_parties(),
        config=paper_config.plot,
        timeseries="year",
        percentage=True,
        save_path=output_dir / "paper_prevalence_over_time_by_party_percent.png",
    )

    plotter.plot_prevalence_over_time(
        df_trigger_speeches=df_trigger_speeches,
        df_all_speeches=df_all_speeches,
        title="Proportion of immigration-related speeches by party (%)",
        parties_of_interest=paper_config.data.get_default_parties(),
        config=paper_config.plot,
        timeseries="year",
        percentage=True,
        per_about=True,
        save_path=output_dir / "paper_prevalence_over_time_by_party_percent_about.png",
    )

    # Plot Card et al. (2015)'s frames: crime and victims
    frames_plotter = FramesPlotter()

    frames_plotter.plot_frames(
        df=df,
        title="Yearly share of snippets by frame",
        frames_dict=paper_config.data.get_frames(),
        frames=["crime", "victims"],
        config=paper_config.plot,
        timeseries="year",
        save_path=output_dir / "paper_frames_over_time.png",
    )

    # Ternary plots of term sentiment, from 1923
    ternary_plotter = TernaryPlotter()
    df_since_1923 = df[df["date"] >= "1923-01-01"]

    ternary_plotter.plot_term_sentiment(
        df=df_since_1923,
        min_df=50,
        save_path=output_dir / "ternary_plot_overall.png",
    )

    ternary_plotter.plot_term_sentiment(
        df=df_since_1923,
        min_df=10,  # Fewer snippets per period, so a lower min_df
        save_path=output_dir / "ternary_plot_over_time.png",
        periods=DataConfig.get_custom_periods(),
    )

    ternary_plotter.plot_term_sentiment(
        df=ternary_plotter.processor.filter_to_speeches_mentioning(
            df_since_1923, "asylum"
        ),
        min_df=1,
        save_path=output_dir / "ternary_plot_over_time_asylum.png",
        periods=DataConfig.get_custom_periods(),
    )


def parse_args() -> Namespace:
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "-l",
        "--labelled-snippets",
        dest="labelled_snippets",
        help="Path to stage08_inference's labelled_snippets_{version}_{min_date}_{max_date}.jsonl",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-s",
        "--speeches",
        dest="speeches",
        help="Path to stage08_inference's speeches_{version}_{min_date}_{max_date}.jsonl",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-a",
        "--all-speeches",
        dest="all_speeches",
        help="Path to stage08_inference's all_clean_speeches.csv",
        type=str,
        default=str(DATA_DIR / "analysis_datasets" / "all_clean_speeches.csv"),
    )
    parser.add_argument(
        "-t",
        "--trigger-file",
        dest="trigger_file",
        help="CSV of trigger terms, with a 'term' column",
        type=str,
        default=str(ROOT_DIR / "data" / "trigger_terms.csv"),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        help="Directory to write the figures to",
        type=str,
        default=str(ROOT_DIR / "figures"),
    )
    parser.add_argument(
        "--cache-dir",
        dest="cache_dir",
        help="Directory to cache the sensitivity analysis' term presence matrices in",
        type=str,
        default=str(DATA_DIR / "term_presence_cache"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_make_figures(**vars(args))
