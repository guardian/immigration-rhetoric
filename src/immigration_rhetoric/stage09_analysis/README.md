# stage09_analysis

Measure sentiment towards immigration, and how often it is discussed, over time and by
party, and make the figures.

## What it does

`run_make_figures.py` makes every figure, using:

- `SentimentDataProcessor` (`sentiment_processor.py`) to group snippets (by year, party or trigger term)
  and `SentimentCalculator` (`sentiment_calculator.py`) to score each group. A group's **net sentiment** is
  (positive - negative) / (positive + neutral + negative) x 100, over its snippets about immigration.
- `SentimentPlotter` (`sentiment_plotter.py`) and `FramesPlotter` (`frames_plotter.py`) to plot them, styled by
  `PlotStyler` (`plot_styler.py`) and the settings in `plot_config.py` (parties, colours, figure sizes).
- `TernaryDataProcessor` (`ternary_processor.py`) and `TernaryPlotter` (`ternary_plotter.py`) to plot which
  terms are used in positive, neutral and negative snippets.

Details:

- **Count corrections**: sentiment over time is calculated from the *estimated true* label counts, not the
  predicted ones (see [Count corrections](#count-corrections)). Sentiment per trigger term is not corrected,
  since misclassification rates are likely to differ between terms. Prevalence isn't corrected either: the
  corrections adjust counts of snippets, so they can't say which speeches are about immigration.
- **Parties**: some party names are merged (e.g. `Labour Co-operative` into `Labour Party`), and the SNP (from
  1999-05-17, when the Scottish Parliament was established) and UKIP (1993-09-03 to 2019-01-01) are only shown
  while active. Their snippets outside these windows still count towards the overall sentiment.
- **Jackknife ribbons**: to show how much the results depend on the choice of trigger terms, the terms are
  shuffled (with a fixed seed) and split into groups of 1.5% of the terms. Each run drops one group, keeping only
  the speeches with a snippet that still contains one of the remaining terms, and recalculates the sentiment;
  the ribbon spans the lowest to highest value across the runs. Which snippets contain which term is cached
  in `--cache-dir`.
- **Prevalence**: the number of speeches with at least one snippet about immigration, or as a percentage of all
  speeches (the denominator comes from `all_clean_speeches.csv`).
- **Frames**: the share of snippets about immigration that use each of Card et al. (2015)'s frames, matching
  Porter-stemmed frame terms against the stemmed snippet text.
- **Ternary plots**: the snippet text is Porter-stemmed, and each term (excluding English stop words and terms
  of 3 characters or fewer) in at least `min_df` of the positive, neutral and negative snippets is scored by
  summing its TF-IDF weights over the snippets with each label. Terms with a total score above 5 are plotted
  at their shares of the three, coloured by the largest share. The 6 terms closest to each corner (with a share
  over 0.5) and the 6 remaining terms closest to the centre are labelled with the number of snippets they
  appear in. `min_df` is loosened when there are too few snippets for it. The labels are moved apart with
  adjustText for a fixed number of iterations, so the layout is the same on every run. The labels may be placed
  slightly differently from the paper's ternary plots, but the points, colours and labelled terms are the same.
  The ternary plots use matplotlib's default style rather than the `.mplstyle` files used for the other figures.

## Input / Output

- **Input**:
  - `stage08_inference`'s `labelled_snippets_{version}_{min_date}_{max_date}.jsonl`,
    `speeches_{version}_{min_date}_{max_date}.jsonl` and `all_clean_speeches.csv`.
  - The trigger terms used by `stage03_snippet_extraction` (`data/trigger_terms.csv` by default).
- **Output** (in `figures/` by default): 20 png figures and 1 pdf.

## Run

From the repository root, e.g. for the published dataset (speeches up to the end of 2025):

```bash
# Once: download the tokeniser used to match frames
uv run python -m nltk.downloader punkt_tab

uv run python -m immigration_rhetoric.stage09_analysis.run_make_figures \
  -l data/pipeline_output/analysis_datasets/labelled_snippets_v1.0.0_19190101_20251218.jsonl \
  -s data/pipeline_output/analysis_datasets/speeches_v1.0.0_19190101_20251218.jsonl
```

The jackknife ribbons and per-term sentiment search every snippet for every trigger term, so the first run is
slow; later runs on the same data reuse the cached term presence matrices.

## Count corrections

No classifier is perfect, so counting snippets by predicted label misstates the number with each true label: some
snippets predicted negative are really neutral, and so on. `DataConfig.COUNT_CORRECTIONS` (in `plot_config.py`)
holds, for each predicted label, the share of snippets with that prediction that have each true label (so each
predicted label's shares sum to 1). The estimated true count of each label in any group of snippets is then a
weighted sum of its predicted counts. For example, 61% of snippets predicted positive are truly positive, as
are 17% of those predicted neutral, 8% of those predicted negative and 1% of those predicted not about
immigration, so:

```
corrected positive = 0.61 x predicted positive + 0.17 x predicted neutral + 0.08 x predicted negative + 0.01 x predicted not about
```

The shares come from the published models' predictions on `stage07_model_training`'s holdout set, which the
models never saw during training. Only the holdout snippets chosen by random sampling were used: the snippets
added to the training data by hand as extra examples of negative sentiment were removed first, so that the
shares reflect the mix of snippets the models label across the whole archive. As the holdout files don't
record which snippets were added by hand, the shares are fixed in `plot_config.py` rather than recalculated.

## Figures

Figures starting `paper_` are those in the paper (or its supplementary information). Unless stated, each runs
from 1923 to 2025, by year. The net sentiment figures show all parties plus Labour, the Conservatives and
Reform UK.

| Figure | Shows |
| --- | --- |
| `paper_sentiment_over_time_events_jackknife_ribbons` | Net sentiment, with key events marked, jackknife ribbons and 1923-1947 shaded (the paper's main figure) |
| `paper_sentiment_over_time_jackknife_ribbons_high_res_no_title` (pdf) | The same, without a title |
| `paper_sentiment_over_time_uncorrected` | Overall net sentiment, corrected and uncorrected |
| `paper_sentiment_over_time_uncorrected_parties` | Net sentiment by party, corrected and uncorrected |
| `paper_terms_sentiment_abs` | Net sentiment of the snippets containing each trigger term (terms in at least 50 snippets) |
| `paper_prevalence_over_time_percent` | Speeches about immigration as a percentage of all speeches |
| `paper_prevalence_over_time_by_party` | Number of speeches about immigration, stacked by party (Conservative, Labour, Liberal Democrats, Reform UK, other) |
| `paper_prevalence_over_time_by_party_percent` | The same, as a percentage of all speeches |
| `paper_prevalence_over_time_by_party_percent_about` | The same, as a percentage of all speeches about immigration |
| `paper_frames_over_time` | Share of snippets using the crime and victims frames |
| `ternary_plot_overall` | Each term's shares of positive, neutral and negative score, from 1923 (terms in at least 50 snippets) |
| `ternary_plot_over_time_{start}-{end}` | The same, for each period in `DataConfig.CUSTOM_PERIODS` (1948-1961, 1962-1980, 1981-1991, 1992-2003, 2004-2025; terms in at least 10 snippets) |
| `ternary_plot_over_time_asylum_{start}-{end}` | The same, for the speeches with a snippet mentioning asylum (terms in at least 1 snippet) |
