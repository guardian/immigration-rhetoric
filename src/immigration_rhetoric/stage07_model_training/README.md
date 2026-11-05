# stage07_model_training

Train and evaluate two spaCy text classification models - one for
**relevance** (is a snippet about immigration?) and one for **sentiment** (positive/neutral/negative
towards immigration, for relevant snippets only).

## What it does

`build_training_data.py` (`TrainingDataBuilder`) converts labelled snippets into spaCy `DocBin` files: a fixed 10%
holdout set (kept out of every split), then one 80/20 train/eval split of the rest per random seed, each with
relevance and sentiment files.

Training itself uses spaCy's own CLI (no custom code). `evaluate_models.py` (`ModelEvaluator`) scores the two
trained models **combined** - relevance first, then sentiment for snippets predicted relevant - on each
split's eval set. `evaluate_models_over_time.py` (`PerformanceOverTimeEvaluator`) then plots how that combined
performance varies with the decade of each eval snippet.

`bayesian_analysis.py` (`PerformanceUncertaintyEstimator`) estimates each metric with a credible interval
across all splits (see [Bayesian analysis](#bayesian-analysis)). The published models are the ones trained on
the `rs17` split.

## Input / Output

- **Input**: `stage06_agreement`'s `{label}_{date}_consolidated.jsonl` - one record per snippet with `text`,
  `accept` (the agreed label(s)) and `snippet_metadata` (`snippet_id`, `date`). Relevance is
  `NOT_ABOUT_IMMIGRATION` if that label was accepted, else `ABOUT_IMMIGRATION`; sentiment is the accepted
  `POSITIVE_`/`NEUTRAL_`/`NEGATIVE_TOWARDS_IMMIGRATION` label, and snippets with more than one accepted label
  are left out of the sentiment files. Snippets with no accepted label are dropped.
- **Output**:
  - `corpus/`: `holdout_{relevance,sentiment}.spacy` and `rs{seed}_{train,eval}_{relevance,sentiment}.spacy`
    (each with a matching `.jsonl`).
  - `training/rs{seed}/{relevance,sentiment}/`: the trained `model-best` and its `metrics.json`.
  - `training/rs{seed}/combined/`: combined `metrics.json`, `predictions.jsonl`, confusion matrix pngs and
    `performance_over_time_overall_by_decade.png`.
  - `training/bayesian_summary.csv`: the Bayesian estimate of each metric across all splits.

## Run

> **Using the Zenodo archive:** the archive contains the `.spacy` files but not the annotations, so skip step 1 and the `build_training_data` command. After copying the files into `corpus/` (see the top-level README), run these for each seed instead of `all`:
>
> ```bash
> for seed in 17 23 31 43 59; do
>   for cmd in train_relevance evaluate_relevance train_sentiment evaluate_sentiment evaluate_combined evaluate_over_time; do
>     spacy project run $cmd src/immigration_rhetoric/stage07_model_training --vars.seed $seed
>   done
> done
> spacy project run bayesian_analysis src/immigration_rhetoric/stage07_model_training
> ```

This folder is a [spaCy project](https://spacy.io/usage/projects): [`project.yml`](project.yml) defines the
commands below, and running them creates `assets/`, `corpus/`, `training/` and `packages/` folders here
(all gitignored). Paths are given relative to the repository root.

```bash
# 1. Copy the consolidated annotations from stage06_agreement into the project
mkdir -p src/immigration_rhetoric/stage07_model_training/assets
cp data/annotations/processed_annotations/{label}_{date}_consolidated.jsonl \
  src/immigration_rhetoric/stage07_model_training/assets/consolidated.jsonl

# 2. Build the data, then train and evaluate both models on each split
for seed in 17 23 31 43 59; do
  spacy project run all src/immigration_rhetoric/stage07_model_training --vars.seed $seed
done

# 3. Estimate the uncertainty in each metric across the splits
spacy project run bayesian_analysis src/immigration_rhetoric/stage07_model_training

# 4. (Optional) package the rs17 models as installable wheels
spacy project run package src/immigration_rhetoric/stage07_model_training
```

| Command | Description |
| --- | --- |
| `build_training_data` | Build the holdout set and one train/eval split per seed as spaCy DocBin files |
| `train_relevance` | Train the relevance model |
| `evaluate_relevance` | Evaluate the relevance model on its eval split |
| `train_sentiment` | Train the sentiment model |
| `evaluate_sentiment` | Evaluate the sentiment model on its eval split |
| `evaluate_combined` | Evaluate the relevance and sentiment models combined, with confusion matrix plots |
| `evaluate_over_time` | Plot the combined models' performance by the decade of each eval snippet |
| `bayesian_analysis` | Estimate the uncertainty in model performance from the eval metrics of every seed's models |
| `package` | Package both models as installable wheels, with the combined metrics and confusion matrix |

The `all` workflow covers one split (`seed`), so step 2 runs it once per seed. Commands are skipped if their
inputs haven't changed since the last run (except `bayesian_analysis`, which always runs); add `--force` to
re-run.

> **Note:** spaCy only remembers the most recent run of each command, whatever its `seed`. So switching seeds
> counts as a change: `build_training_data` rebuilds every time (quick, and gives the same files), and
> running `all` again for a seed that was already trained **retrains it from scratch**, overwriting its
> models. For example, after step 2, `spacy project run all ... --vars.seed 17` retrains `rs17` even though
> nothing changed. To redo one step for an existing split, run that command on its own instead of `all`.

Settings are `vars` in `project.yml` and can be overridden per run, e.g. to train on a GPU (needs
`uv sync --group cuda`):

```bash
spacy project run all src/immigration_rhetoric/stage07_model_training --vars.gpu_id 0 --vars.seed 23
```

| Var | Default | Meaning |
| --- | --- | --- |
| `annotations` | `assets/consolidated.jsonl` | Input consolidated annotations |
| `seeds` | `17 23 31 43 59` | Seeds to build train/eval splits for, and to include in the Bayesian analysis |
| `seed` | `17` | Which split to train, evaluate and package |
| `gpu_id` | `-1` | GPU to use, `-1` for CPU (works, but training is much slower) |
| `version` | `1.0.0` | Version of the packaged models |

## Training config

Both models use the same config, [`configs/textcat.cfg`](configs/textcat.cfg) (a `roberta-base` transformer
feeding spaCy's `TextCatEnsemble`); only the train/eval corpus paths differ between them.

The published models were trained on a GPU with the seed set in the config, using the `rs17` split. GPU
training isn't bit-for-bit deterministic, so a retrained model will score close to, but not exactly the same
as, the published one.

## Bayesian analysis

A single train/eval split gives one score per metric, with no sense of how much it depends on which snippets
happened to land in the eval set. `bayesian_analysis` takes the eval scores of the models trained on each
split and estimates each metric's mean with a 95% highest density interval (priors in `PRIORS`, set from
earlier training runs).

| Metric | Source |
| --- | --- |
| `overall` | Combined evaluation, macro F |
| `relevance` | Relevance model's evaluation, macro F |
| `sentiment` | Sentiment model's evaluation, macro F |
| `positive`, `neutral`, `negative`, `not_about` | Combined evaluation, F of that label |

Sampling uses a fixed random seed, so re-running it on the same metrics gives the same result.

## Performance over time

`evaluate_over_time` reads the combined `predictions.jsonl` and plots, for each decade, the number of eval
snippets and how many were misclassified, against the F1 score, precision and recall (macro averages across the
four labels, calculated from all of the decade's snippets). It makes
`performance_over_time_overall_by_decade.png`, which is the paper's Figure A3.

It leaves out snippets before 1923, where the paper's figures start (`--min-year`, `0` keeps every snippet),
and starts the x-axis at the beginning of that decade (1920).
