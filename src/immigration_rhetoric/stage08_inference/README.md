# stage08_inference

Label every extracted snippet with the trained relevance and sentiment models, then
combine the results into the snippet-level and speech-level datasets used for the analysis.

## What it does

`classify_snippets.py` (`SnippetClassifier`) runs the **relevance** model over each snippet, then the
**sentiment** model over the snippets predicted `ABOUT_IMMIGRATION`. Each snippet gets:

| Column | Meaning |
| --- | --- |
| `relevance_label` | `ABOUT_IMMIGRATION` or `NOT_ABOUT_IMMIGRATION`, whichever scores highest |
| `relevance_label_scores` | Both relevance scores, as a json string |
| `sentiment_label` | `POSITIVE_`/`NEUTRAL_`/`NEGATIVE_TOWARDS_IMMIGRATION`, whichever scores highest (null if not relevant) |
| `sentiment_label_scores` | All three sentiment scores, as a json string (null if not relevant) |
| `label` | The final label: `sentiment_label` if relevant, else `NOT_ABOUT_IMMIGRATION` |

`combine_inference_outputs.py` (`InferenceOutputCombiner`) merges the yearly labelled snippets into one snippet
file and one speech file, leaving out any snippet whose speaker is unknown. Each speech has its snippet ids, the
number of snippets with each label, and a sentiment score.

`run_combine_speeches.py` writes every speech (not just those with snippets) with a reduced set of columns, for
analyses that need the total number of speeches.

## Input / Output

- **Input**:
  - `stage03_snippet_extraction`'s `clean_snippets/snippets_hansard_data_clean_{year}.csv`.
  - The trained relevance and sentiment models from `stage07_model_training`: either a `model-best` directory or
    the name of an installed model package.
  - `stage02_speaker_metadata`'s `speeches/speeches_{year}.csv`, for `run_combine_speeches.py`.
- **Output** (under `data/pipeline_output/` by default):
  - `inference/labelled_snippets_{year}.jsonl`: every input column plus the label columns above.
  - `analysis_datasets/labelled_snippets_{version}_{min_date}_{max_date}.jsonl`: one row per snippet, with its
    speech and speaker details and the label columns.
  - `analysis_datasets/speeches_{version}_{min_date}_{max_date}.jsonl`: one row per speech with at least one
    snippet, with its snippet ids, per-label snippet counts and sentiment score.
  - `analysis_datasets/clean_speeches/clean_speeches_{year}.csv` and `analysis_datasets/all_clean_speeches.csv`:
    every speech, with reduced columns.

## Run

From the repository root, e.g. for the published dataset (speeches up to the end of 2025):

```bash
# 1. Label the snippets (add --gpu-id 0 to use a GPU)
uv run python -m immigration_rhetoric.stage08_inference.run_inference \
  -r en_immigration_relevance -s en_immigration_sentiment --end-year 2025

# 2. Combine them into the snippet and speech datasets
uv run python -m immigration_rhetoric.stage08_inference.run_combine_inference_outputs \
  -v v1.0.0 --max-speech-date 20251218

# 3. Write the reduced speech files
uv run python -m immigration_rhetoric.stage08_inference.run_combine_speeches --end-year 2025
```

`-r`/`-s` take a path to a `model-best` directory or the name of a model package installed with `pip install`
(see `stage07_model_training`'s `package` command). `--min-speech-date`/`--max-speech-date` only set the output
file names; `--start-year`/`--end-year` select which years are labelled.
