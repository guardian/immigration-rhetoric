# stage04_prepare_annotation_data

Sample extracted snippets and format them for manual annotation (e.g. in Prodigy).

## What it does

`sample_snippets.py` (`SnippetSampler`):

1. `load_snippets()` — loads the extracted snippets for a range of years.
2. `sample_by_column_value()` — samples an equal number of snippets per year (or year-month).
3. `exclude_already_sampled()` — optionally drops snippets from a previous sample.
4. `output_sample()` — writes the sample to CSV and Prodigy-format JSONL.

`run_snippet_sampling.py` runs this end-to-end over a range of years (1919 to current year by default).

## Input / Output

- **Input**: `data/pipeline_output/clean_snippets/snippets_hansard_data_clean_<year>.csv` (from `stage03_snippet_extraction`).
- **Output**: `{output_dir}/sample_on_<sample_col>.csv` and `{output_dir}/sample_on_<sample_col>.jsonl`.

## Run

```bash
python -m immigration_rhetoric.stage04_prepare_annotation_data.run_snippet_sampling \
  --input-dir data/pipeline_output/clean_snippets \
  --output-dir data/pipeline_output/snippet_samples \
  --number-to-sample 5000
```
