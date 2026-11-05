# stage03_snippet_extraction

Clean speech text and extract immigration-related snippets around trigger terms.

## What it does

`clean_and_extract_snippets.py` (`SnippetExtractor`) — for each year of joined speeches:

1. Drops speeches with missing text, optionally drops speeches below `--min-word-count`, and optionally
   samples `--speeches-per-year` speeches at random.
2. Cleans the speech text: transliterates accented characters, strips CJK/Japanese punctuation, normalises
   quotation marks and hyphens, removes periods from acronyms, fixes punctuation spacing, collapses
   whitespace, and expands/removes titles (e.g. "hon." -> "honourable") and bracketed audience reactions.
3. Pre-filters to speeches that contain at least one trigger term.
4. Splits each remaining speech into sentences using spaCy's `senter` pipe, then extracts a snippet of
   sentences around each sentence containing a trigger term (2 sentences either side by default), avoiding
   overlapping snippets.
5. Explodes the extracted snippets into one row per snippet and writes them to CSV.

`run_extract_snippets.py` runs this across all years in parallel (one process per year) via a
`multiprocessing.Pool`.

## Input / Output

- **Input**: `data/pipeline_output/speeches/speeches_<year>.csv` (from `stage02_speaker_metadata`), and a trigger
  terms CSV with a `term` column (`data/trigger_terms.csv`).
- **Output**: `{output_dir}/clean_snippets/snippets_hansard_data_clean_<year>.csv`.

## Run

```bash
python -m immigration_rhetoric.stage03_snippet_extraction.run_extract_snippets \
  --output-dir data/pipeline_output \
  --trigger-file data/trigger_terms.csv \
  --min-word-count 10 \
  --num-processes 7
```
