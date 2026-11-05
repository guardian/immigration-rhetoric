# stage06_agreement

Consolidate multiple annotators' labels into training data, and measure
inter-annotator agreement (Krippendorff's alpha) between any combination of human/LLM annotators.

## What it does

`consolidate_annotations.py` (`ConsolidateAnnotations`) combines one or more annotators' raw Prodigy-style
annotation exports (as produced by human annotation via Prodigy, or `stage05_llm_annotation`'s LLM labelling
once converted with its `reshape_llm_annotations.py`) into two outputs per `label`:

- **`{label}_{date}_consolidated.jsonl`**: one row per snippet, keeping snippets where annotators agreed (or
  only one annotator labelled it) and dropping disagreements - suitable as training data for
  `stage07_model_training`.
- **`{label}_{date}_multi.jsonl`**: every row for snippets labelled by more than one annotator (not merged) -
  the required input shape for measuring agreement below.

`evaluate_agreement.py` (`AnnotatorAgreement`) then computes Krippendorff's alpha for any chosen subset of
annotators over a `_multi` file, via the [`krippendorff`](https://pypi.org/project/krippendorff/) package.

## Input / Output

- **Input**: `{input_dir}/{filename}.jsonl` raw annotation exports (`--input-dir` defaults to
  `data/annotations/raw_annotations`), each a JSONL of
  `{"text": ..., "accept": [label_or_empty], "_annotator_id": ...}` rows (Prodigy's export format).
  Repeated identical ratings from one annotator are deduplicated. Conflicting ratings from one annotator
  for the same `text` are reported as an error before any output files are written.
- **Output**: `{output_dir}/{label}_{date}_consolidated.jsonl` and `..._multi.jsonl` (`--output-dir`
  defaults to `data/annotations/processed_annotations`). `stage07_model_training` takes the `_consolidated` file
  as its input.

## Run

```bash
# Combine one or more annotators' raw exports into consolidated + multi-annotator files
python -m immigration_rhetoric.stage06_agreement.run_consolidation \
  --files human_annotations_A human_annotations_B llm_annotations_claude \
  --label human_and_llm \
  --date 20241006

# Measure agreement between chosen annotator groups over the resulting 'multi' file
python -m immigration_rhetoric.stage06_agreement.run_evaluate_agreement \
  --annotations-file human_and_llm_20241006_multi.jsonl \
  --annotator-group "humans=A,B,C" \
  --annotator-group "humans-and-llm=A,B,C,claude"
```
