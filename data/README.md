# data

Static, checked-in input data used by the pipeline (as opposed to `data/pipeline_output/`, which holds
generated/downloaded data and is not tracked in git).

- `trigger_terms.csv`: immigration-related trigger terms (including plural/variant forms) used by
  `stage03_snippet_extraction` to identify which sentences to extract snippets around.
