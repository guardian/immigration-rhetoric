# stage05_llm_annotation

Annotate sampled snippets with sentiment towards immigration, using LLMs.

## What it does

A two-step prompt (`prompts.py`) is sent to the model for each snippet.

`batch_annotations.py` (`BatchAnnotations`) implements the batching/looping logic shared by both backends;
`bedrock_annotations.py`/`openai_annotations.py` implement the actual API calls (with exponential-backoff
retries) for AWS Bedrock (Claude, Llama, Mistral models) and OpenAI (GPT models) respectively.

`reshape_llm_annotations.py` converts a run's output into the Prodigy-style `text`/`accept`/`_annotator_id` rows
that `stage06_agreement` takes as input, dropping any snippet whose reply couldn't be parsed into labels.

Results are streamed to stdout one JSON line per snippet as they complete, so a run can be redirected to a
file and tailed (`tail -f`) without losing progress if interrupted partway through. We recommend writing it to
`data/annotations/llm_labelled_snippets/`, as in the examples below.

**Note:** the models specified in this code (e.g. `claude_sonnet_4`, `gpt4_1`) may no longer be available from the providers; update the model IDs before running.

## Input / Output

- **Input**: a CSV or JSONL file of snippets to annotate, with at least a `text` column (e.g. the JSONL
  output of `stage04_prepare_annotation_data`).
- **Output**: one JSON object per line on stdout, containing the data from each input row plus `{model_name}_step1_responses`, `{model_name}_step2_responses`, `{model_name}_labels`. Optionally also written to a combined
  `--output-file`.

## Run

```bash
mkdir -p data/annotations/llm_labelled_snippets

# AWS Bedrock models - requires AWS credentials configured (e.g. via `aws configure` or environment variables)
python -m immigration_rhetoric.stage05_llm_annotation.run_bedrock_annotations \
  --model-name claude_sonnet_4 \
  --input-file data/pipeline_output/snippet_samples/sample_on_year.jsonl \
  --chunk-size 10 \
  --max-processes 7 \
  1> data/annotations/llm_labelled_snippets/claude_sonnet_4.jsonl

# OpenAI models - requires OPENAI_API_KEY set in the environment
python -m immigration_rhetoric.stage05_llm_annotation.run_openai_annotations \
  --model-name gpt4_1 \
  --input-file data/pipeline_output/snippet_samples/sample_on_year.jsonl \
  --chunk-size 10 \
  --max-processes 7 \
  1> data/annotations/llm_labelled_snippets/gpt4_1.jsonl

# Reshape a run's output for stage06_agreement (writes data/annotations/raw_annotations/gpt4_1.jsonl by default)
python -m immigration_rhetoric.stage05_llm_annotation.reshape_llm_annotations \
  --model-name gpt4_1 \
  --input-file data/annotations/llm_labelled_snippets/gpt4_1.jsonl
```
