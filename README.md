# Measuring immigration rhetoric in the UK Parliament

GitHub repo: https://github.com/guardian/immigration-rhetoric

Replication code for the analysis of sentiment towards immigration in UK parliamentary speeches. The pipeline downloads Hansard debates, attaches speaker metadata, extracts immigration-related snippets, prepares and consolidates annotations, trains relevance and sentiment classifiers, applies them to the full archive of speeches, and reproduces the paper's figures.

## Repository contents

The Python package is in `src/immigration_rhetoric/`. The release workflow follows the package-stage names:

- [`stage01_scrape_speeches`](src/immigration_rhetoric/stage01_scrape_speeches/README.md): download and parse parliamentary debates.
- [`stage02_speaker_metadata`](src/immigration_rhetoric/stage02_speaker_metadata/README.md): download, clean, and join speaker metadata.
- [`stage03_snippet_extraction`](src/immigration_rhetoric/stage03_snippet_extraction/README.md): clean speeches and extract snippets around immigration trigger terms.
- [`stage04_prepare_annotation_data`](src/immigration_rhetoric/stage04_prepare_annotation_data/README.md): draw a stratified annotation sample.
- [`stage05_llm_annotation`](src/immigration_rhetoric/stage05_llm_annotation/README.md): optionally annotate snippets with OpenAI or AWS Bedrock models.
- [`stage06_agreement`](src/immigration_rhetoric/stage06_agreement/README.md): consolidate annotations and calculate inter-annotator agreement.
- [`stage07_model_training`](src/immigration_rhetoric/stage07_model_training/README.md): build training data, train the classifiers, evaluate them, and estimate uncertainty.
- [`stage08_inference`](src/immigration_rhetoric/stage08_inference/README.md): label every snippet with the trained models and build the snippet-level and speech-level datasets.
- [`stage09_analysis`](src/immigration_rhetoric/stage09_analysis/README.md): reproduce the paper's figures from the labelled datasets.

Each linked stage README documents its inputs, outputs, commands, and stage-specific options.

## System requirements

The release environment uses Python 3.12 because the version of spaCy used for this analysis is restricted to versions 3.7.2-3.7.x. The project has been developed and tested on macOS 15 or later on Apple Silicon; it has not been tested on Linux or Windows.

| Requirement | Details |
| --- | --- |
| Operating system | macOS 15+ (tested) |
| Python | 3.12.x only (`>=3.12,<3.13`) |
| CPU and memory | A 64-bit multi-core CPU and enough RAM for pandas data processing; 16 GB RAM is recommended for the complete local pipeline |
| Storage | Space for the Zenodo archive, the environment, downloaded XML, generated CSV/JSONL files, and trained models. A full rerun of stages 01-09 needs substantially more space than using the archive alone |
| Network | Internet access is required for stages 01-02 and installation; stage 05 also requires access to the selected model provider |
| GPU | Not required. GPU training is optional and requires an NVIDIA CUDA 12-compatible setup. CPU training works but is substantially slower. |

On Apple Silicon, the development dependency `thinc-apple-ops` is installed automatically.

## Software environment

Exact package versions are locked in `uv.lock`; direct dependencies and Python constraints are declared in `pyproject.toml`.

[`mise`](https://mise.jdx.dev/) is used to install the pinned tool versions:

- Python 3.12
- `uv` 0.12
- `pre-commit` 4.6

Install [`mise`](https://mise.jdx.dev/getting-started.html), then run from this directory:

```bash
mise install
uv sync
```

`uv sync` creates `.venv` and installs the locked runtime and development dependencies. Run all subsequent commands with `uv run` so that the locked environment is used:

```bash
uv run pytest tests/
```

Alternatively, activate the environment with `source .venv/bin/activate`.

To enable CUDA training, install the optional dependency group after the normal setup:

```bash
uv sync --group cuda
```

## Data availability and setup

Code, data and trained models are archived together on Zenodo under a single DOI: 10.5281/zenodo.23982091. The archive contains this repository (`immigration-rhetoric/`) and a `data/` folder alongside it:

```
README.md
immigration-rhetoric/    this repository
data/
├── training_data/       the datasets used when training the models (.spacy)
├── models/              the trained relevance and sentiment models (spaCy packages)
└── analysis_datasets/   snippets labelled by the model and speeches, used in the analysis
```

Copy or install the files into this repository as follows:

| Archive folder | Where it goes | Used by |
| --- | --- | --- |
| `data/training_data/` | `immigration-rhetoric/src/immigration_rhetoric/stage07_model_training/corpus/` | stage 07 |
| `data/models/` | `immigration-rhetoric/data/models/`, then install the wheels with `uv pip install data/models/*.whl` after `uv sync` | stage 08 |
| `data/analysis_datasets/` | `immigration-rhetoric/data/pipeline_output/analysis_datasets/` | stage 09 |

```bash
# From the directory containing the extracted archive
mkdir -p immigration-rhetoric/data/pipeline_output
cp -r data/analysis_datasets immigration-rhetoric/data/pipeline_output/
mkdir -p immigration-rhetoric/src/immigration_rhetoric/stage07_model_training/corpus
cp data/training_data/* immigration-rhetoric/src/immigration_rhetoric/stage07_model_training/corpus/
mkdir -p immigration-rhetoric/data/models
cp data/models/*.whl immigration-rhetoric/data/models/
cd immigration-rhetoric
# Run after `uv sync`: re-running `uv sync` removes packages not in the lockfile, including these
uv pip install data/models/*.whl
```

To reproduce the paper's figures, use the command in the [stage 09 README](src/immigration_rhetoric/stage09_analysis/README.md), passing the labelled snippets and speeches files in `analysis_datasets/` as `-l` and `-s`.

The archive does not include the output of stages 01-03. Stages 01 and 02 can regenerate the raw speeches from public sources, but this uses live TheyWorkForYou and Wikidata data, so the result may differ.

## Reproduction workflow

Run commands from the project root. The stage READMEs linked above contain the commands and their inputs and outputs.

| Stages | Purpose | When to run |
| --- | --- | --- |
| 01-04 | Regenerate the raw speeches, speaker metadata, snippets, and annotation sample from public sources. | Optional. Uses live data, so results may differ from the paper. |
| 05 | Produce new LLM annotations. | Optional. It requires provider credentials and may incur charges. |
| 06 | Consolidate annotation exports and measure agreement. | Only with your own annotations; the original raw annotations are not archived. |
| 07 | Train, evaluate, and estimate uncertainty for the classifiers. | Start from `training_data/` to reproduce the published models. |
| 08 | Label every snippet and build the analysis datasets. | Needs stage 03 output and the packaged models. |
| 09 | Reproduce the paper's figures. | Start from `analysis_datasets/`. |

By default, stages 01-03 write beneath `data/pipeline_output/`; use the stage-specific command options to choose another location. Stages 04-09 have their own documented input and output paths. The archive contains the inputs for stages 07 and 09, so these can be reproduced without running the earlier stages.

## Reproducibility notes

- Stages 01-02 download live data, which can change.
- GPU training is not bit-for-bit deterministic; retrained models score close to, but not exactly the same as, the published ones.
- Stage 05 LLM outputs are non-deterministic, and the models used to annotate them may no longer be available.

## Citation

If you use this code or data, please cite the Zenodo archive (DOI: 10.5281/zenodo.23982091) and the associated paper (TODO: paper DOI). Citation metadata is in `CITATION.cff`.

## License and attribution

- Code: licensed under the [MIT License](LICENSE).
- Datasets, training data and models: licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and archived on Zenodo.
- Parliamentary data: contains Parliamentary information licensed under the [Open Parliament Licence v3.0](https://www.parliament.uk/site-information/copyright-parliament/open-parliament-licence/).
- Speech text and speaker parsing: provided by [TheyWorkForYou](https://www.theyworkforyou.com/) (mySociety).
- Speaker metadata: derived from [Wikidata](https://www.wikidata.org/) (CC0).
