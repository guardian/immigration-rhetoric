# stage02_speaker_metadata

Normalise MPs' names and assign party affiliation, by joining raw speeches to speaker
metadata (TheyWorkForYou people data and Wikidata MP data).

## What it does

1. `download_speaker_metadata.py` (`SpeakerMetadataDownloader`, `WikidataQuerier`) — downloads
   TheyWorkForYou's `people.json`, then queries Wikidata for parliament term IDs and per-term MP records
   (party, constituency, dates), caching each file so re-runs skip already-downloaded data unless
   `--force-download` is set.
2. `preprocess_speaker_metadata.py` (`SpeakerMetadataPreprocessor`) — loads and cleans the downloaded people
   and Wikidata data ready for joining: resolves redirected membership IDs, extracts numeric IDs from
   `people.json` URLs, and normalises Wikidata party start/end dates.
3. `speeches_join_metadata.py` (`SpeechMetadataJoiner`) — joins raw speeches to the cleaned Wikidata MP
   records in three stages, in order of preference:
   - **Stage 1**: speeches with a `person_id` are joined directly to Wikidata.
   - **Stage 2**: speeches with only a `membership_id` are first joined to a `person_id` via the memberships
     data, then joined to Wikidata as in Stage 1.
   - **Stage 3**: remaining speeches are matched to Wikidata by cleaned speaker name.

## Input / Output

- **Input**: `data/pipeline_output/raw_speeches/raw_speeches_<year>.csv` (from `stage01_scrape_speeches`).
- **Output**: `{metadata_dir}/speaker_metadata/*` (downloaded people/Wikidata metadata) and
  `{output_dir}/speeches_<year>.csv` (speeches joined to speaker metadata).

## Run

```bash
python -m immigration_rhetoric.stage02_speaker_metadata.run_get_metadata \
  --output-dir data/pipeline_output

python -m immigration_rhetoric.stage02_speaker_metadata.run_speech_speaker_joins \
  --metadata-dir data/pipeline_output/speaker_metadata \
  --raw-speech-dir data/pipeline_output/raw_speeches \
  --output-dir data/pipeline_output/speeches
```
