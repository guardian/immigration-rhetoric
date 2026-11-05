# stage01_scrape_speeches

Download Hansard House of Commons debate data from TheyWorkForYou, and extract individual speeches from the raw debate XML.

## What it does

1. `download_debates.py` (`DebatesDownloader`) — downloads parliamentary debate XML files from `https://www.theyworkforyou.com/pwdata/scrapedxml/debates/` for a given date range, skipping files already present unless `--force-download` is set.
2. `process_speeches.py` (`SpeechesProcessor`) — parses the downloaded XML into structured per-speech records, cleans invalid/short/duplicate speeches, and writes one CSV per year.

## Input / Output

- **Input**: none (downloads directly from TheyWorkForYou).
- **Output**: `{output_dir}/xml_files/*.xml` (raw debate XML) and `{output_dir}/raw_speeches/raw_speeches_<year>.csv` (per-year speech records).

## Run

```bash
uv run python -m immigration_rhetoric.stage01_scrape_speeches.run_get_speeches \
  --start-date 1919-02-04 --end-date 2024-01-01 \
  --output-dir data/pipeline_output
```
