import json

import pandas as pd
import pytest
from immigration_rhetoric.stage04_prepare_annotation_data.sample_snippets import (
    SnippetSampler,
)


@pytest.fixture
def sampler(tmp_path):
    input_dir = tmp_path / "clean_snippets"
    input_dir.mkdir()
    return SnippetSampler(input_dir=input_dir, output_dir=tmp_path / "samples")


class TestLoadSnippets:
    def test_loads_only_files_within_year_range_and_adds_year_month(self, sampler):
        pd.DataFrame({"snippet": ["a"], "date": ["1985-03-01"]}).to_csv(
            sampler.input_dir / "snippets_hansard_data_clean_1985.csv", index=False
        )
        pd.DataFrame({"snippet": ["b"], "date": ["1986-06-15"]}).to_csv(
            sampler.input_dir / "snippets_hansard_data_clean_1986.csv", index=False
        )

        result = sampler.load_snippets(start_year=1986, end_year=1986)

        assert list(result["snippet"]) == ["b"]
        assert list(result["year"]) == [1986]
        assert list(result["month"]) == [6]


class TestSampleByColumnValue:
    def test_samples_evenly_across_unique_values(self):
        df = pd.DataFrame({"year": [1986] * 10 + [1987] * 2})

        result = SnippetSampler.sample_by_column_value(df, "year", approx_total=4)

        assert result["year"].value_counts().to_dict() == {1986: 2, 1987: 2}

    def test_reduces_sample_size_if_larger_than_dataset(self):
        df = pd.DataFrame({"year": [1986, 1986, 1986]})

        result = SnippetSampler.sample_by_column_value(df, "year", approx_total=100)

        assert len(result) == 3

    def test_reproducible_with_fixed_random_state(self):
        df = pd.DataFrame({"year": [1986] * 10, "id": range(10)})

        first = SnippetSampler.sample_by_column_value(df, "year", approx_total=3)
        second = SnippetSampler.sample_by_column_value(df, "year", approx_total=3)

        assert list(first["id"]) == list(second["id"])


class TestExcludeAlreadySampled:
    def test_drops_snippets_present_in_previous_sample(self):
        df = pd.DataFrame({"snippet": ["a", "b", "c"]})
        previous_sample = pd.DataFrame({"snippet": ["b"]})

        result = SnippetSampler.exclude_already_sampled(df, previous_sample)

        assert list(result["snippet"]) == ["a", "c"]


class TestOutputSample:
    def test_writes_csv_and_annotation_jsonl(self, sampler):
        sample = pd.DataFrame(
            {
                "snippet": ["about asylum policy"],
                "qualifier": ["asylum"],
                "snippet_id": ["1_1"],
                "date": ["1986-01-01"],
                "year": [1986],
            }
        )

        sampler.output_sample(sample, sample_col="year")

        csv_path = sampler.output_dir / "sample_on_year.csv"
        jsonl_path = sampler.output_dir / "sample_on_year.jsonl"
        assert csv_path.exists()
        assert jsonl_path.exists()

        row = json.loads(jsonl_path.read_text().splitlines()[0])
        assert row["text"] == "about asylum policy"
        assert row["meta"] == {"qualifier": "asylum"}
        assert row["snippet_metadata"] == {"snippet_id": "1_1", "date": "1986-01-01"}
