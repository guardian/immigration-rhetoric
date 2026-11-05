import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from immigration_rhetoric.common.logging_utils import TqdmLoggingHandler
from immigration_rhetoric.common.utils import extract_numbers_as_int

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[TqdmLoggingHandler()],
)


class SnippetSampler:
    """Samples an equal number of extracted snippets per year (or year-month), ready for manual annotation."""

    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_snippets(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Load every year's extracted-snippets csv within the given year range, adding `year`/`month` columns."""
        dataframes = []
        for file_path in sorted(
            self.input_dir.glob("snippets_hansard_data_clean_*.csv")
        ):
            year = extract_numbers_as_int(file_path.name)
            if start_year <= year <= end_year:
                dataframes.append(pd.read_csv(file_path, low_memory=False))

        df = pd.concat(dataframes, ignore_index=True)
        dates = pd.to_datetime(df["date"])
        df["year"] = dates.dt.year
        df["month"] = dates.dt.month
        return df

    @staticmethod
    def sample_by_column_value(
        df: pd.DataFrame, col: str, approx_total: int, random_state: int = 23
    ) -> pd.DataFrame:
        """Sample an equal number of rows per unique value of `col`, e.g. `col='year'` to sample evenly across specified time period."""
        n_distinct_col_values = df[col].nunique()
        if n_distinct_col_values == 0:
            raise ValueError(
                f"No rows left to sample from (no values of {col}); "
                "reconsider the exclusion list, which may have removed every snippet."
            )
        logging.info(f"Sampling on {col} ({n_distinct_col_values} unique value(s))")

        if approx_total > len(df):
            logging.info(
                f"Sample size ({approx_total}) larger than dataset size, reducing to {len(df)}"
            )
            approx_total = len(df)

        if approx_total < n_distinct_col_values:
            raise ValueError(
                f"Sample size ({approx_total}) must be at least the number of "
                f"unique {col} values ({n_distinct_col_values}) to sample every group."
            )

        n_to_sample = approx_total // n_distinct_col_values
        logging.info(f"Sampling {n_to_sample} row(s) for each value of {col}")

        samples = []
        for value in df[col].unique():
            subset = df[df[col] == value]
            samples.append(
                subset.sample(min(n_to_sample, len(subset)), random_state=random_state)
            )
        return pd.concat(samples, ignore_index=True)

    @staticmethod
    def exclude_already_sampled(
        df: pd.DataFrame, previous_sample: pd.DataFrame, text_col: str = "snippet"
    ) -> pd.DataFrame:
        """Drop rows whose `text_col` value already appears in a previous sample, to avoid re-annotating duplicates."""
        before = len(df)
        df = df[~df[text_col].isin(previous_sample[text_col])]
        logging.info(
            f"Excluded {before - len(df)} snippet(s) already present in the previous sample"
        )
        return df

    def output_sample(
        self,
        sample: pd.DataFrame,
        sample_col: str,
        meta_cols: Optional[List[str]] = None,
        metadata_cols: Optional[List[str]] = None,
    ) -> None:
        """Write the sample to CSV, and to a Prodigy-format JSONL with `text`/`meta`/`snippet_metadata` columns."""
        meta_cols = meta_cols if meta_cols is not None else ["qualifier"]
        metadata_cols = (
            metadata_cols if metadata_cols is not None else ["snippet_id", "date"]
        )

        csv_path = self.output_dir / f"sample_on_{sample_col}.csv"
        sample.to_csv(csv_path, index=False)

        annotation_df = pd.DataFrame(
            {
                "text": sample["snippet"],
                "meta": sample.apply(
                    lambda row: {c: row[c] for c in meta_cols}, axis=1
                ),
                "snippet_metadata": sample.apply(
                    lambda row: {c: row[c] for c in metadata_cols}, axis=1
                ),
            }
        )
        jsonl_path = self.output_dir / f"sample_on_{sample_col}.jsonl"
        annotation_df.to_json(jsonl_path, orient="records", lines=True)

        logging.info(
            f"Saved {len(sample)} sampled snippet(s) to {csv_path} and {jsonl_path}"
        )
