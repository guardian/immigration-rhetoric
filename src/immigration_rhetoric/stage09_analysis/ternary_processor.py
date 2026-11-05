"""Data processing for ternary plots of the sentiment of the terms used in snippets."""

import logging
import re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pandas as pd
from immigration_rhetoric.stage09_analysis.plot_config import DataConfig
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

TOKEN_PATTERN = r"(?u)\b\w\w+\b"  # Alphanumeric tokens, 2+ chars
MAX_DF = 0.95

_STEMMER = PorterStemmer()


@lru_cache(maxsize=100_000)
def _stem_word(word: str) -> str:
    return _STEMMER.stem(word)


class TernaryDataProcessor:
    """Scores how much each (stemmed) term is used in positive, neutral and negative snippets."""

    def __init__(self, data_config: Optional[DataConfig] = None):
        self.data_config = data_config or DataConfig()

    @staticmethod
    def filter_to_speeches_mentioning(df: pd.DataFrame, word: str) -> pd.DataFrame:
        """Keep every snippet from the speeches with a snippet that mentions `word` (case-insensitive)."""
        mentions = df["snippet_text"].str.contains(
            rf"\b{re.escape(word)}\b", case=False, na=False, regex=True
        )
        filtered = df[df["speech_id"].isin(df.loc[mentions, "speech_id"].unique())]
        logger.info(
            f"Kept the {filtered['speech_id'].nunique()} of {df['speech_id'].nunique()} speeches mentioning "
            f"'{word}' ({len(filtered)} of {len(df)} snippets)"
        )
        return filtered

    @staticmethod
    def _stem_text(text: str) -> str:
        return " ".join(_stem_word(word) for word in text.split())

    @staticmethod
    def _adjust_df_thresholds(
        n_docs: int, min_df: int, max_df: float
    ) -> Tuple[int, float]:
        """Loosen the TF-IDF document frequency thresholds so that small sets of snippets still have terms."""
        adjusted = False

        if n_docs < min_df + 1:
            min_df = max(1, min_df if n_docs == 0 else min(min_df, n_docs))
            adjusted = True

        if max_df * n_docs < min_df:
            if n_docs <= 2:
                max_df = 1.0
            else:
                min_df = max(1, int(max_df * n_docs))
            adjusted = True

        if adjusted:
            logger.info(
                f"Adjusted TF-IDF thresholds for {n_docs} snippets: min_df={min_df}, max_df={max_df}"
            )

        return min_df, max_df

    def compute_scores(
        self, df: pd.DataFrame, min_df: int
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, int]]:
        """Sum each stemmed term's TF-IDF scores over the positive, neutral and negative snippets.

        Terms of 3 characters or fewer are dropped. Also returns the number of snippets each term appears in.

        Args:
            df: DataFrame with 'label' and 'snippet_text' columns
            min_df: Minimum number of snippets a term must appear in, loosened for small sets of snippets

        Returns:
            Positive, neutral and negative scores by term, and snippet counts by term
        """

        def stemmed_snippets(label: str) -> List[str]:
            snippets = df.loc[df["label"] == label, "snippet_text"]
            return [
                self._stem_text(text)
                for text in snippets.dropna().astype(str)
                if text.strip()
            ]

        stemmed = {
            label: stemmed_snippets(label)
            for label in [
                self.data_config.POSITIVE_LABEL,
                self.data_config.NEGATIVE_LABEL,
                self.data_config.NEUTRAL_LABEL,
            ]
        }
        all_snippets = [text for texts in stemmed.values() for text in texts]
        if not all_snippets:
            raise ValueError("No positive, neutral or negative snippets to score")

        min_df, max_df = self._adjust_df_thresholds(len(all_snippets), min_df, MAX_DF)
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_df=max_df,
            min_df=min_df,
            token_pattern=TOKEN_PATTERN,
        )
        try:
            vectorizer.fit(all_snippets)
        except ValueError as error:
            raise ValueError(
                "No terms remain after stop-word and document-frequency filtering"
            ) from error
        terms = vectorizer.get_feature_names_out()

        def scores(label: str) -> Dict[str, float]:
            totals = vectorizer.transform(stemmed[label]).sum(axis=0).tolist()[0]
            return {term: s for term, s in zip(terms, totals) if len(term) > 3}

        token_re = re.compile(TOKEN_PATTERN)
        doc_freqs: Dict[str, int] = {}
        for text in all_snippets:
            for token in set(token_re.findall(text)):
                doc_freqs[token] = doc_freqs.get(token, 0) + 1

        return (
            scores(self.data_config.POSITIVE_LABEL),
            scores(self.data_config.NEUTRAL_LABEL),
            scores(self.data_config.NEGATIVE_LABEL),
            doc_freqs,
        )
