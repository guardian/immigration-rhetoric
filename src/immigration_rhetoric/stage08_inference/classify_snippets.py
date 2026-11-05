import json
import os
import sys
from typing import List, Tuple

import pandas as pd
from spacy.language import Language
from tqdm import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# When output is redirected to a file/log, tqdm can't overwrite its own progress line, so disable the bar.
TQDM_DISABLE = not sys.stderr.isatty()

RELEVANT_LABEL = "ABOUT_IMMIGRATION"
NOT_RELEVANT_LABEL = "NOT_ABOUT_IMMIGRATION"


class SnippetClassifier:
    """Labels snippets with a relevance model, then with a sentiment model for the snippets predicted relevant."""

    def __init__(
        self, relevance_nlp: Language, sentiment_nlp: Language, n_process: int = 1
    ):
        self.relevance_nlp = relevance_nlp
        self.sentiment_nlp = sentiment_nlp
        self.n_process = n_process

    def predict(
        self, nlp: Language, texts: List[str], desc: str
    ) -> Tuple[List[str], List[str]]:
        """Returns the highest scoring label and the json-encoded scores of every label, for each text."""
        labels, scores = [], []
        for doc in tqdm(
            nlp.pipe(texts, n_process=self.n_process),
            total=len(texts),
            desc=desc,
            disable=TQDM_DISABLE,
        ):
            labels.append(max(doc.cats, key=doc.cats.get))
            scores.append(json.dumps(doc.cats))
        return labels, scores

    def classify(self, df: pd.DataFrame, snippet_col: str = "snippet") -> pd.DataFrame:
        """Adds relevance, sentiment and final label columns to a dataframe of snippets.

        Sentiment columns are null for snippets predicted not relevant, and `label` is the sentiment label
        for relevant snippets, else NOT_ABOUT_IMMIGRATION.
        """
        df = df.copy()
        relevance_labels, relevance_scores = self.predict(
            self.relevance_nlp, df[snippet_col].to_list(), "Relevance"
        )
        df["relevance_label"] = relevance_labels
        df["relevance_label_scores"] = relevance_scores

        relevant = df["relevance_label"] == RELEVANT_LABEL
        sentiment_labels, sentiment_scores = self.predict(
            self.sentiment_nlp, df.loc[relevant, snippet_col].to_list(), "Sentiment"
        )
        df = df.join(
            pd.DataFrame(
                {
                    "sentiment_label": sentiment_labels,
                    "sentiment_label_scores": sentiment_scores,
                },
                index=df.index[relevant],
                dtype=object,
            )
        )

        df["label"] = df["sentiment_label"].where(relevant, NOT_RELEVANT_LABEL)
        return df
