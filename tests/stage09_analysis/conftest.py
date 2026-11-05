import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest

matplotlib.use("Agg")

POS = "POSITIVE_TOWARDS_IMMIGRATION"
NEU = "NEUTRAL_TOWARDS_IMMIGRATION"
NEG = "NEGATIVE_TOWARDS_IMMIGRATION"
NOT = "NOT_ABOUT_IMMIGRATION"
LABELS = [POS, NEU, NEG, NOT]

PARTIES = ["Labour Party", "Conservative Party", "Reform UK"]
TERMS = ["migrant", "asylum", "refugee"]


def identity_corrections():
    return {pred: {true: float(pred == true) for true in LABELS} for pred in LABELS}


@pytest.fixture
def corrections():
    return identity_corrections()


@pytest.fixture
def snippets():
    """Five snippets per party per year, 2000-2003, each containing every term in TERMS."""
    rows = []
    for year in range(2000, 2004):
        for party in PARTIES:
            for i, label in enumerate([POS, NEU, NEG, NEG, NOT]):
                rows.append(
                    {
                        "snippet_id": f"{year}_{party}_{i}",
                        "speech_id": f"{year}_{party}_{i // 2}",
                        "date": f"{year}-0{i + 1}-15",
                        "party_label": party,
                        "label": label,
                        "relevance_label": (
                            NOT if label == NOT else "ABOUT_IMMIGRATION"
                        ),
                        "snippet_text": "The migrant, asylum and refugee crime victims of Europe",
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")
