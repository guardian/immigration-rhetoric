from unittest.mock import patch

import pandas as pd
from immigration_rhetoric.stage08_inference.run_combine_speeches import (
    parse_args,
    run_combine_speeches,
    select_speech_columns,
)


def speeches(year, ids):
    return pd.DataFrame(
        {
            "person_id": [1] * len(ids),
            "id": ids,
            "speakername": ["An MP"] * len(ids),
            "text": [f"text of {i}" for i in ids],
            "date": [f"{year}-01-01"] * len(ids),
            "year": [year] * len(ids),
            "personLabel": ["An MP"] * len(ids),
            "partyLabel": ["A Party"] * len(ids),
            "constituencyLabel": ["Somewhere"] * len(ids),
        }
    )


class TestParseArgs:
    def test_defaults(self):
        with patch("sys.argv", ["script.py"]):
            args = parse_args()

        assert args.input_dir.endswith("speeches")
        assert args.start_year == 1919


class TestSelectSpeechColumns:
    def test_keeps_and_renames_columns(self):
        result = select_speech_columns(speeches(2020, ["a"]))

        assert result.columns.to_list() == [
            "speech_id",
            "speech_text",
            "constituency_label",
            "party_label",
            "person_label",
            "date",
        ]


class TestRunCombineSpeeches:
    def test_writes_yearly_and_combined_files_in_range(self, tmp_path):
        input_dir = tmp_path / "speeches"
        input_dir.mkdir()
        speeches(2024, ["a", "b"]).to_csv(input_dir / "speeches_2024.csv", index=False)
        speeches(2025, ["c"]).to_csv(input_dir / "speeches_2025.csv", index=False)
        speeches(2026, ["d"]).to_csv(input_dir / "speeches_2026.csv", index=False)

        run_combine_speeches(str(input_dir), str(tmp_path), 1919, 2025)

        assert sorted(p.name for p in (tmp_path / "clean_speeches").iterdir()) == [
            "clean_speeches_2024.csv",
            "clean_speeches_2025.csv",
        ]
        combined = pd.read_csv(tmp_path / "all_clean_speeches.csv")
        assert combined["speech_id"].to_list() == ["a", "b", "c"]
        assert combined["date"].to_list() == ["2024-01-01", "2024-01-01", "2025-01-01"]
