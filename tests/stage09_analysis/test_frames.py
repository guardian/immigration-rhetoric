import pandas as pd
import pytest
from immigration_rhetoric.stage09_analysis.frames_plotter import FramesPlotter
from immigration_rhetoric.stage09_analysis.frames_processor import FramesDataProcessor
from immigration_rhetoric.stage09_analysis.plot_config import DataConfig, PlotConfig

FRAMES_DICT = {
    "crime": ["arrest (n), criminal (a),", "smuggler (n)"],
    "victims": ["refugee (n), victim (n)"],
}


@pytest.fixture
def processor():
    processor = FramesDataProcessor()
    processor._tokenize = str.split
    return processor


class TestCleanFramesDict:
    def test_removes_parts_of_speech_and_splits_terms(self, processor):
        assert processor.clean_frames_dict(FRAMES_DICT) == {
            "crime": ["arrest", "criminal", "smuggler"],
            "victims": ["refugee", "victim"],
        }

    def test_all_card_frames_parse(self, processor):
        cleaned = processor.clean_frames_dict(DataConfig.get_frames())

        assert all(cleaned.values())
        assert not any("(" in t for terms in cleaned.values() for t in terms)


class TestFramesShareOverTime:
    def test_percentage_of_relevant_snippets_with_each_frame(self, processor):
        df = pd.DataFrame(
            {
                "date": ["2000-01-01", "2000-06-01", "2000-07-01", "2001-01-01"],
                "relevance_label": ["ABOUT_IMMIGRATION"] * 3
                + ["NOT_ABOUT_IMMIGRATION"],
                "snippet_text": [
                    "Smugglers arrested",
                    "refugees and victims",
                    "nothing here",
                    "criminal victims",
                ],
            }
        )

        result = processor.frames_share_over_time(
            df,
            frames=["crime", "victims"],
            all_frames=processor.clean_frames_dict(FRAMES_DICT),
        )

        assert result["Frame"].tolist() == ["crime", "victims"]
        assert result["Percentage"].tolist() == pytest.approx([100 / 3, 100 / 3])

    def test_missing_frame_has_zero_share(self, processor):
        df = pd.DataFrame({"date": ["2000-01-01"], "snippet_text": ["refugee"]})

        result = processor.frames_share_over_time(
            df, frames=["economic"], all_frames={}
        )

        assert result["Percentage"].tolist() == [0]


class TestPlotFrames:
    def test_saves_plot(self, snippets, tmp_path):
        plotter = FramesPlotter()
        plotter.processor._tokenize = str.split
        save_path = tmp_path / "frames.png"

        plotter.plot_frames(
            df=snippets,
            title="Frames",
            frames_dict=DataConfig.get_frames(),
            frames=["crime", "victims"],
            config=PlotConfig(figsize=(4, 3), dpi=20),
            save_path=save_path,
        )

        assert save_path.exists()
