from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
DATA_DIR = ROOT_DIR / "data" / "pipeline_output"
ANNOTATIONS_DIR = ROOT_DIR / "data" / "annotations"
STYLES_DIR = ROOT_DIR / "styles"

DATE_TODAY = datetime.today().strftime("%Y-%m-%d")
THIS_YEAR = datetime.today().year
