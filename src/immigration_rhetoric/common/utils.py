import re
from datetime import datetime


def extract_date(filename):
    """Searches for a regex of the form yyyy-mm-dd in a string."""
    match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    if match:
        return datetime.strptime(match.group(1), "%Y-%m-%d")
    return datetime.strptime("9999-01-01", "%Y-%m-%d")


def extract_numbers_as_int(string: str) -> int:
    """Extracts the first run of digits found in a string and returns it as an int."""
    numbers = re.findall(r"\d+", string)[0]
    return int(numbers)


def text_after_last_slash(text: str) -> str:
    """Extracts the substring after the last '/' in a string, e.g. to pull a numeric ID from a URL."""
    return text.split("/")[-1]
