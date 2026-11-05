import logging
import re
from ast import literal_eval
from pathlib import Path
from typing import List, Optional

import pandas as pd
from immigration_rhetoric.common.logging_utils import TqdmLoggingHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[TqdmLoggingHandler()],
)


class SnippetExtractor:
    """Cleans speech text and extracts snippets of sentences around immigration-related trigger terms."""

    def __init__(
        self,
        output_dir: str,
        trigger_terms: List[str],
        min_word_count: int = 0,
        speeches_per_year: Optional[int] = None,
    ):
        self.output_dir = Path(output_dir)
        self.speech_dir = self.output_dir / "speeches"
        self.clean_snippets_dir = self.output_dir / "clean_snippets"
        self.clean_snippets_dir.mkdir(parents=True, exist_ok=True)

        self.trigger_terms = trigger_terms
        self.min_word_count = min_word_count
        self.speeches_per_year = speeches_per_year

    @staticmethod
    def _remove_special_characters(txt: str) -> str:
        """Transliterate accented characters to their Latin equivalents, and remove Asian script characters."""
        # Lowercase accents
        txt = re.sub("[àáâãäåā]", "a", txt)
        txt = re.sub("[èéêëēėę]", "e", txt)
        txt = re.sub("[ìíîïıīįi̇]", "i", txt)
        txt = re.sub("[òóôõöøō]", "o", txt)
        txt = re.sub("[ùúûüū]", "u", txt)
        txt = re.sub("[ýÿ]", "y", txt)
        txt = re.sub("[çć]", "c", txt)
        txt = re.sub("č", "ch", txt)
        txt = re.sub("ğ", "g", txt)
        txt = re.sub("[ñń]", "n", txt)
        txt = re.sub("ś", "s", txt)
        txt = re.sub("[şš]", "sh", txt)
        txt = re.sub("æ", "ae", txt)
        txt = re.sub("œ", "oe", txt)
        txt = re.sub("ł", "l", txt)
        txt = re.sub("[žźż]", "z", txt)
        txt = re.sub("ð", "d", txt)

        # Uppercase accents
        txt = re.sub("[ÀÁÂÃÄÅĀ]", "A", txt)
        txt = re.sub("[ÈÉÊËĒĖĘ]", "E", txt)
        txt = re.sub("[ÌÍÎÏİĪĮ]", "I", txt)
        txt = re.sub("[ÒÓÔÕÖØŌ]", "O", txt)
        txt = re.sub("[ÙÚÛÜŪ]", "U", txt)
        txt = re.sub("[ÝŸ]", "Y", txt)
        txt = re.sub("[ÇĆ]", "C", txt)
        txt = re.sub("Č", "Ch", txt)
        txt = re.sub("Ğ", "G", txt)
        txt = re.sub("[ÑŃ]", "N", txt)
        txt = re.sub("Ś", "S", txt)
        txt = re.sub("[ŞŠ]", "Sh", txt)
        txt = re.sub("Æ", "Ae", txt)
        txt = re.sub("Œ", "Oe", txt)
        txt = re.sub("Ł", "L", txt)
        txt = re.sub("[ŽŹŻ]", "Z", txt)
        txt = re.sub("Ð", "D", txt)

        # Other characters
        txt = re.sub("ß", "ss", txt)
        txt = re.sub("þ", "th", txt)
        txt = re.sub("Þ", "Th", txt)
        txt = re.sub("[¿¡]", "", txt)
        txt = re.sub("đ", "d", txt)

        # Non-printable ascii characters like `\x..`
        # Note: These non-printable ascii characters appear most probably because of
        # copy and paste article's text from some other editor e.g. MS-Word.
        txt = re.sub("\x19", "'", txt)
        txt = re.sub("[\x14|\x1c|\x1d]", "", txt)

        # Japanese style punctuation
        txt = re.sub("[\u301d-\u301e]", '"', txt)
        txt = re.sub("[\u3000-\u303f]", "", txt)
        # Katakana
        txt = re.sub("[\u30a0-\u30ff]", "", txt)
        # Hiragana
        txt = re.sub("[\u3040-\u309f]", "", txt)
        # Kanji (chinese)
        txt = re.sub("[\u4e00-\u9fff]", "", txt)

        return txt

    @staticmethod
    def _count_missing_quotation_marks(txt: str) -> int:
        """Count paragraphs that end before a quote is closed, i.e. multi-paragraph quotes."""
        mq_count = 0
        quote_count = 0
        open_flag = False
        for i, char in enumerate(txt):
            if char == '"' and open_flag is False:
                quote_count += 1
                open_flag = True
            elif char == '"' and open_flag is True:
                quote_count += 1
                open_flag = False
            elif txt[i : i + 4] == "</p>" and open_flag is True:
                open_flag = False
                mq_count += 1

        return mq_count

    def _standardise_quotation_marks(self, txt: str) -> str:
        """Normalise quote mark variants, and re-insert quotes missing from multi-paragraph quotes."""
        # Remove duplicate single quotes
        txt = re.sub(r"([\'`‘´’]){2}", "\\1", txt)
        # Normalise double quotes
        txt = re.sub(r"[\"”“〝〞]", '"', txt)
        # NOTE: keep single quotes for now as they are very common outside quotes

        # Find missing quote marks from multi-paragraph quotes and correct them
        loops_to_run = self._count_missing_quotation_marks(txt)
        quote_count = 0
        for _ in range(loops_to_run):
            open_flag = False
            for i, char in enumerate(txt):
                if char == '"' and open_flag is False:
                    quote_count += 1
                    open_flag = True
                elif char == '"' and open_flag is True:
                    quote_count += 1
                    open_flag = False
                elif txt[i : i + 4] == "</p>" and open_flag is True:
                    txt = txt[:i] + '"' + txt[i:]
                    break
        return txt

    @staticmethod
    def _standardise_hyphens(txt: str) -> str:
        """Normalise hyphen variants to a standard ' - '."""
        return re.sub(r"([‐‑‒–—―])", r" - ", txt)

    @staticmethod
    def _standardise_acronyms(text: str) -> str:
        """
        Remove periods from acronyms, e.g. "N.A.T.O." -> "NATO", and from "a.m."/"p.m.".

        Example:
        >>> SnippetExtractor._standardise_acronyms("N.A.T.O. is an acronym. p.m. should be pm.")
        "NATO is an acronym. pm should be pm."
        """
        pattern = r"(\b(?:[A-Z]\.){2,})|(p.m.)|(a.m.)"

        def replace_function(match):
            return match.group(0).replace(".", "")

        return re.sub(pattern, replace_function, text)

    @staticmethod
    def _fix_punctuation_spacing(text: str) -> str:
        """
        Ensure a space follows punctuation marks (.,!?;:), unless surrounded by digits (e.g. "3.14").

        Example:
        >>> SnippetExtractor._fix_punctuation_spacing("Hello!This is a test.There's a number 3,14,right here.")
        "Hello! This is a test. There's a number 3,14, right here."
        """
        pattern = r"(?!\d)([.,!?;:])(?![\d\s])"
        return re.sub(pattern, r"\1 ", text)

    @staticmethod
    def _remove_breaks_and_spaces(txt: str) -> str:
        """Collapse newlines, repeated whitespace and repeated full stops."""
        txt = re.sub(r"\r+", " ", txt)
        txt = re.sub(r"\n+", " ", txt)
        txt = re.sub(r"\s\s+", " ", txt)
        txt = re.sub(r"\.( *\.+)+", ".", txt)
        # Fix non-breaking space in unicode
        txt = re.sub("\xa0", " ", txt)
        txt = re.sub(r" {2,}", " ", txt)
        return txt

    @staticmethod
    def _format_titles_remove_replies(txt: str) -> str:
        """Expand "hon." to "honourable", drop periods from other titles, and strip bracketed audience reactions."""
        txt = re.sub(r"\bhon\.? ", "honourable ", txt)
        txt = re.sub(r"\bHon\.? ", "Honourable ", txt)
        txt = re.sub(r"Mr\.", "Mr", txt)
        txt = re.sub(r"Mrs\.", "Mrs", txt)
        txt = re.sub(r"Dr\.", "Dr", txt)
        txt = re.sub(r"Prof\.", "Professor", txt)
        txt = re.sub(r"Rev\.", "Reverend", txt)
        # Removes audience reactions like "[hon. members: ...]", which can affect sentiment
        txt = re.sub(r"\s*\[\s*hon\.?.*?\]\s*", " ", txt, flags=re.IGNORECASE)
        return txt

    def clean_text(self, txt: str) -> str:
        """Apply all text-cleaning steps to a speech, in sequence."""
        txt = self._remove_special_characters(txt)
        txt = self._standardise_quotation_marks(txt)
        txt = self._standardise_hyphens(txt)
        txt = self._standardise_acronyms(txt)
        txt = self._fix_punctuation_spacing(txt)
        txt = self._remove_breaks_and_spaces(txt)
        txt = self._format_titles_remove_replies(txt)
        return txt

    def trigger_in_text(self, txt: str) -> bool:
        """Check whether any trigger term is present in the text."""
        for phrase in self.trigger_terms:
            pattern = re.compile(r"\b" + re.escape(phrase) + r"[\W]*\b", re.IGNORECASE)
            if re.search(pattern, txt):
                return True
        return False

    def filter_clean_speech_data(
        self, df: pd.DataFrame, export_filename: str
    ) -> pd.DataFrame:
        """Drop missing/short/non-triggering speeches, clean the remaining text, and sample if requested."""
        df = df.dropna(subset=["text"])

        if self.min_word_count > 0:
            word_count = df["text"].apply(lambda x: len(str(x).split()))
            df = df[word_count >= self.min_word_count]

        if self.speeches_per_year and len(df) > self.speeches_per_year:
            before = len(df)
            df = df.sample(n=self.speeches_per_year, random_state=42).reset_index(
                drop=True
            )
            logging.info(f"{export_filename}: sampled {len(df)}/{before} speeches")

        df["text_clean"] = df["text"].apply(self.clean_text)
        df = df.dropna(subset=["text_clean"]).reset_index(drop=True)

        df["has_trigger"] = df["text"].str.lower().apply(self.trigger_in_text)
        df = df.loc[df["has_trigger"]]

        return df

    @staticmethod
    def _find_matching_words(text: str, word_list: List[str]) -> List[str]:
        """Find every trigger word present in a snippet of text (case-insensitive, returned lowercased)."""
        matching_words = []
        for word in word_list:
            pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
            matching_words.extend(re.findall(pattern, text))
        return [word.lower() for word in matching_words]

    def extract_snippets(
        self, sentences: List[str], sentences_around_trigger: int, allowed_overlap: int
    ) -> List[dict]:
        """
        Extract snippets of sentences centred on each trigger term found.

        Each snippet spans `sentences_around_trigger` sentences before and after the triggering
        sentence. Sentences already used by an earlier snippet are skipped, unless within
        `allowed_overlap` sentences of the end of that snippet.
        """
        snippets = []
        used_indices = set()

        for i, sentence in enumerate(sentences):
            if i in used_indices:
                continue

            for phrase in self.trigger_terms:
                pattern = re.compile(
                    r"\b" + re.escape(phrase) + r"[\W]*\b", re.IGNORECASE
                )
                if not re.search(pattern, sentence):
                    continue

                start = max(0, i - sentences_around_trigger)
                while (start + allowed_overlap) in used_indices:
                    start += 1
                end = min(i + sentences_around_trigger + 1, len(sentences))

                snippet = " ".join(sentences[start:end])
                triggers_found = self._find_matching_words(snippet, self.trigger_terms)

                snippets.append(
                    {
                        "snippet_num": str(len(snippets) + 1),
                        "snippet": snippet,
                        "qualifier": phrase,
                        "trigger_list": triggers_found,
                        "num_triggers": len(triggers_found),
                    }
                )
                used_indices.update(range(start, end))
                break

        return snippets

    @staticmethod
    def _safe_literal_eval(x):
        """Parse a stringified snippet list back into a list of dicts, for exploding into separate rows."""
        if isinstance(x, str):
            try:
                return literal_eval(x.replace("'", '"'))
            except (ValueError, SyntaxError):
                return None
        return x

    def snippet_extract(self, df: pd.DataFrame, export_filename: str, nlp) -> tuple:
        """
        Clean and filter a year's speeches, split them into sentences, and extract trigger-centred snippets to CSV.

        Returns a (speeches_with_trigger, snippets_extracted) tuple, so callers can aggregate totals
        across years.
        """
        before = len(df)
        df = self.filter_clean_speech_data(df, export_filename)
        after = len(df)

        output_path = (
            self.clean_snippets_dir
            / f"snippets_hansard_data_clean_{export_filename}.csv"
        )

        # A year can end up with no speeches at all, e.g. after aggressive filtering/sampling.
        if df.empty:
            logging.info(
                f"{export_filename}: {before} speeches -> {after} with trigger terms -> 0 snippets extracted"
            )
            pd.DataFrame(
                columns=["id", "date", "speaker_id", "snippet_id", "snippet", "text"]
            ).to_csv(output_path, index=False)
            return (after, 0)

        df["sentences"] = [
            [sent.text for sent in doc.sents] for doc in nlp.pipe(df["text_clean"])
        ]
        df["snippets"] = df["sentences"].apply(
            self.extract_snippets, sentences_around_trigger=2, allowed_overlap=0
        )
        df = df.drop(columns=["sentences"])
        df["number_of_snippets"] = df["snippets"].apply(len)

        df["snippets"] = df["snippets"].apply(self._safe_literal_eval)
        df_exploded = df.explode("snippets").reset_index(drop=True)
        snippets_df = pd.json_normalize(df_exploded["snippets"]).reset_index(drop=True)
        result_df = pd.concat(
            [df_exploded.drop(columns=["snippets"]), snippets_df], axis=1
        )
        result_df = result_df.drop(columns=["has_trigger", "text"]).rename(
            columns={"text_clean": "text"}
        )

        if "snippet_num" in result_df.columns:
            result_df["snippet_id"] = (
                result_df[["id", "snippet_num"]].astype(str).agg("_".join, axis=1)
            )
            result_df = result_df.drop(columns=["snippet_num"])
        else:
            result_df["snippet_id"] = (
                result_df["id"].astype(str)
                + "_"
                + [str(i) for i in range(len(result_df))]
            )

        result_df = result_df.dropna(subset=["snippet"])
        result_df.to_csv(output_path, index=False)
        logging.info(
            f"{export_filename}: {before} speeches -> {after} with trigger terms -> {len(result_df)} snippets extracted"
        )
        return (after, len(result_df))
