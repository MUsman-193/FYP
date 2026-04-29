"""Text preprocessing utilities for toxic comment datasets."""

from __future__ import annotations

import re
import string
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Tuple

try:
    # Preferred: sklearn's built-in list (fast, no extra downloads).
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as _ENGLISH_STOP_WORDS
except Exception:
    _ENGLISH_STOP_WORDS = None


def _default_stop_words() -> set[str]:
    if _ENGLISH_STOP_WORDS is not None:
        return set(_ENGLISH_STOP_WORDS)
    # Fallback: lightweight built-in list (keeps UI usable without sklearn).
    return {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
        "you",
        "your",
    }


@dataclass
class PreprocessConfig:
    """Configuration for preprocessing steps."""

    lowercase: bool = False
    remove_punctuation: bool = False
    remove_stopwords: bool = False
    remove_numbers: bool = False
    normalize_whitespace: bool = False


class TextPreprocessor:
    """Applies configurable preprocessing on text."""

    def __init__(self) -> None:
        self._stop_words = _default_stop_words()

    @staticmethod
    def _is_punctuation_char(ch: str) -> bool:
        """True for ASCII punctuation and Unicode punctuation marks (Pi, Pf, Po, etc.)."""
        if len(ch) != 1:
            return False
        if ch in string.punctuation:
            return True
        return unicodedata.category(ch).startswith("P")

    def _remove_punctuation(self, text: str) -> str:
        return "".join(ch for ch in text if not self._is_punctuation_char(ch))

    def apply(self, text: str, config: PreprocessConfig) -> str:
        if not isinstance(text, str):
            text = "" if text is None else str(text)

        if config.lowercase:
            text = text.lower()
        if config.remove_punctuation:
            text = self._remove_punctuation(text)
        if config.remove_numbers:
            text = re.sub(r"\d+", "", text)
        if config.remove_stopwords:
            text = " ".join(
                token for token in text.split() if token.lower() not in self._stop_words
            )
        if config.normalize_whitespace:
            text = re.sub(r"\s+", " ", text).strip()

        return text

    def suggest(self, texts: List[str]) -> Tuple[PreprocessConfig, Dict[str, float]]:
        """Suggest preprocessing based on dataset-level text statistics."""
        if not texts:
            return PreprocessConfig(), {}

        total_chars = 0
        uppercase_chars = 0
        punctuation_chars = 0
        digit_chars = 0
        messy_ws_count = 0
        stopword_ratio_sum = 0.0
        tokenized_count = 0

        for raw in texts:
            text = "" if raw is None else str(raw)
            total_chars += len(text)
            uppercase_chars += sum(1 for ch in text if ch.isupper())
            punctuation_chars += sum(1 for ch in text if self._is_punctuation_char(ch))
            digit_chars += sum(1 for ch in text if ch.isdigit())
            if re.search(r"\s{2,}|\n|\t", text):
                messy_ws_count += 1

            tokens = text.split()
            if tokens:
                stop_cnt = sum(1 for tok in tokens if tok.lower() in self._stop_words)
                stopword_ratio_sum += stop_cnt / len(tokens)
                tokenized_count += 1

        safe_total_chars = max(total_chars, 1)
        uppercase_ratio = uppercase_chars / safe_total_chars
        punctuation_ratio = punctuation_chars / safe_total_chars
        digit_ratio = digit_chars / safe_total_chars
        messy_ws_ratio = messy_ws_count / len(texts)
        stopword_ratio = (
            stopword_ratio_sum / tokenized_count if tokenized_count > 0 else 0.0
        )

        config = PreprocessConfig(
            lowercase=uppercase_ratio > 0.08,
            remove_punctuation=punctuation_ratio > 0.05,
            remove_stopwords=stopword_ratio > 0.35,
            remove_numbers=digit_ratio > 0.03,
            normalize_whitespace=messy_ws_ratio > 0.20,
        )

        stats = {
            "uppercase_ratio": round(uppercase_ratio, 4),
            "punctuation_ratio": round(punctuation_ratio, 4),
            "digit_ratio": round(digit_ratio, 4),
            "messy_whitespace_ratio": round(messy_ws_ratio, 4),
            "avg_stopword_ratio": round(stopword_ratio, 4),
        }
        return config, stats
