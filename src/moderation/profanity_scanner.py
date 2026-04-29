"""Profanity scanning utilities.

This is intentionally lightweight/offline-friendly:
- No external APIs
- No required downloads

The goal is to flag rows that contain vulgar/profane terms, including some basic
obfuscation patterns (e.g. "$" for "s", "@" for "a", or punctuation inserted).
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class ScanResult:
    has_profanity: bool
    profanity_count: int
    matches: List[str]


class ProfanityScanner:
    """Detects profane/vulgar words using a small lexicon + normalization."""

    # NOTE: Keep this list project-safe and editable. Expand as needed for your domain.
    _DEFAULT_TERMS: Sequence[str] = (
        "asshole",
        "bastard",
        "bitch",
        "bullshit",
        "crap",
        "damn",
        "dick",
        "fuck",
        "motherfucker",
        "piss",
        "shit",
        "slut",
        "whore",
    )

    _LEET_MAP = str.maketrans(
        {
            "@": "a",
            "4": "a",
            "3": "e",
            "1": "i",
            "!": "i",
            "0": "o",
            "$": "s",
            "5": "s",
            "7": "t",
        }
    )

    def __init__(self, terms: Iterable[str] | None = None) -> None:
        base_terms = list(terms) if terms is not None else list(self._DEFAULT_TERMS)
        cleaned = sorted({t.strip().lower() for t in base_terms if t and t.strip()})
        self._terms = cleaned
        # Word boundary helps avoid matching substrings (e.g., "ass" in "class").
        escaped = [re.escape(t) for t in cleaned]
        self._pattern = re.compile(r"\b(" + "|".join(escaped) + r")\b", flags=re.IGNORECASE)

    @property
    def terms(self) -> List[str]:
        """The normalized profanity lexicon used for scanning."""
        return list(self._terms)

    def scan_text(self, text: str) -> ScanResult:
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        if not text.strip():
            return ScanResult(has_profanity=False, profanity_count=0, matches=[])

        matches: list[str] = []

        # Pass 1: direct word-boundary matching.
        direct = [m.group(1).lower() for m in self._pattern.finditer(text)]
        matches.extend(direct)

        # Pass 2: normalize common obfuscations (leet + remove punctuation between letters).
        normalized = self._normalize_for_scan(text)
        if normalized != text:
            norm_direct = [m.group(1).lower() for m in self._pattern.finditer(normalized)]
            matches.extend(norm_direct)

        # De-duplicate but keep stable order.
        seen: set[str] = set()
        uniq: list[str] = []
        for term in matches:
            if term not in seen:
                seen.add(term)
                uniq.append(term)

        return ScanResult(has_profanity=bool(uniq), profanity_count=len(uniq), matches=uniq)

    def scan_texts(self, texts: Sequence[str]) -> List[ScanResult]:
        return [self.scan_text(t) for t in texts]

    @classmethod
    def _normalize_for_scan(cls, text: str) -> str:
        t = text.lower().translate(cls._LEET_MAP)
        # Replace non-letters/numbers with spaces then collapse.
        t = re.sub(r"[^a-z0-9]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

