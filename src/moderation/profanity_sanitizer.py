"""Profanity sanitization/replacement utilities."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class SanitizeResult:
    clean_text: str
    replacements: List[str]


class ProfanitySanitizer:
    """Replaces profane terms either by masking or substitution.

    This sanitizer targets:
    - direct word matches (word boundaries)
    - simple obfuscation where punctuation/spaces are inserted between letters
      (e.g. "f*u*c*k", "f u c k", "f-uck")
    """

    def __init__(self, terms: Iterable[str]) -> None:
        cleaned = sorted({t.strip().lower() for t in terms if t and t.strip()})
        self._terms = cleaned
        # Match whole words, plus common inflections so "fuck" catches "fucking", etc.
        suffix = r"(?:'s|s|es|ed|ing|er|ers|y|ies)?"
        self._direct = re.compile(
            r"\b(" + "|".join(re.escape(t) for t in cleaned) + r")" + suffix + r"\b",
            flags=re.IGNORECASE,
        )
        # Obfuscation: letters separated by non-word chars/underscores/spaces.
        self._obfuscated: list[tuple[str, re.Pattern[str]]] = [
            (t, re.compile(self._obfuscated_pattern_for_term(t), flags=re.IGNORECASE))
            for t in cleaned
        ]

    @staticmethod
    def _obfuscated_pattern_for_term(term: str) -> str:
        # Example: "fuck" -> r"\b(f[\W_]*u[\W_]*c[\W_]*k)\b"
        parts = [re.escape(ch) + r"[\W_]*" for ch in term]
        core = "".join(parts).rstrip(r"[\W_]*")
        return r"\b(" + core + r")\b"

    def sanitize(
        self,
        text: str,
        *,
        mode: str = "mask",
        mask_token: str = "[censored]",
        replacement_map: Dict[str, str] | None = None,
    ) -> SanitizeResult:
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        if not text.strip():
            return SanitizeResult(clean_text=text, replacements=[])

        mode = (mode or "mask").strip().lower()
        if mode not in {"mask", "replace"}:
            raise ValueError("mode must be 'mask' or 'replace'")

        rep_map = {k.lower(): v for k, v in (replacement_map or {}).items()}
        replacements: list[str] = []
        out = text

        def repl_for(matched_term: str) -> str:
            t = matched_term.lower()
            if mode == "replace" and t in rep_map:
                return rep_map[t]
            return mask_token

        # Pass 1: direct.
        def _direct_sub(m: re.Match[str]) -> str:
            base = m.group(1)
            replacements.append(base.lower())
            return repl_for(base)

        out = self._direct.sub(_direct_sub, out)

        # Pass 2: obfuscated (only for terms not already replaced in pass 1).
        for canonical, pattern in self._obfuscated:
            if canonical in replacements:
                continue
            if not pattern.search(out):
                continue

            def _obf_sub(m: re.Match[str], canon: str = canonical) -> str:
                replacements.append(canon)
                return repl_for(canon)

            out = pattern.sub(_obf_sub, out)

        # De-dupe (stable).
        seen: set[str] = set()
        uniq: list[str] = []
        for t in replacements:
            if t not in seen:
                seen.add(t)
                uniq.append(t)

        return SanitizeResult(clean_text=out, replacements=uniq)

    def sanitize_texts(
        self,
        texts: List[str],
        *,
        mode: str = "mask",
        mask_token: str = "[censored]",
        replacement_map: Dict[str, str] | None = None,
    ) -> List[SanitizeResult]:
        return [
            self.sanitize(t, mode=mode, mask_token=mask_token, replacement_map=replacement_map)
            for t in texts
        ]

