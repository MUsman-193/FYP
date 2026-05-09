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


# Whole-token replacements (lookup after lowercasing). Keeps toxic/social text readable for ML.
_SLANG_LEXICON: Dict[str, str] = {
    "u": "you",
    "ur": "your",
    "r": "are",
    "y": "why",
    "w": "with",
    "bc": "because",
    "b/c": "because",
    "cuz": "because",
    "cos": "because",
    "bcuz": "because",
    "pls": "please",
    "plz": "please",
    "thx": "thanks",
    "ty": "thank you",
    "np": "no problem",
    "idk": "I do not know",
    "idc": "I do not care",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "tbh": "to be honest",
    "ngl": "not going to lie",
    "btw": "by the way",
    "fwiw": "for what it is worth",
    "irl": "in real life",
    "af": "as hell",
    "rn": "right now",
    "atm": "at the moment",
    "wbu": "what about you",
    "hbu": "how about you",
    "omg": "oh my god",
    "omfg": "oh my god",
    "lol": "laughing out loud",
    "lmao": "laughing hard",
    "lmfao": "laughing hard",
    "rofl": "laughing hard",
    "smh": "shaking my head",
    "ikr": "I know right",
    "ik": "I know",
    "fr": "for real",
    "sus": "suspicious",
    "lowkey": "somewhat",
    "highkey": "very",
    "ppl": "people",
    "abt": "about",
    "msg": "message",
    "dm": "direct message",
    "gf": "girlfriend",
    "bf": "boyfriend",
    "ne1": "anyone",
    "sum1": "someone",
    "nvm": "never mind",
    "jk": "just kidding",
    "jfc": "for crying out loud",
    "stfu": "shut up",
    "gtfo": "get out",
    "ffs": "for goodness sake",
    "wtf": "what the hell",
    "wth": "what the hell",
    "wya": "where you at",
    "wyd": "what you doing",
    "ttyl": "talk to you later",
    "omw": "on my way",
    "icymi": "in case you missed it",
    "ftw": "for the win",
    "fml": "I am upset",
    "tl": "too long",
    "dr": "did not read",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "kinda": "kind of",
    "sorta": "sort of",
    "prolly": "probably",
    "dunno": "do not know",
    "lemme": "let me",
    "gimme": "give me",
    "imma": "I am going to",
    "tryna": "trying to",
    "coulda": "could have",
    "woulda": "would have",
    "shoulda": "should have",
    "mighta": "might have",
    "musta": "must have",
    "doin": "doing",
    "goin": "going",
    "nothin": "nothing",
    "somethin": "something",
    "anythin": "anything",
    "everythin": "everything",
    "fav": "favorite",
    "pic": "picture",
    "obvi": "obviously",
    "def": "definitely",
    "abs": "absolutely",
    "govt": "government",
    "info": "information",
    "ima": "I am going to",
}

# Contractions and informal verb forms (token -> normalized phrase)
_CONTRACTION_LEXICON: Dict[str, str] = {
    "can't": "can not",
    "cannot": "can not",
    "won't": "will not",
    "i'm": "I am",
    "i've": "I have",
    "i'll": "I will",
    "i'd": "I would",
    "you're": "you are",
    "you've": "you have",
    "you'll": "you will",
    "you'd": "you would",
    "we're": "we are",
    "we've": "we have",
    "we'll": "we will",
    "we'd": "we would",
    "they're": "they are",
    "they've": "they have",
    "they'll": "they will",
    "they'd": "they would",
    "he's": "he is",
    "he'll": "he will",
    "he'd": "he would",
    "she's": "she is",
    "she'll": "she will",
    "she'd": "she would",
    "it's": "it is",
    "it'll": "it will",
    "it'd": "it would",
    "that's": "that is",
    "that'll": "that will",
    "there's": "there is",
    "here's": "here is",
    "what's": "what is",
    "who's": "who is",
    "where's": "where is",
    "how's": "how is",
    "when's": "when is",
    "why's": "why is",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "doesn't": "does not",
    "didn't": "did not",
    "don't": "do not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "mightn't": "might not",
    "mustn't": "must not",
    "needn't": "need not",
    "ain't": "is not",
    "y'all": "you all",
    "ya": "you",
    "ye": "you",
}

# Tokens that suggest slang-heavy comments (for auto-suggest)
_SLANG_HINT_TOKENS = frozenset(
    {
        "u",
        "ur",
        "r",
        "y",
        "pls",
        "plz",
        "thx",
        "idk",
        "tbh",
        "ngl",
        "lol",
        "lmao",
        "smh",
        "imo",
        "btw",
        "omg",
        "wtf",
        "gonna",
        "wanna",
        "kinda",
        "ima",
        "imma",
        "aint",
        "dunno",
    }
)

_TOKEN_EDGE = re.compile(r"^(\W*)([\w']+)(\W*)$", re.UNICODE)

# Two+ consecutive double-quote-like glyphs (ASCII + common Unicode / fullwidth / ditto / guillemets).
_QUOTE_FENCE_RE = re.compile(
    "(?:["
    + '"'
    + "\u201c\u201d\u201e\u201f\u2033\u2036\uff02\u301d\u301e\u275d\u275e\u3003\u00ab\u00bb"
    + "]){2,}"
)


@dataclass
class PreprocessConfig:
    """Configuration for preprocessing steps."""

    lowercase: bool = False
    remove_punctuation: bool = False
    remove_stopwords: bool = False
    remove_numbers: bool = False
    normalize_whitespace: bool = False
    normalize_unicode: bool = False
    remove_noise: bool = False
    reduce_elongations: bool = False
    map_slang: bool = False
    expand_contractions: bool = False


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

    @staticmethod
    def _normalize_unicode_nfkc(text: str) -> str:
        return unicodedata.normalize("NFKC", text)

    @staticmethod
    def _strip_quote_emphasis_noise(text: str) -> str:
        """Remove copy-paste / social emphasis fences like \"\"\"\"word\"\"\"\".

        Strips invisible format chars, then collapses runs of two or more
        double-quote-like characters (ASCII or common Unicode lookalikes) to a
        single space. Runs whenever ``apply`` is called (not only when
        ``remove_noise`` is enabled).
        """
        text = re.sub(
            r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff\u00ad\u034f\u061c]",
            "",
            text,
        )
        text = _QUOTE_FENCE_RE.sub(" ", text)
        return text

    @staticmethod
    def _remove_noise(text: str) -> str:
        """Strip URLs, emails, HTML-like tags, handles, hashtags-as-markup, leet noise."""
        # HTML / angle-bracket markup (bounded length to avoid pathological input)
        text = re.sub(r"<[^>]{1,2000}>", " ", text)
        # URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
        # Email addresses
        text = re.sub(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,63}\b", " ", text)
        # @mentions (social noise)
        text = re.sub(r"@\w{1,80}\b", " ", text)
        # Hashtag: keep word, drop hash
        text = re.sub(r"#(\w{1,80})\b", r"\1", text)
        # Standalone hash tokens
        text = re.sub(r"#+\b", " ", text)
        # Leading retweet boilerplate
        text = re.sub(r"(?i)^\s*rt\s+[:@]?\s*", " ", text)
        # Repeated punctuation clusters (e.g. "???!!!") -> single space boundary
        text = re.sub(r"[!?]{3,}", " ", text)
        return text

    @staticmethod
    def _reduce_elongations(text: str) -> str:
        """Collapse exaggerated character repeats (e.g. fuuuuck -> fuuck)."""
        return re.sub(r"([A-Za-z])\1{2,}", r"\1\1", text)

    def _map_tokens(self, text: str, *, slang: bool, contractions: bool) -> str:
        """Whole-token slang / contraction expansion (preserves outer punctuation per token)."""
        if not slang and not contractions:
            return text

        def replace_one(raw: str) -> str:
            m = _TOKEN_EDGE.match(raw)
            if not m:
                return raw
            lead, core, trail = m.group(1), m.group(2), m.group(3)
            key = core.lower()
            if contractions and key in _CONTRACTION_LEXICON:
                return f"{lead}{_CONTRACTION_LEXICON[key]}{trail}"
            if slang and key in _SLANG_LEXICON:
                return f"{lead}{_SLANG_LEXICON[key]}{trail}"
            return raw

        parts = re.split(r"(\s+)", text)
        out: List[str] = []
        for p in parts:
            if not p or p.isspace():
                out.append(p)
            else:
                out.append(replace_one(p))
        return "".join(out)

    def apply(self, text: str, config: PreprocessConfig) -> str:
        if not isinstance(text, str):
            text = "" if text is None else str(text)

        if config.normalize_unicode:
            text = self._normalize_unicode_nfkc(text)
        text = self._strip_quote_emphasis_noise(text)
        if config.remove_noise:
            text = self._remove_noise(text)
        if config.reduce_elongations:
            text = self._reduce_elongations(text)
        if config.lowercase:
            text = text.lower()
        text = self._map_tokens(
            text,
            slang=config.map_slang,
            contractions=config.expand_contractions,
        )
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
        url_rows = 0
        mention_rows = 0
        tag_rows = 0
        elong_rows = 0
        non_ascii_chars = 0
        slang_hint_rows = 0
        quote_fence_rows = 0

        url_re = re.compile(r"https?://|www\.", re.I)
        mention_re = re.compile(r"@\w")
        tag_re = re.compile(r"<[^>]{1,2000}>")
        elong_re = re.compile(r"(.)\1{2,}")

        for raw in texts:
            text = "" if raw is None else str(raw)
            total_chars += len(text)
            uppercase_chars += sum(1 for ch in text if ch.isupper())
            punctuation_chars += sum(1 for ch in text if self._is_punctuation_char(ch))
            digit_chars += sum(1 for ch in text if ch.isdigit())
            non_ascii_chars += sum(1 for ch in text if ord(ch) > 127)
            if re.search(r"\s{2,}|\n|\t", text):
                messy_ws_count += 1
            if url_re.search(text):
                url_rows += 1
            if mention_re.search(text):
                mention_rows += 1
            if tag_re.search(text):
                tag_rows += 1
            if elong_re.search(text):
                elong_rows += 1
            if _QUOTE_FENCE_RE.search(text):
                quote_fence_rows += 1

            tokens = text.split()
            if tokens:
                stop_cnt = sum(1 for tok in tokens if tok.lower() in self._stop_words)
                stopword_ratio_sum += stop_cnt / len(tokens)
                tokenized_count += 1
                lowered = {t.lower().strip(string.punctuation + "'.,") for t in tokens}
                if lowered & _SLANG_HINT_TOKENS:
                    slang_hint_rows += 1

        safe_total_chars = max(total_chars, 1)
        uppercase_ratio = uppercase_chars / safe_total_chars
        punctuation_ratio = punctuation_chars / safe_total_chars
        digit_ratio = digit_chars / safe_total_chars
        messy_ws_ratio = messy_ws_count / len(texts)
        stopword_ratio = (
            stopword_ratio_sum / tokenized_count if tokenized_count > 0 else 0.0
        )
        non_ascii_ratio = non_ascii_chars / safe_total_chars
        n = len(texts)
        url_ratio = url_rows / n
        mention_ratio = mention_rows / n
        tag_ratio = tag_rows / n
        elong_ratio = elong_rows / n
        slang_hint_ratio = slang_hint_rows / n
        quote_fence_ratio = quote_fence_rows / n

        noisy = (
            url_ratio > 0.02
            or mention_ratio > 0.03
            or tag_ratio > 0.02
            or quote_fence_ratio > 0.015
        )

        config = PreprocessConfig(
            lowercase=uppercase_ratio > 0.08,
            remove_punctuation=punctuation_ratio > 0.05,
            remove_stopwords=stopword_ratio > 0.35,
            remove_numbers=digit_ratio > 0.03,
            normalize_whitespace=messy_ws_ratio > 0.20,
            normalize_unicode=non_ascii_ratio > 0.015,
            remove_noise=noisy,
            reduce_elongations=elong_ratio > 0.06,
            map_slang=slang_hint_ratio > 0.12,
            expand_contractions=slang_hint_ratio > 0.08,
        )

        stats = {
            "uppercase_ratio": round(uppercase_ratio, 4),
            "punctuation_ratio": round(punctuation_ratio, 4),
            "digit_ratio": round(digit_ratio, 4),
            "messy_whitespace_ratio": round(messy_ws_ratio, 4),
            "avg_stopword_ratio": round(stopword_ratio, 4),
            "non_ascii_ratio": round(non_ascii_ratio, 4),
            "rows_with_url_like": round(url_ratio, 4),
            "rows_with_mentions": round(mention_ratio, 4),
            "rows_with_html_angle_tags": round(tag_ratio, 4),
            "rows_with_char_elongations": round(elong_ratio, 4),
            "rows_with_slang_hint_tokens": round(slang_hint_ratio, 4),
            "rows_with_multi_quote_fences": round(quote_fence_ratio, 4),
        }
        return config, stats
