"""Text augmentation utilities."""

from __future__ import annotations

import random
import re
import string
from dataclasses import dataclass
from typing import List, Sequence

try:
    from nltk.corpus import wordnet as wn
except Exception:  # pragma: no cover
    wn = None


@dataclass
class AugmentConfig:
    synonym_replacement: bool = False
    random_insertion: bool = False
    random_swap: bool = False
    random_deletion: bool = False
    back_translation: bool = False
    paraphrasing: bool = False
    sentence_shuffling: bool = False
    sentence_cropping: bool = False
    noise_injection: bool = False
    strength: float = 0.1


class TextAugmenter:
    """Applies a chain of augmentation methods."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    def apply(self, text: str, config: AugmentConfig) -> str:
        if not isinstance(text, str):
            text = "" if text is None else str(text)

        if not text.strip():
            return text

        out = text
        if config.synonym_replacement:
            out = self._synonym_replacement(out, config.strength)
        if config.random_insertion:
            out = self._random_insertion(out, config.strength)
        if config.random_swap:
            out = self._random_swap(out, config.strength)
        if config.random_deletion:
            out = self._random_deletion(out, config.strength)
        if config.back_translation:
            out = self._pseudo_back_translation(out)
        if config.paraphrasing:
            out = self._paraphrase(out)
        if config.sentence_shuffling:
            out = self._sentence_shuffle(out)
        if config.sentence_cropping:
            out = self._sentence_crop(out, config.strength)
        if config.noise_injection:
            out = self._noise_injection(out, config.strength)
        return out

    def _tokenize(self, text: str) -> List[str]:
        return text.split()

    def _detokenize(self, tokens: Sequence[str]) -> str:
        return " ".join(tokens).strip()

    def _synonyms(self, word: str) -> List[str]:
        if wn is None:
            return []
        syns = set()
        try:
            synsets = wn.synsets(word)
        except LookupError:
            return []
        for synset in synsets:
            for lemma in synset.lemmas():
                name = lemma.name().replace("_", " ")
                if name.lower() != word.lower() and name.isascii():
                    syns.add(name)
        return list(syns)

    def _synonym_replacement(self, text: str, strength: float) -> str:
        tokens = self._tokenize(text)
        if len(tokens) < 2:
            return text
        replace_count = max(1, int(len(tokens) * strength))
        indices = list(range(len(tokens)))
        self._rng.shuffle(indices)
        for idx in indices[:replace_count]:
            cands = self._synonyms(tokens[idx].strip(string.punctuation))
            if cands:
                tokens[idx] = self._rng.choice(cands)
        return self._detokenize(tokens)

    def _random_insertion(self, text: str, strength: float) -> str:
        tokens = self._tokenize(text)
        if not tokens:
            return text
        insert_count = max(1, int(len(tokens) * strength))
        for _ in range(insert_count):
            base = self._rng.choice(tokens)
            cands = self._synonyms(base.strip(string.punctuation))
            insert_token = self._rng.choice(cands) if cands else base
            pos = self._rng.randint(0, len(tokens))
            tokens.insert(pos, insert_token)
        return self._detokenize(tokens)

    def _random_swap(self, text: str, strength: float) -> str:
        tokens = self._tokenize(text)
        if len(tokens) < 2:
            return text
        swap_count = max(1, int(len(tokens) * strength))
        for _ in range(swap_count):
            i, j = self._rng.sample(range(len(tokens)), 2)
            tokens[i], tokens[j] = tokens[j], tokens[i]
        return self._detokenize(tokens)

    def _random_deletion(self, text: str, strength: float) -> str:
        tokens = self._tokenize(text)
        if len(tokens) <= 2:
            return text
        keep = [tok for tok in tokens if self._rng.random() > strength]
        if not keep:
            keep = [self._rng.choice(tokens)]
        return self._detokenize(keep)

    def _pseudo_back_translation(self, text: str) -> str:
        # Offline-safe approximation when external translation APIs are unavailable.
        tokens = self._tokenize(text)
        if len(tokens) <= 3:
            return text
        mid = tokens[1:-1]
        mid.reverse()
        return self._detokenize([tokens[0], *mid, tokens[-1]])

    def _paraphrase(self, text: str) -> str:
        # Lightweight paraphrasing using synonym replacement + a small shuffle.
        replaced = self._synonym_replacement(text, 0.15)
        return self._random_swap(replaced, 0.05)

    def _sentence_split(self, text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p for p in parts if p]

    def _sentence_shuffle(self, text: str) -> str:
        sents = self._sentence_split(text)
        if len(sents) < 2:
            return text
        self._rng.shuffle(sents)
        return " ".join(sents)

    def _sentence_crop(self, text: str, strength: float) -> str:
        tokens = self._tokenize(text)
        if len(tokens) <= 4:
            return text
        remove_count = max(1, int(len(tokens) * min(strength, 0.4)))
        return self._detokenize(tokens[:-remove_count])

    def _noise_injection(self, text: str, strength: float) -> str:
        chars = list(text)
        if not chars:
            return text
        noise_count = max(1, int(len(chars) * min(strength, 0.08)))
        letters = "abcdefghijklmnopqrstuvwxyz"
        for _ in range(noise_count):
            pos = self._rng.randint(0, len(chars) - 1)
            if chars[pos].isspace():
                continue
            chars[pos] = self._rng.choice(letters)
        return "".join(chars)
