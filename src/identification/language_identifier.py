from __future__ import annotations

import math
import unicodedata
from collections import Counter
from typing import Iterable, Mapping

from config.constants import LANGUAGES, LATIN_LANGS, SPACE_RE

ModelState = dict[str, object]


class LanguageIdentifier:
    def __init__(self, alpha: float = 0.1) -> None:
        # additive smoothing parameter for all Naive Bayes scores
        self.alpha = alpha
        # latin trigram features (used to split en/fr/de/it)
        self.counts = {lang: Counter() for lang in LATIN_LANGS}
        self.total = {lang: 0 for lang in LATIN_LANGS}
        self.vocab = set()
        # script-agnostic character features (used for broader fallback)
        self.char_counts = {lang: Counter() for lang in LANGUAGES}
        self.char_total = {lang: 0 for lang in LANGUAGES}
        self.char_vocab = set()
        # document priors per language
        self.lang_docs = {lang: 0 for lang in LANGUAGES}

    @staticmethod
    def _in_range(ch: str, start: int, end: int) -> bool:
        """Return whether a codepoint is within an inclusive Unicode range."""
        return start <= ord(ch) <= end

    def _script_hint(self, text: str) -> str | None:
        """Return a high-confidence script hint or None."""
        has_han = False
        for ch in text:
            # kana often co-occurs with han in Japanese
            if self._in_range(ch, 0x3040, 0x30FF) or self._in_range(ch, 0x31F0, 0x31FF):
                return "ja"
            if self._in_range(ch, 0xAC00, 0xD7AF) or self._in_range(ch, 0x1100, 0x11FF):
                return "ko"
            if self._in_range(ch, 0x0600, 0x06FF) or self._in_range(ch, 0x0750, 0x077F):
                return "ar"
            if self._in_range(ch, 0x0900, 0x097F):
                return "hi"
            if self._in_range(ch, 0x0400, 0x052F):
                return "ru"
            # han alone is ambiguous between Chinese and Japanese
            if self._in_range(ch, 0x4E00, 0x9FFF):
                has_han = True
        if has_han:
            return "han"
        return None

    @staticmethod
    def _is_latin_char(ch: str) -> bool:
        """Return True for alphabetic Latin characters."""
        return ch.isalpha() and "LATIN" in unicodedata.name(ch, "")

    def _latin_trigrams(self, text: str) -> list[str]:
        """Extract normalized Latin character trigrams from `text`."""
        chars = []
        for ch in text.lower():
            chars.append(ch if self._is_latin_char(ch) else " ")

        cleaned = SPACE_RE.sub(" ", "".join(chars)).strip()
        padded = f"  {cleaned} "
        return [padded[i : i + 3] for i in range(len(padded) - 2)]

    def train(self, by_lang: Mapping[str, Iterable[str]]) -> None:
        """Learns per-language feature counts and priors from labeled data."""
        for lang in LANGUAGES:
            for line in by_lang.get(lang, []):
                self.lang_docs[lang] += 1
                chars = [ch for ch in line if not ch.isspace()]
                if chars:
                    self.char_counts[lang].update(chars)
                if lang not in LATIN_LANGS:
                    continue
                grams = self._latin_trigrams(line)
                if not grams:
                    continue
                self.counts[lang].update(grams)

        # finalize denominators and vocabularies for smoothed scoring
        for lang in LATIN_LANGS:
            self.total[lang] = sum(self.counts[lang].values())
            self.vocab.update(self.counts[lang].keys())
        for lang in LANGUAGES:
            self.char_total[lang] = sum(self.char_counts[lang].values())
            self.char_vocab.update(self.char_counts[lang].keys())

    def _score_latin(self, text: str) -> dict[str, float]:
        """Score Latin languages with smoothed trigram Naive Bayes."""
        grams = self._latin_trigrams(text)
        if not grams:
            return {lang: float("-inf") for lang in LATIN_LANGS}

        scores = {}
        total_docs = sum(self.lang_docs[lang] for lang in LATIN_LANGS) + len(LATIN_LANGS)
        vocab_size = max(1, len(self.vocab))

        # score formula per language: log P(lang) + sum log P(gram | lang)
        # with add-alpha smoothing so unseen grams remain usable.
        for lang in LATIN_LANGS:
            logp = math.log((self.lang_docs[lang] + 1) / total_docs)
            denom = self.total[lang] + self.alpha * vocab_size
            if denom <= 0:
                scores[lang] = float("-inf")
                continue

            counter = self.counts[lang]
            for gram in grams:
                logp += math.log((counter.get(gram, 0) + self.alpha) / denom)
            scores[lang] = logp

        return scores

    def _score_chars(self, text: str, candidates: Iterable[str]) -> dict[str, float]:
        """Score candidate languages with smoothed character Naive Bayes."""
        chars = [ch for ch in text if not ch.isspace()]
        if not chars:
            return {lang: float("-inf") for lang in candidates}

        vocab_size = max(1, len(self.char_vocab))
        total_docs = sum(self.lang_docs.values()) + len(LANGUAGES)
        scores = {}

        for lang in candidates:
            denom = self.char_total[lang] + self.alpha * vocab_size
            if denom <= 0:
                scores[lang] = float("-inf")
                continue

            logp = math.log((self.lang_docs[lang] + 1) / total_docs)
            counter = self.char_counts[lang]
            for ch in chars:
                logp += math.log((counter.get(ch, 0) + self.alpha) / denom)
            scores[lang] = logp

        return scores

    def infer(self, text: str) -> str:
        """Predict the most likely language code for `text`."""
        # early return when strongly identify a language
        hint = self._script_hint(text)
        if hint in ("ru", "ar", "hi", "ko", "ja"):
            return hint

        # resolve ambiguity between zh/ja
        if hint == "han":
            han_scores = self._score_chars(text, ("zh", "ja"))
            return max(han_scores.items(), key=lambda kv: kv[1])[0]

        # Latin-language classifier with trigram features
        latin_scores = self._score_latin(text)
        best_latin_lang, best_latin_score = max(latin_scores.items(), key=lambda kv: kv[1])
        if best_latin_score != float("-inf"):
            return best_latin_lang

        # global fallback over all supported languages
        all_scores = self._score_chars(text, LANGUAGES)
        best_lang, best_score = max(all_scores.items(), key=lambda kv: kv[1])
        if best_score != float("-inf"):
            return best_lang

        # fallback to English if all else fails
        return "en"

    def ranked_latin(self, text: str) -> list[str]:
        """Return Latin language candidates sorted by descending score."""
        scores = self._score_latin(text)
        return [lang for lang, _ in sorted(scores.items(), key=lambda kv: -kv[1])]

    def to_state(self) -> ModelState:
        """Serialize learned statistics into a checkpoint payload."""
        return {
            "alpha": self.alpha,
            "counts": self.counts,
            "total": self.total,
            "vocab": list(self.vocab),
            "char_counts": self.char_counts,
            "char_total": self.char_total,
            "char_vocab": list(self.char_vocab),
            "lang_docs": self.lang_docs,
        }

    @classmethod
    def from_state(cls, state: ModelState) -> LanguageIdentifier:
        """Restore a LanguageIdentifier from a serialized payload."""
        obj = cls(alpha=state["alpha"])
        obj.counts = {lang: Counter(state["counts"][lang]) for lang in LATIN_LANGS}
        obj.total = state["total"]
        obj.vocab = set(state["vocab"])
        obj.char_counts = {lang: Counter(state["char_counts"][lang]) for lang in LANGUAGES}
        obj.char_total = state["char_total"]
        obj.char_vocab = set(state["char_vocab"])
        obj.lang_docs = state["lang_docs"]
        return obj
