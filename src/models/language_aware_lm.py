from __future__ import annotations

import gzip
import os
import pickle
from typing import Iterable, Mapping, Sequence

from config.constants import LANGUAGES
from identification.language_identifier import LanguageIdentifier
from models.char_ngram_model import CharNGramModel


class LanguageAwareNGramLM:
    def __init__(self, max_order: int = 8, top_next_chars: int = 16, interp_alpha: float = 1.5) -> None:
        # shared hyperparameters for all language-specific LMs
        self.max_order = max_order
        self.top_next_chars = top_next_chars
        self.interp_alpha = float(interp_alpha)
        # one character LM per target language
        self.lang_models = {
            lang: CharNGramModel(
                max_order=max_order,
                top_next_chars=top_next_chars,
                interp_alpha=self.interp_alpha,
            )
            for lang in LANGUAGES
        }
        # lightweight language identifier to select per-language LM
        self.lang_id = LanguageIdentifier(alpha=0.1)

    def train(self, by_lang: Mapping[str, Iterable[str]]) -> None:
        """Train per-language character LMs and the language identifier."""
        # train each language model on its own shard of labeled data
        for lang in LANGUAGES:
            lines = by_lang.get(lang, [])
            self.lang_models[lang].train(lines)
        # train language identifier on the same supervision
        self.lang_id.train(by_lang)

    def _predict_one(self, context: str, k: int = 3) -> str:
        """Predict top-k next-character guesses for a single context string."""
        # Infer language from a recent suffix to better handle code-switching.
        inferred = self.lang_id.infer(context)
        return "".join(self.lang_models[inferred].predict_ranked_chars(context, k))

    def predict(self, contexts: Sequence[str]) -> list[str]:
        """Predict ranked next-character strings for a batch of contexts."""
        return [self._predict_one(ctx, k=3) for ctx in contexts]

    def save(self, work_dir: str) -> None:
        """Serialize model state to `work_dir/n_gram_lm.checkpoint`."""
        payload = {
            "max_order": self.max_order,
            "top_next_chars": self.top_next_chars,
            "interp_alpha": self.interp_alpha,
            "lang_models": {lang: model.to_state() for lang, model in self.lang_models.items()},
            "lang_id": self.lang_id.to_state(),
        }
        os.makedirs(work_dir, exist_ok=True)
        with gzip.open(os.path.join(work_dir, "n_gram_lm.checkpoint"), "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, work_dir: str) -> LanguageAwareNGramLM:
        """Load a serialized model checkpoint from `work_dir`."""
        with gzip.open(os.path.join(work_dir, "n_gram_lm.checkpoint"), "rb") as f:
            payload = pickle.load(f)

        # recreate LanguageAwareNGramLM with matching hyperparameters
        obj = cls(
            max_order=payload["max_order"],
            top_next_chars=payload["top_next_chars"],
            interp_alpha=payload["interp_alpha"],
        )
        obj.lang_models = {
            lang: CharNGramModel.from_state(payload["lang_models"][lang]) for lang in LANGUAGES
        }
        obj.lang_id = LanguageIdentifier.from_state(payload["lang_id"])
        return obj
