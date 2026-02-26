from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Mapping

ModelState = dict[str, object]


class CharNGramModel:
    # used to left-pad contexts so high-order n-grams exist at line start
    START_TOKEN = "\u0002"
    # hard fallback when no model evidence is available
    DEFAULT_FALLBACK = [" ", "e", "a", "i", "o", "t", "n", "r", ".", ","]

    def __init__(
        self,
        max_order: int = 8,
        top_next_chars: int = 16,
        interp_alpha: float = 1.5,
        fallback_limit: int = 256,
    ):
        # core decoding/training hyperparameters
        self.max_order = max_order
        self.top_next_chars = top_next_chars
        self.interp_alpha = float(interp_alpha)
        self.fallback_limit = int(fallback_limit)
        # sparse table mapping order -> context -> ranked next chars
        self.counts = {order: {} for order in range(1, max_order + 1)}
        # cached raw totals per context for interpolation weights
        self.context_totals = {order: {} for order in range(1, max_order + 1)}
        # global unigram fallback distribution
        self.fallback_chars = []
        self.fallback_probs = {}

    @staticmethod
    def _rank(counter: Mapping[str, int], limit: int | None = None) -> list[tuple[str, int]]:
        # Rank by descending count, then char for deterministic ties.
        ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked if limit is None else ranked[:limit]

    def train(self, lines: Iterable[str]) -> None:
        # temporary raw counts before pruning/truncation.
        by_order = {order: defaultdict(Counter) for order in range(1, self.max_order + 1)}
        unigram = Counter()
        pad = self.START_TOKEN * (self.max_order - 1)

        for line in lines:
            # prefix start tokens so early positions still have full context
            text = pad + line
            for i in range(self.max_order - 1, len(text)):
                next_char = text[i]
                # global unigram count for fallback
                unigram[next_char] += 1
                # update every n-gram order at this character position
                for order in range(1, self.max_order + 1):
                    ctx_len = order - 1
                    context = text[i - ctx_len : i]
                    by_order[order][context][next_char] += 1

        # convert to compact ranked tables without context pruning
        counts = {order: {} for order in range(1, self.max_order + 1)}
        context_totals = {order: {} for order in range(1, self.max_order + 1)}
        for order, contexts in by_order.items():
            for context, counter in contexts.items():
                total = sum(counter.values())
                counts[order][context] = self._rank(counter, self.top_next_chars)
                context_totals[order][context] = total

        # commit trained tables
        self.counts = counts
        self.context_totals = context_totals

        # build normalized unigram fallback
        ranked_unigram = self._rank(unigram)
        total_unigram = sum(unigram.values()) or 1
        self.fallback_chars = [ch for ch, _ in ranked_unigram]
        self.fallback_probs = {ch: cnt / total_unigram for ch, cnt in ranked_unigram}

    def predict_ranked_chars(self, context: str, k: int) -> list[str]:
        # Interpolated decoder: combine evidence from many n-gram orders.
        pad = self.START_TOKEN * (self.max_order - 1)
        text = pad + context

        scores = defaultdict(float)
        # probability mass not yet assigned to a specific order
        residual = 1.0

        for order in range(self.max_order, 0, -1):
            ctx_len = order - 1
            key = text[-ctx_len:] if ctx_len > 0 else ""

            ranked = self.counts.get(order, {}).get(key)
            total = self.context_totals.get(order, {}).get(key)
            if not ranked:
                continue

            # trust strong contexts more where larger totals => larger lambda
            lam = total / (total + self.interp_alpha)
            weight = residual * lam
            # add weighted conditional probability for each candidate char
            for ch, count in ranked:
                scores[ch] += weight * (count / total)
            # pass remaining mass to shorter contexts
            residual *= 1.0 - lam

        # any leftover mass goes to unigram fallback
        for ch in self.fallback_chars[: self.fallback_limit]:
            scores[ch] += residual * self.fallback_probs.get(ch, 0.0)

        # return top-k unique characters by final score
        ranked_chars = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        guesses = [ch for ch, _ in ranked_chars[:k]]

        # pad top-k guess with spaces if insufficient
        while len(guesses) < k:
            guesses.append(" ")

        return guesses

    def to_state(self) -> ModelState:
        # Serialize all learned tables + runtime hyperparameters.
        return {
            "max_order": self.max_order,
            "top_next_chars": self.top_next_chars,
            "interp_alpha": self.interp_alpha,
            "fallback_limit": self.fallback_limit,
            "counts": self.counts,
            "context_totals": self.context_totals,
            "fallback_chars": self.fallback_chars,
            "fallback_probs": self.fallback_probs,
        }

    @classmethod
    def from_state(cls, state: ModelState) -> CharNGramModel:
        # Rebuild model object from serialized checkpoint payload.
        model = cls(
            max_order=state["max_order"],
            top_next_chars=state["top_next_chars"],
            interp_alpha=state["interp_alpha"],
            fallback_limit=state["fallback_limit"],
        )
        model.counts = state["counts"]
        model.context_totals = state["context_totals"]
        model.fallback_chars = state["fallback_chars"]
        model.fallback_probs = state["fallback_probs"]
        return model
