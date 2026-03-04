from __future__ import annotations

import csv
import glob
import os
import unicodedata
from typing import Iterator

from config.constants import LANGUAGES, SPACE_RE
from identification.language_identifier import LanguageIdentifier

BOOTSTRAP_INPUT = "data/raw/open-dev/input.txt"
BOOTSTRAP_LANG = "data/raw/open-dev/lang.txt"
COMMON_ROOT = "data/processed/common"
DOMAIN_ROOT = "data/raw/domain"
KAGGLE_TRAIN_CSV = "data/raw/train/train.csv"
KAGGLE_TEST_CSV = "data/raw/train/kaggle.csv"
LANG_ID_ALPHA = 0.1

LanguageBuckets = dict[str, list[str]]


def normalize_line(text: str) -> str | None:
    """Normalize whitespace/Unicode and drop empty lines."""
    text = unicodedata.normalize("NFC", text)
    text = SPACE_RE.sub(" ", text.strip())
    return text or None


def add_line(by_lang: LanguageBuckets, lang: str, raw_text: str) -> None:
    """Append a line to a language bucket after normalization."""
    if lang not in by_lang:
        return
    text = normalize_line(raw_text)
    if text is None:
        return
    by_lang[lang].append(text)


def build_language_identifier(by_lang: LanguageBuckets) -> LanguageIdentifier:
    """Train a language identifier from current labeled buckets."""
    lang_id = LanguageIdentifier(alpha=LANG_ID_ALPHA)
    lang_id.train(by_lang)
    return lang_id


def iter_csv_rows(csv_path: str) -> Iterator[dict[str, str | None]]:
    """Yield rows from a CSV as dictionaries."""
    with open(csv_path, "rt", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def load_bootstrap_data(by_lang: LanguageBuckets) -> None:
    """Load line-level supervision from open-dev input + language labels."""
    with open(BOOTSTRAP_INPUT, "rt", encoding="utf-8", errors="ignore") as fin, open(
        BOOTSTRAP_LANG, "rt", encoding="utf-8", errors="ignore"
    ) as flang:
        for raw_text, raw_lang in zip(fin, flang):
            lang = raw_lang.strip().lower()
            add_line(by_lang, lang, raw_text)


def load_text_corpora(by_lang: LanguageBuckets) -> None:
    """Load language-sharded text corpora from processed/raw folders."""
    for root in (COMMON_ROOT, DOMAIN_ROOT):
        for lang in LANGUAGES:
            pattern = os.path.join(root, "**", lang, "**", "*.txt")
            for path in sorted(glob.glob(pattern, recursive=True)):
                with open(path, "rt", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        add_line(by_lang, lang, line)


def load_kaggle_train_csv(by_lang: LanguageBuckets) -> None:
    """Load pseudo-labeled examples from train.csv (context + prediction)."""
    lang_id = build_language_identifier(by_lang)
    for row in iter_csv_rows(KAGGLE_TRAIN_CSV):
        context = row.get("context", "")
        prediction = row.get("prediction", "")
        if context is None:
            continue
        if prediction is None:
            prediction = ""
        text = f"{context}{prediction}"
        lang = lang_id.infer(text)
        add_line(by_lang, lang, text)


def load_kaggle_test_csv(by_lang: LanguageBuckets) -> None:
    """Load pseudo-labeled adaptation contexts from kaggle.csv."""
    lang_id = build_language_identifier(by_lang)
    for row in iter_csv_rows(KAGGLE_TEST_CSV):
        context = row.get("context", "")
        if context is None:
            continue
        lang = lang_id.infer(context)
        add_line(by_lang, lang, context)


def load_training_data() -> LanguageBuckets:
    """Assemble the full multilingual training corpus used by the LM."""
    by_lang: LanguageBuckets = {lang: [] for lang in LANGUAGES}

    load_bootstrap_data(by_lang)
    load_text_corpora(by_lang)
    load_kaggle_train_csv(by_lang)
    load_kaggle_test_csv(by_lang)

    return by_lang
