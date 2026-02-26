from __future__ import annotations

import csv
import glob
import os
import unicodedata

from config.constants import LANGUAGES, SPACE_RE
from identification.language_identifier import LanguageIdentifier

BOOTSTRAP_INPUT = "data/raw/open-dev/input.txt"
BOOTSTRAP_LANG = "data/raw/open-dev/lang.txt"
COMMON_ROOT = "data/processed/common"
DOMAIN_ROOT = "data/raw/domain"
TRAIN_CSV = "data/raw/train/train.csv"


def normalize_line(text: str) -> str | None:
    text = unicodedata.normalize("NFC", text)
    text = SPACE_RE.sub(" ", text.strip())
    return text or None


def add_line(by_lang: dict[str, list[str]], lang: str, raw_text: str) -> None:
    text = normalize_line(raw_text.rstrip("\n"))
    if text is None:
        return
    by_lang[lang].append(text)


def load_training_data() -> dict[str, list[str]]:
    by_lang = {lang: [] for lang in LANGUAGES}

    with open(BOOTSTRAP_INPUT, "rt", encoding="utf-8", errors="ignore") as fin, open(
        BOOTSTRAP_LANG, "rt", encoding="utf-8", errors="ignore"
    ) as flang:
        for raw_text, raw_lang in zip(fin, flang):
            lang = raw_lang.strip().lower()
            if lang not in by_lang:
                continue
            add_line(by_lang, lang, raw_text)

    for root in (COMMON_ROOT, DOMAIN_ROOT):
        if not os.path.isdir(root):
            continue
        for lang in LANGUAGES:
            pattern = os.path.join(root, "**", lang, "**", "*.txt")
            for path in sorted(glob.glob(pattern, recursive=True)):
                with open(path, "rt", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        add_line(by_lang, lang, line)

    if os.path.isfile(TRAIN_CSV):
        # bootstrap weak labels for train.csv from already labeled corpora.
        csv_lang_id = LanguageIdentifier(alpha=0.1)
        csv_lang_id.train(by_lang)

        with open(TRAIN_CSV, "rt", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                context = row.get("context", "")
                prediction = row.get("prediction", "")
                if context is None:
                    continue
                if prediction is None:
                    prediction = ""
                text = f"{context}{prediction}"
                lang = csv_lang_id.infer(text)
                add_line(by_lang, lang, text)

    return by_lang
