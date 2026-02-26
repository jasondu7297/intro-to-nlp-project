#!/usr/bin/env python3
import csv
import glob
import os
import unicodedata
from argparse import ArgumentParser

from lm.language_aware_lm import LanguageAwareNGramLM
from lm.language_identifier import LanguageIdentifier
from lm.lm_constants import LANGUAGES, SPACE_RE

# data layout defaults (keep fixed for simplicity).
BOOTSTRAP_INPUT = "data/raw/open-dev/input.txt"
BOOTSTRAP_LANG = "data/raw/open-dev/lang.txt"
COMMON_ROOT = "data/processed/common"
DOMAIN_ROOT = "data/raw/domain"
TRAIN_CSV = "data/raw/train/train.csv"

# model defaults
MAX_ORDER = 8
TOP_NEXT_CHARS = 16
INTERP_ALPHA = 1.5

# CLI defaults
DEFAULT_WORK_DIR = "work_lm"
DEFAULT_TEST_OUTPUT = "pred_n_gram_lm.txt"


def normalize_line(text):
    text = unicodedata.normalize("NFC", text)
    text = SPACE_RE.sub(" ", text.strip())
    return text or None


def add_line(by_lang, lang, raw_text):
    text = normalize_line(raw_text.rstrip("\n"))
    if text is None:
        return False
    by_lang[lang].append(text)
    return True


def load_training_data():
    by_lang = {lang: [] for lang in LANGUAGES}
    from_bootstrap = {lang: 0 for lang in LANGUAGES}
    from_corpus = {lang: 0 for lang in LANGUAGES}
    from_train_csv = {lang: 0 for lang in LANGUAGES}

    with open(BOOTSTRAP_INPUT, "rt", encoding="utf-8", errors="ignore") as fin, open(
        BOOTSTRAP_LANG, "rt", encoding="utf-8", errors="ignore"
    ) as flang:
        for raw_text, raw_lang in zip(fin, flang):
            lang = raw_lang.strip().lower()
            if lang not in by_lang:
                continue
            if add_line(by_lang, lang, raw_text):
                from_bootstrap[lang] += 1

    for root in (COMMON_ROOT, DOMAIN_ROOT):
        if not os.path.isdir(root):
            continue
        for lang in LANGUAGES:
            pattern = os.path.join(root, "**", lang, "**", "*.txt")
            for path in sorted(glob.glob(pattern, recursive=True)):
                with open(path, "rt", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if add_line(by_lang, lang, line):
                            from_corpus[lang] += 1

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
                if add_line(by_lang, lang, text):
                    from_train_csv[lang] += 1

    return by_lang, from_bootstrap, from_corpus, from_train_csv


def run_train(work_dir):
    print("Loading training data")
    by_lang, from_bootstrap, from_corpus, from_train_csv = load_training_data()
    for lang in LANGUAGES:
        print(
            f"[{lang}] lines={len(by_lang[lang])} "
            f"(bootstrap={from_bootstrap[lang]}, corpus={from_corpus[lang]}, "
            f"train_csv={from_train_csv[lang]})"
        )
    print("Training language-aware n-gram model")
    model = LanguageAwareNGramLM(MAX_ORDER, TOP_NEXT_CHARS, INTERP_ALPHA)
    model.train(by_lang)
    print("Saving checkpoint")
    model.save(work_dir)


def run_test(work_dir, test_data, test_output):
    print("Loading model")
    model = LanguageAwareNGramLM.load(work_dir)
    print(f"Loading test data from {test_data}")
    with open(test_data, "rt", encoding="utf-8", errors="ignore") as f:
        contexts = [line.rstrip("\n") for line in f]
    print("Predicting")
    preds = model.predict(contexts)
    print(f"Writing predictions to {test_output}")
    with open(test_output, "wt", encoding="utf-8") as f:
        f.writelines(f"{p}\n" for p in preds)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("mode", choices=("train", "test"), help="what to run")
    parser.add_argument("--work_dir", default=DEFAULT_WORK_DIR, help="where to save model/checkpoint")
    parser.add_argument("--test_data", default=BOOTSTRAP_INPUT, help="path to test data")
    parser.add_argument("--test_output", default=DEFAULT_TEST_OUTPUT, help="path to write predictions")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "train":
        run_train(args.work_dir)
    else:
        run_test(args.work_dir, args.test_data, args.test_output)


if __name__ == "__main__":
    main()
