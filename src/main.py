from __future__ import annotations

from argparse import ArgumentParser

from data.training_data import BOOTSTRAP_INPUT, load_training_data
from models.language_aware_lm import LanguageAwareNGramLM

# model defaults
MAX_ORDER = 8
TOP_NEXT_CHARS = 16
INTERP_ALPHA = 1.5

# CLI defaults
DEFAULT_WORK_DIR = "work_lm"
DEFAULT_TEST_OUTPUT = "pred_n_gram_lm.txt"


def run_train(work_dir: str) -> None:
    print("Loading training data")
    by_lang = load_training_data()
    for lang in by_lang:
        print(
            f"[{lang}] lines={len(by_lang[lang])}"
        )
    print("Training language-aware n-gram model")
    model = LanguageAwareNGramLM(MAX_ORDER, TOP_NEXT_CHARS, INTERP_ALPHA)
    model.train(by_lang)
    print("Saving checkpoint")
    model.save(work_dir)


def run_test(work_dir: str, test_data: str, test_output: str) -> None:
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


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        run_train(args.work_dir)
    else:
        run_test(args.work_dir, args.test_data, args.test_output)


if __name__ == "__main__":
    main()
