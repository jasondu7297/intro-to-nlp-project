import hashlib
import json
import random
from typing import Any, Iterator, Sequence

def load_json(path: str) -> dict[str, Any]:
    """Load a UTF-8 JSON file."""
    with open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def make_rng(seed: int, *parts: object) -> random.Random:
    """Create a deterministic RNG scoped by seed and context parts."""
    key = ":".join(str(x) for x in parts)
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    derived = (seed + int(digest[:8], 16)) % (2**32)
    return random.Random(derived)


def iter_file_lines(path: str) -> Iterator[str]:
    """Yield stripped lines from a UTF-8 text file."""
    with open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            yield line.rstrip("\n")


def write_lines(path: str, lines: Sequence[str]) -> None:
    """Write one example per line in UTF-8."""
    with open(path, "wt", encoding="utf-8") as f:
        for line in lines:
            f.write(line)
            f.write("\n")