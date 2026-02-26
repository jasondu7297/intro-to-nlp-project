#!/usr/bin/env python3
"""Build multilingual Wikipedia corpus from a fixed list of specific pages."""

from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

from utils import write_lines


DEFAULT_LANGUAGES = ["en", "ru", "zh", "ja", "hi", "ar", "ko", "fr", "de", "it"]
SEED_LANGUAGE = "en"
SPACE_RE = re.compile(r"\s+")
# Include Devanagari danda and Arabic punctuation so Hindi/Arabic text splits properly.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？؟۔।॥])\s*")
CLAUSE_SPLIT_RE = re.compile(r"(?<=[,;:،，、])\s*")
WIKI_API_TIMEOUT_SECS = 300
WIKI_API_RETRIES = 3
WIKI_HEADERS = {
    "User-Agent": "cs489-wikipedia-corpus-builder/1.0 (student project)",
    "Accept": "application/json",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download exact Wikipedia pages and ingest text by language.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        type=str.lower,
        default=DEFAULT_LANGUAGES,
        metavar="LANG",
        help="Target language codes (space-separated).",
    )
    parser.add_argument(
        "--page_list",
        default="data/raw/seed/wikipedia_space_pages.txt",
        help="Path to newline-delimited seed page titles in source language.",
    )
    parser.add_argument(
        "--output_root",
        default="data/processed/common",
        help="Output root for per-language wikipedia.txt files.",
    )
    parser.add_argument(
        "--min_chars",
        type=int,
        default=15,
        help="Minimum line length after normalization.",
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=500,
        help="Maximum line length after normalization.",
    )
    parser.add_argument(
        "--max_lines_per_article",
        type=int,
        default=1000,
        help="Maximum output lines per article.",
    )
    return parser.parse_args()


def load_wiki_titles(path: str) -> list[str]:
    """Load non-empty seed Wikipedia titles from a newline-delimited file."""
    titles = []
    with open(path, "rt", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line in titles:
                continue
            titles.append(line)
    return titles


def wiki_api_get(lang: str, params: dict[str, str]) -> dict[str, object]:
    """Execute a Wikipedia API request and return decoded JSON."""
    query = urllib.parse.urlencode(params)
    url = f"https://{lang}.wikipedia.org/w/api.php?{query}"
    req = urllib.request.Request(url, headers=WIKI_HEADERS)
    insecure_ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(req, timeout=WIKI_API_TIMEOUT_SECS, context=insecure_ctx) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _safe_wiki_api_get(lang: str, params: dict[str, str]) -> dict[str, object] | None:
    """Return API payload or None when the request/JSON parsing fails."""
    for attempt in range(1, WIKI_API_RETRIES + 1):
        try:
            return wiki_api_get(lang, params)
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
            if attempt >= WIKI_API_RETRIES:
                return None
            time.sleep(0.5 * (2 ** (attempt - 1)))
    return None


def _iter_query_pages(payload: dict[str, object] | None) -> list[dict[str, object]]:
    """Extract page dictionaries from a MediaWiki query response."""
    if payload is None:
        return []
    query = payload.get("query")
    if not isinstance(query, dict):
        return []
    pages = query.get("pages")
    if not isinstance(pages, dict):
        return []
    return [page for page in pages.values() if isinstance(page, dict)]


def resolve_title_for_language(source_title: str, target_lang: str) -> str | None:
    """Resolve a seed title into its target-language equivalent."""
    if target_lang == SEED_LANGUAGE:
        return source_title

    params = {
        "action": "query",
        "format": "json",
        "redirects": "1",
        "titles": source_title,
        "prop": "langlinks",
        "lllang": target_lang,
        "lllimit": "1",
    }

    payload = _safe_wiki_api_get(SEED_LANGUAGE, params)
    for page in _iter_query_pages(payload):
        langlinks = page.get("langlinks")
        if not isinstance(langlinks, list):
            continue
        for langlink in langlinks:
            title = langlink.get("*")
            if not isinstance(title, str):
                continue
            title = title.strip()
            if title:
                return title
    return None


def fetch_extract(title: str, lang: str) -> dict[str, object] | None:
    """Fetch the first non-empty plaintext extract for a page title."""
    params = {
        "action": "query",
        "format": "json",
        "redirects": "1",
        "titles": title,
        "prop": "extracts",
        "explaintext": "1",
        "exsectionformat": "plain",
    }

    payload = _safe_wiki_api_get(lang, params)
    # find the first page with non-empty content
    for page in _iter_query_pages(payload):
        pageid = page.get("pageid")
        extract = page.get("extract")
        if not isinstance(extract, str) or not extract.strip():
            continue
        page_title = page.get("title") or title
        return {"pageid": pageid, "title": page_title, "extract": extract}
    return None


def normalize_line(text: str, min_chars: int, max_chars: int) -> str | None:
    """Normalize and length-filter a sentence candidate."""
    line = unicodedata.normalize("NFC", text).replace("\u00a0", " ")
    line = SPACE_RE.sub(" ", line).strip()
    if len(line) < min_chars or len(line) > max_chars:
        return None
    return line


def split_overlong_text(text: str, max_chars: int) -> list[str]:
    """Split oversized sentence candidates into max_chars-bounded chunks."""
    normalized = SPACE_RE.sub(" ", text).strip()
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    chunks = []
    clauses = [part.strip() for part in CLAUSE_SPLIT_RE.split(normalized) if part.strip()]
    if not clauses:
        clauses = [normalized]

    current = ""
    for clause in clauses:
        candidate = clause if not current else f"{current} {clause}"
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(clause) <= max_chars:
            current = clause
            continue

        words = clause.split()
        word_chunk = ""
        for word in words:
            word_candidate = word if not word_chunk else f"{word_chunk} {word}"
            if len(word_candidate) <= max_chars:
                word_chunk = word_candidate
                continue

            if word_chunk:
                chunks.append(word_chunk)
                word_chunk = ""

            if len(word) <= max_chars:
                word_chunk = word
                continue

            # Hard fallback for extremely long tokens.
            for i in range(0, len(word), max_chars):
                chunks.append(word[i : i + max_chars])

        if word_chunk:
            current = word_chunk

    if current:
        chunks.append(current)

    return chunks


def extract_lines(extract: str, min_chars: int, max_chars: int, max_lines_per_article: int) -> list[str]:
    """Split an article extract into normalized sentence lines."""
    lines = []
    paragraphs = [part.strip() for part in extract.splitlines() if part.strip()]
    for paragraph in paragraphs:
        sentences = SENTENCE_SPLIT_RE.split(paragraph)
        for sentence in sentences:
            # for candidate in split_overlong_text(sentence, max_chars=max_chars):
            normalized = normalize_line(sentence, min_chars=min_chars, max_chars=max_chars)
            if normalized is None:
                continue
            lines.append(normalized)
            if len(lines) >= max_lines_per_article:
                return lines
    return lines


def main() -> None:
    """Download and write per-language Wikipedia corpora."""
    args = parse_args()
    languages = args.languages

    seed_titles = load_wiki_titles(args.page_list)

    for lang in languages:
        seen_page_ids = set()
        seen_lines = set()
        output_lines = []

        for seed_title in seed_titles:
            resolved_title = resolve_title_for_language(seed_title, lang)
            if resolved_title is None:
                continue
            article = fetch_extract(resolved_title, lang)
            if article is None:
                continue
            pageid = article["pageid"]
            if pageid in seen_page_ids:
                continue
            seen_page_ids.add(pageid)

            for line in extract_lines(article["extract"], args.min_chars, args.max_chars, args.max_lines_per_article):
                if line in seen_lines:
                    continue
                seen_lines.add(line)
                output_lines.append(line)

        out_path = os.path.join(args.output_root, lang, "wikipedia.txt")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        write_lines(out_path, output_lines)

        print(
            f"[{lang}] seeds={len(seed_titles)} pages={len(seen_page_ids)} "
            f"lines={len(output_lines)} output={out_path}"
        )


if __name__ == "__main__":
    main()
