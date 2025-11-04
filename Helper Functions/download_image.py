# download_image.py
"""
Generic image downloader (works as CLI and importable module).

Features:
- Accepts a search term (Wikimedia Commons; optional LoremFlickr fallback) OR a direct image URL.
- Validates JPEG using magic bytes (0xFF 0xD8).
- CLI flags for output path/dir, filename, retries, timeouts, quiet mode, and seed.
- Importable function: download_image(...)

Usage (CLI):
    python download_image.py --term "mountain lake" --out-dir images
    python download_image.py --url "https://example.com/photo.jpg" --out-file ./photo.jpg
    python download_image.py --term cat --max-results 100 --quiet

Usage (Python / Notebook):
    from download_image import download_image
    path = download_image(term="pizza", out_dir="images", quiet=False)
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import requests

# --- Endpoints & headers ---
# Use the Wikimedia Commons MediaWiki Action API (NOT the Wikipedia Featured Feed).
WIKI_API = "https://commons.wikimedia.org/w/api.php"

# Public read access does NOT require Authorization. Use a descriptive UA only.
HEADERS = {
    "User-Agent": "Mini Food Vision Model (anirguha@hotmail.com)"
}

JPEG_MAGIC_PREFIX = b"\xff\xd8"  # JPEG files start with 0xFF 0xD8


# ------------------------- Helpers: find images -------------------------- #
def find_commons_jpg(term: str, max_results: int = 50, timeout: int = 20) -> Optional[str]:
    """
    Search Wikimedia Commons (File: namespace) for images related to `term` and return a random JPG URL.
    Returns None if nothing found or on error.
    """
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"{term} filetype:bitmap",
        "gsrnamespace": 6,                 # File: namespace
        "gsrlimit": min(50, max_results),  # non-bot cap is usually 50
        "prop": "imageinfo",
        "iiprop": "url|mime",
        "redirects": 1,
    }
    try:
        r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None

    pages = data.get("query", {}).get("pages", {}) or {}
    jpgs = []
    for p in pages.values():
        infos = p.get("imageinfo", []) or []
        for ii in infos:
            url = ii.get("url")
            mime = (ii.get("mime") or "").lower()
            if not url:
                continue
            if url.lower().endswith((".jpg", ".jpeg")) or "jpeg" in mime:
                jpgs.append(url)

    return random.choice(jpgs) if jpgs else None


def fallback_loremflickr(term: str, width: int = 1200, height: int = 800) -> str:
    """Simple fallback random JPG provider. Not guaranteed license-friendly for redistribution."""
    safe_term = (term or "random").replace(" ", ",")
    return f"https://loremflickr.com/{width}/{height}/{safe_term}.jpg"


# ------------------------- Helpers: download/validate -------------------- #
def _stream_download(url: str, path: Path, timeout: int = 30, chunk_size: int = 8192, headers=None) -> None:
    # Use a minimal UA-only header for image hosts/CDNs.
    headers = headers or {"User-Agent": HEADERS["User-Agent"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, headers=headers, timeout=timeout, stream=True) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)


def _is_jpeg(path: Path) -> bool:
    """Check JPEG magic bytes (and a quick content-type sniff via file extension)."""
    try:
        with open(path, "rb") as f:
            prefix = f.read(2)
            return prefix.startswith(JPEG_MAGIC_PREFIX)
    except Exception:
        return False


def _head_content_type_is_jpeg(url: str, timeout: int = 10) -> bool:
    """Use HEAD to inspect Content-Type (best-effort; some servers don't honor HEAD)."""
    try:
        r = requests.head(
            url,
            headers={"User-Agent": HEADERS["User-Agent"]},
            timeout=timeout,
            allow_redirects=True,
        )
        ctype = (r.headers.get("Content-Type") or "").lower()
        return ("jpeg" in ctype) or ("jpg" in ctype) or ("image/" in ctype)
    except Exception:
        return False


# ------------------------- Public API ---------------------------------- #
def download_image(
    term: Optional[str] = None,
    *,
    url: Optional[str] = None,
    out_dir: str | Path = ".",
    out_file: Optional[str | Path] = None,
    max_results: int = 50,
    retries: int = 2,
    seed: Optional[int] = None,
    timeout_commons: int = 20,
    timeout_download: int = 30,
    quiet: bool = False,
    allow_fallback: bool = False,
) -> Path:
    """
    Download an image either from `url` (direct) or by searching `term` (Commons -> optional fallback).
    Returns Path to saved file.

    Only JPGs are accepted/validated (magic bytes). Raises exceptions on unrecoverable errors.
    """
    if seed is not None:
        random.seed(seed)

    if (url is None) and (term is None):
        raise ValueError("Either 'url' or 'term' must be provided.")

    # Determine final save path
    out_dir = Path(out_dir)
    if out_file:
        save_path = Path(out_file)
        if not save_path.is_absolute():
            save_path = out_dir / save_path
    else:
        # Default filename based on term (if present) or last part of URL
        if url:
            filename = Path(url.split("?")[0]).name or "download.jpg"
            if not filename.lower().endswith((".jpg", ".jpeg")):
                filename += ".jpg"
        else:
            filename = f"{term.strip().replace(' ', '_')}.jpg"
        save_path = out_dir / filename

    # Build download candidates
    download_candidates: list[Tuple[str, str]] = []  # (url, source)

    if url:
        download_candidates.append((url, "direct"))

    if term:
        try:
            commons_url = find_commons_jpg(term, max_results=max_results, timeout=timeout_commons)
            if commons_url:
                download_candidates.append((commons_url, "Wikimedia Commons"))
        except Exception:
            # ignore and proceed to fallback candidates (if enabled)
            pass
        if allow_fallback:
            download_candidates.append((fallback_loremflickr(term), "LoremFlickr"))

    if not download_candidates:
        raise RuntimeError("No download candidates available (no URL provided and Commons search returned nothing).")

    last_exc = None
    for candidate_url, source in download_candidates:
        if not quiet:
            print(f"\u2139\uFE0F  Attempting: {source} -> {candidate_url}")
        # Quick HEAD check for non-jpg extensions
        if not candidate_url.lower().endswith((".jpg", ".jpeg")):
            if not _head_content_type_is_jpeg(candidate_url):
                if not quiet:
                    print("\u26A0\uFE0F  URL doesn't look like a JPEG by HEAD content-type; still attempting download.")

        # Try multiple times if allowed
        attempt = 0
        while attempt <= retries:
            try:
                _stream_download(candidate_url, save_path, timeout=timeout_download)
                # Validate magic bytes
                if not _is_jpeg(save_path):
                    raise ValueError("Downloaded file does not appear to be a JPEG (bad magic bytes).")
                if not quiet:
                    print(f"\u2705  Saved: {save_path} (source: {source}; attempts: {attempt+1})")
                return save_path
            except Exception as e:
                last_exc = e
                attempt += 1
                if attempt <= retries:
                    if not quiet:
                        print(f"\u27F3  Failed attempt {attempt} for {candidate_url}: {e}; retrying...")
                    time.sleep(1.0 + attempt * 0.5)
                else:
                    if not quiet:
                        print(f"\u274C  Giving up on {candidate_url}: {e}")
                    break

    # If we got here, all candidates failed
    raise RuntimeError(f"All download attempts failed. Last error: {last_exc}")


# --------------------------- CLI setup ----------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generic JPG downloader (Commons fallback + direct URL).")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("term", nargs="?", help="Search term for image lookup.")
    group.add_argument("--url", help="Direct image URL.")
    p.add_argument("--out-dir", default=".", help="Output directory.")
    p.add_argument("--out-file", default=None, help="Explicit output filename.")
    p.add_argument("--max-results", type=int, default=60, help="Max Commons results.")
    p.add_argument("--retries", type=int, default=2, help="Retries per candidate URL.")
    p.add_argument("--seed", type=int, default=None, help="Random seed.")
    p.add_argument("--timeout-commons", type=int, default=20, help="Timeout for Commons API calls.")
    p.add_argument("--timeout-download", type=int, default=30, help="Timeout for downloads.")
    p.add_argument("--quiet", action="store_true", help="Suppress output.")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        path = download_image(
            args.term,
            url=args.url,
            out_dir=args.out_dir,
            out_file=args.out_file,
            max_results=args.max_results,
            retries=args.retries,
            seed=args.seed,
            timeout_commons=args.timeout_commons,
            timeout_download=args.timeout_download,
            quiet=args.quiet,
        )
        if not args.quiet:
            print(f"\u2705 Done → {path}")
        return 0
    except Exception as e:
        print(f"u1F4A3 {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
