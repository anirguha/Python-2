# download_image.py
"""
Generic image downloader (works as CLI and importable module).

Features:
- Accepts a search term (Wikimedia Commons -> LoremFlickr fallback) OR a direct image URL.
- Validates JPEG using Content-Type and JPEG magic bytes.
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

WIKI_API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {
    # Friendly User-Agent to avoid 403s from some image hosts / APIs
    "User-Agent": "GenericImageDownloader/1.0 (https://example.com; contact: you@example.com)"
}

JPEG_MAGIC_PREFIX = b"\xff\xd8"  # JPEG files start with 0xFF 0xD8


# ------------------------- Helpers: find images -------------------------- #
def find_commons_jpg(term: str, max_results: int = 60, timeout: int = 20) -> Optional[str]:
    """
    Search Wikimedia Commons for images related to `term` and return a random JPG URL.
    Returns None if nothing found or on error.
    """
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"{term} filetype:bitmap",
        "gsrnamespace": 6,
        "gsrlimit": max_results,
        "prop": "imageinfo",
        "iiprop": "url",
    }
    try:
        r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None

    jpgs = []
    for p in pages.values():
        infos = p.get("imageinfo", [])
        if not infos:
            continue
        url = infos[0].get("url")
        if url and url.lower().endswith((".jpg", ".jpeg")):
            jpgs.append(url)

    if not jpgs:
        return None
    return random.choice(jpgs)


def fallback_loremflickr(term: str, width: int = 1200, height: int = 800) -> str:
    """Simple fallback random JPG provider. Not guaranteed license-friendly for redistribution."""
    return f"https://loremflickr.com/{width}/{height}/{term}.jpg"


# ------------------------- Helpers: download/validate -------------------- #
def _stream_download(url: str, path: Path, timeout: int = 30, chunk_size: int = 8192, headers=None) -> None:
    headers = headers or HEADERS
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
        r = requests.head(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        ctype = r.headers.get("Content-Type", "").lower()
        return "jpeg" in ctype or "jpg" in ctype or "image/" in ctype
    except Exception:
        return False


# ------------------------- Public API ---------------------------------- #
def download_image(
    term: Optional[str] = None, 
    *,
    url: Optional[str] = None,
    out_dir: str | Path = ".",
    out_file: Optional[str | Path] = None,
    max_results: int = 60,
    retries: int = 2,
    seed: Optional[int] = None,
    timeout_commons: int = 20,
    timeout_download: int = 30,
    quiet: bool = False,
) -> Path:
    """
    Download an image either from `url` (direct) or by searching `term` (Commons -> fallback).
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
        else:
            # sanitize term -> replace spaces with underscores
            filename = f"{term.strip().replace(' ', '_')}.jpg"
        save_path = out_dir / filename

    # If direct url provided, try that first
    download_candidates: list[Tuple[str, str]] = []  # (url, source)
    if url:
        download_candidates.append((url, "direct"))

    # If term provided, attempt commons then fallback
    
    try:
        commons_url = find_commons_jpg(term, max_results=max_results, timeout=timeout_commons)
        if commons_url:
            download_candidates.append((commons_url, "Wikimedia Commons"))
    except Exception:
        # ignore and proceed to fallback
        pass
    # fallback service (random)
    download_candidates.append((fallback_loremflickr(term), "LoremFlickr"))

    last_exc = None
    for candidate_url, source in download_candidates:
        if not quiet:
            print(f"[info] Attempting: {source} -> {candidate_url}")
        # Skip non-jpg URLs quickly if they obviously don't end with jpg/jpeg
        if not candidate_url.lower().endswith((".jpg", ".jpeg")):
            # We'll still try (some services redirect to .jpg), but add a HEAD check.
            if not _head_content_type_is_jpeg(candidate_url):
                if not quiet:
                    print("[warn] URL doesn't look like a JPEG by HEAD content-type; still attempting download.")
        # Try multiple times if allowed
        attempt = 0
        while attempt <= retries:
            try:
                _stream_download(candidate_url, save_path, timeout=timeout_download)
                # Validate magic bytes
                if not _is_jpeg(save_path):
                    raise ValueError("Downloaded file does not appear to be a JPEG (bad magic bytes).")
                if not quiet:
                    print(f"[ok] Saved: {save_path} (source: {source}; attempts: {attempt+1})")
                return save_path
            except Exception as e:
                last_exc = e
                attempt += 1
                if attempt <= retries:
                    if not quiet:
                        print(f"[retry] Failed attempt {attempt} for {candidate_url}: {e}; retrying...")
                    time.sleep(1.0 + attempt * 0.5)
                else:
                    if not quiet:
                        print(f"[error] Giving up on {candidate_url}: {e}")
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
            print(f"Done → {path}")
        return 0
    except Exception as e:
        print(f"[fatal] {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
