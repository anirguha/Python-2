
#rename_files_cli.py

import os
import re
import urllib.parse
from pathlib import Path
import argparse


def is_url_like_string(name: str) -> bool:
    """
    Heuristics to decide if a filename looks like it came from a URL or
    some encoded / 'dirty' source.
    """
    # Real URL-encoded bytes like %20, %2F, %3A etc.
    if re.search(r"%[0-9A-Fa-f]{2}", name):
        return True

    # Invalid filename characters (mostly Windows, but good heuristic)
    if re.search(r'[<>:"/\\|?*\x00-\x1F]', name):
        return True

    # "20" used as stand-in for space (our specific case)
    if re.search(r"(?<!\d)20(?!\d)", name):
        return True

    # '+' often used as space in URLs
    if "+" in name:
        return True

    return False


def replace_encoded_20(s: str) -> str:
    """
    Replace '20' with a space only when it is NOT part of a number.

    Examples:
      ABC20JAPANESE20HOUSEWIFE → ABC JAPANESE HOUSEWIFE
      Movie20Trailer           → Movie Trailer
      Top20Songs               → Top20Songs   (unchanged)
      Holiday2020              → Holiday2020  (unchanged)
    """
    s = re.sub(r'(?<!\d)20(?!\d)', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def clean_encoded_name(original_name: str, slice_start: int = 0) -> str:
    """
    Takes an original filename and returns a cleaned version:
    - URL-decodes if needed
    - replaces encoded '20' with spaces safely
    - collapses multiple spaces
    - optionally slices off first N chars
    """
    name = original_name

    # If it looks URL-encoded, try a decode first (handles %20, +, etc.)
    if re.search(r"%[0-9A-Fa-f]{2}", name) or "+" in name:
        name = urllib.parse.unquote(name)

    # Safely handle '20' acting like space
    name = replace_encoded_20(name)

    # Optional slicing from the start (if requested and safe)
    if slice_start > 0 and len(name) > slice_start:
        name = name[slice_start:]

    # Final trim of whitespace
    name = name.strip()
    return name


def rename_files(folder_path: Path,
                 file_pattern: str,
                 slice_start: int = 0) -> None:
    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        print(f"Error: {folder} is not a valid folder.")
        return

    cnt = 0

    # Iterate deterministically for nicer logs
    for p in sorted(folder.glob(file_pattern)):
        # Skip directories
        if p.is_dir():
            continue

        original_name = p.name  # str

        # Only touch files that look URL-like / encoded / dirty
        if not is_url_like_string(original_name):
            continue

        cleaned_name = clean_encoded_name(original_name, slice_start=slice_start)

        # If name didn't change, skip
        if cleaned_name == original_name:
            continue

        # Guard against empty result
        if cleaned_name == "":
            print(f"⚠ Skipping (empty target name) for: {original_name}")
            continue

        new_path = p.with_name(cleaned_name)

        # Prevent collisions
        if new_path.exists():
            print(f"⚠ Skipping (target exists): {cleaned_name}")
            continue

        # Perform rename
        p.rename(new_path)
        print(f"Renamed: {original_name} → {cleaned_name}")
        cnt += 1

    print(f"✅ Total files renamed: {cnt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename URL-like / encoded filenames.")
    parser.add_argument("folder_path", type=Path, help="Folder containing files to rename")
    parser.add_argument(
        "--pattern", "-p",
        default="*",
        help="Glob pattern to match files (default: '*')"
    )
    parser.add_argument(
        "--slice-start", "-s",
        type=int,
        default=0,
        help="Optionally remove the first N characters from cleaned name"
    )

    args, _ = parser.parse_known_args()
    rename_files(args.folder_path, args.pattern, slice_start=args.slice_start)
