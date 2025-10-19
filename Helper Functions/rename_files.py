import os
import urllib.parse
from pathlib import Path
import argparse

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
        # Decode URL-encoded characters
        decoded_name = urllib.parse.unquote(original_name)

        print(f"original name: {original_name}")
        print(f"Decoded name: {decoded_name}")

        # Optionally remove first N characters (guard against over-slicing)
        if slice_start > 0 and len(decoded_name) > slice_start:
            decoded_name = decoded_name[slice_start:]

        # If name didn't change, skip
        if decoded_name == original_name:
            print("Name didn't change....skipping")
            continue

        if decoded_name == "":
            print(f"⚠ Skipping (empty target name) for: {original_name}")
            continue

        new_path = p.with_name(decoded_name)

        # Prevent collisions
        if new_path.exists():
            print(f"⚠ Skipping (target exists): {decoded_name}")
            continue

        # Perform rename
        p.rename(new_path)
        print(f"Renamed: {original_name} → {decoded_name}")
        cnt += 1

    print(f"✅ Total files renamed: {cnt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename URL-encoded files in a folder")
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the folder")
    parser.add_argument("--file_pattern", type=str, default="*", help="Glob to select files (e.g., '*.txt')")
    parser.add_argument("--slice_start", type=int, default=0,
                        help="Optional: remove first N characters from filename (after decoding)")

    args = parser.parse_args()
    rename_files(args.folder_path, args.file_pattern, args.slice_start)
