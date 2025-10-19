# --- Public API (callable from other Python code) ----------------------------
def run(folder_path,
        file_pattern="*",
        slice_start=0,
        *,
        recursive=False,
        preserve_ext=False,
        decode=True,
        dry_run=False,
        log_file=None,
        replacements=None,
        use_regex=False) -> int:
    """
    Programmatic entry point. Call this from Python:
        import rename_files_2
        rename_files_2.run("/path", "ABC*", 28, preserve_ext=True, dry_run=True)
    """
    from pathlib import Path
    return rename_files(
        folder_path=Path(folder_path),
        file_pattern=file_pattern,
        slice_start=slice_start,
        recursive=recursive,
        preserve_ext=preserve_ext,
        decode=decode,
        dry_run=dry_run,
        log_file=Path(log_file) if log_file else None,
        replacements=replacements,
        use_regex=use_regex,
    )

# --- CLI entry point (parses argv like a script) ----------------------------
def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Rename files by decoding URL-encoded names, slicing, and replacing patterns."
    )

    # Positional args (your preferred style)
    ap.add_argument("folder_path", type=str, help="Path to the base folder")
    ap.add_argument("file_pattern", type=str, help='Glob pattern, e.g. "*.mp4" or "ABC*"')
    ap.add_argument("slice_start", type=int, nargs="?", default=0,
                    help="Remove first N chars (after optional decoding). Default: 0")

    # Optional flags (can be combined with positionals)
    ap.add_argument("--recursive", action="store_true", help="Search subfolders recursively")
    ap.add_argument("--preserve_ext", action="store_true",
                    help="When slicing, keep extension(s) intact")
    decode_group = ap.add_mutually_exclusive_group()
    decode_group.add_argument("--decode", dest="decode", action="store_true",
                              help="Decode URL-encoded characters (default)")
    decode_group.add_argument("--no-decode", dest="decode", action="store_false",
                              help="Do not decode; only apply slicing/replacements")
    ap.set_defaults(decode=True)
    ap.add_argument("--dry_run", action="store_true",
                    help="Show changes without performing renames")
    ap.add_argument("--log_file", type=str, default=None,
                    help="Optional path to write a log of renames")
    ap.add_argument("--replace", nargs=2, metavar=("PATTERN", "REPLACEMENT"),
                    action="append", default=None,
                    help="Replace PATTERN with REPLACEMENT (repeatable). Use --regex to enable regex.")
    ap.add_argument("--regex", action="store_true",
                    help="Interpret PATTERN as a regular expression for --replace")

    args = ap.parse_args(argv)

    return run(
        folder_path=args.folder_path,
        file_pattern=args.file_pattern,
        slice_start=args.slice_start,
        recursive=args.recursive,
        preserve_ext=args.preserve_ext,
        decode=args.decode,
        dry_run=args.dry_run,
        log_file=args.log_file,
        replacements=args.replace,
        use_regex=args.regex,
    )

# Optional: make intent clear when importing from other code
__all__ = ["rename_files", "run", "main"]

# --- Script entry (when executed directly) -----------------------------------
if __name__ == "__main__":
    raise SystemExit(main())
