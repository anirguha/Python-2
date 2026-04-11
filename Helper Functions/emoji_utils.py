"""
emoji_utils.py — tiny helpers for emoji + colored CLI output

- Toggle emoji on/off at runtime (set_use_emoji)
- Fetch icons by key (icon / EMOJI[...] )
- Simple ANSI color helper (color) with auto-disable on Windows old consoles
- Safe to use with or without external deps (no required packages)

Usage:
    from emoji_utils import EMOJI, icon, set_use_emoji, color

    set_use_emoji(True)  # or False to disable all emoji
    print(f"{icon('success')} {color('All good', 'green', bold=True)}")

"""

from __future__ import annotations

import os
import sys
from typing import Dict

# ---- Emoji toggle -----------------------------------------------------------

USE_EMOJI: bool = True  # default; caller may change via set_use_emoji()


def set_use_emoji(flag: bool=True) -> None:
    """Enable/disable emoji globally."""
    global USE_EMOJI
    USE_EMOJI = bool(flag)


def _e(s: str) -> str:
    """Return s if emoji enabled else empty string."""
    return s if USE_EMOJI else ""


# ---- Common emoji dictionary ------------------------------------------------

EMOJI: Dict[str, str] = {
    # Status
    "success": _e("✅"),
    "ok": _e("✔️"),
    "error": _e("❌"),
    "fail": _e("❌"),
    "warning": _e("⚠️"),
    "info": _e("ℹ️"),
    "stop": _e("🛑"),
    "hourglass": _e("⏳"),
    "spinner": _e("🔄"),

    # Files / Folders / IO
    "file": _e("📄"),
    "folder": _e("📁"),
    "open_folder": _e("📂"),
    "log": _e("📝"),
    "backup": _e("📦"),
    "inbox": _e("📥"),
    "outbox": _e("📤"),
    "search": _e("🔍"),
    "link": _e("🔗"),

    # Actions
    "add": _e("➕"),
    "remove": _e("➖"),
    "edit": _e("✏️"),
    "undo": _e("🔁"),
    "refresh": _e("🔄"),
    "arrow": _e("➜"),
}


def icon(key: str, default: str = "") -> str:
    """Return emoji for key or default; respects USE_EMOJI toggle."""
    if not USE_EMOJI:
        return ""
    return EMOJI.get(key, default)


# ---- Minimal color helper (no external deps) --------------------------------

# ANSI color codes (widely supported; on very old Windows, they may be ignored)
_ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
}


def _supports_ansi() -> bool:
    # Heuristic: disable if not a TTY or TERM=dumb
    if os.environ.get("TERM", "") == "dumb":
        return False
    # Modern Windows 10+ terminals support ANSI; fallback otherwise
    if os.name == "nt":
        # If running in newer terminals (Windows Terminal/VSCode), it's OK.
        # Otherwise, honor PY_COLORS=0 to disable.
        return os.environ.get("PY_COLORS", "1") != "0"
    return sys.stdout.isatty()


_ENABLE_ANSI = _supports_ansi()


def set_use_color(flag: bool) -> None:
    """Enable/disable ANSI coloring globally (independent of emoji)."""
    global _ENABLE_ANSI
    _ENABLE_ANSI = bool(flag)


def color(text: str, fg: str | None = None, *, bold: bool = False, dim: bool = False) -> str:
    """Return text wrapped with ANSI color codes (no external deps)."""
    if not _ENABLE_ANSI or not sys.stdout:
        return text
    seq = ""
    if bold:
        seq += _ANSI["bold"]
    if dim:
        seq += _ANSI["dim"]
    if fg and fg in _ANSI:
        seq += _ANSI[fg]
    if not seq:
        return text
    return f"{seq}{text}{_ANSI['reset']}"


# Convenience wrappers
def green(text: str, **kw) -> str:
    return color(text, "green", **kw)


def yellow(text: str, **kw) -> str:
    return color(text, "yellow", **kw)


def red(text: str, **kw) -> str:
    return color(text, "red", **kw)


def blue(text: str, **kw) -> str:
    return color(text, "blue", **kw)


__all__ = [
    "USE_EMOJI",
    "set_use_emoji",
    "EMOJI",
    "icon",
    "set_use_color",
    "color",
    "green",
    "yellow",
    "red",
    "blue",
]
